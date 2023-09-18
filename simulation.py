import collisions
from typing import Sequence
import jax
import jax.numpy as jnp
import typing


class State(typing.NamedTuple):
    time: float
    sphere_centers: jnp.ndarray
    water_heights: jnp.ndarray
    sphere_velocities: jnp.ndarray
    water_velocities: jnp.ndarray
    wave_speed: float
    body_heights: jnp.ndarray
    time_delta: float


class Simulator:
    GRAVITY_CONSTANT = -9.81
    POSITIONAL_DAMPING = 1.0
    VELOCITY_DAMPING = 0.3
    ALPHA = 0.5

    def __init__(
        self,
        spheres: Sequence[collisions.Sphere],
        rectanguloids: Sequence[collisions.Rectanguloid],
        n: int,
        m: int,
        spacing: float,
    ) -> None:
        self._spheres = jax.tree_map(lambda *x: jnp.stack(x), *spheres)
        self._stacked_rectanguloid = jax.tree_map(
            lambda *x: jnp.stack(x), *rectanguloids
        )
        self.update = jax.jit(self.update, inline=True)
        self._n = n
        self._m = m
        self._tank_size = (jnp.array([n, m], dtype=jnp.float32) * spacing) * 0.5
        self._spacing = spacing

    def init_state(self) -> State:
        return State(
            time=0.0,
            sphere_centers=self._spheres.center,
            water_heights=self._stacked_rectanguloid.corner1[:, 1],
            sphere_velocities=jnp.zeros_like(self._spheres.center),
            water_velocities=jnp.zeros_like(self._stacked_rectanguloid.corner1[:, 1]),
            wave_speed=2.0,
            body_heights=jnp.zeros_like(self._stacked_rectanguloid.corner1[:, 1]),
            time_delta=0.0,
        )

    def simulate(self, state: State, time_delta: float) -> State:
        state = self.update(state, time_delta)
        return state

    def update(self, state: State, time_delta: float) -> State:
        spheres = self._spheres._replace(center=state.sphere_centers)

        # Now let us handle the behaviour between the sphere and the water.
        grid_centers = (
            self._stacked_rectanguloid.corner0[:, (0, 2)]
            + self._stacked_rectanguloid.corner1[:, (0, 2)]
        ) / 2.0

        r2 = jnp.sum(
            jnp.square(
                grid_centers[None, :, :]
                - spheres.center[:, jnp.array([0, 2])][:, None, :]
            ),
            axis=-1,
        )

        collision_mask = r2 < jnp.square(spheres.radius[:, None])
        body_half_heights = jnp.sqrt(jnp.square(spheres.radius[:, None]) - r2)
        min_body = jnp.maximum(spheres.center[:, [1]] - body_half_heights, 0.0)
        max_body = jnp.minimum(
            spheres.center[:, [1]] + body_half_heights, state.water_heights[None, :]
        )
        body_heights = jnp.maximum(max_body - min_body, 0.0)
        body_heights = jnp.where(collision_mask, body_heights, 0.0)

        forces = -body_heights * jnp.square(self._spacing) * self.GRAVITY_CONSTANT
        force = jnp.sum(forces, axis=-1)

        sphere_mass = (
            4.0 * jnp.pi / 3.0 * jnp.power(spheres.radius, 3) * spheres.density
        )

        sphere_velocities = state.sphere_velocities.at[:, 1].set(
            state.sphere_velocities[:, 1] + time_delta * force / sphere_mass
        )
        sphere_velocities *= 0.975

        body_heights = jnp.sum(body_heights, axis=0)
        body_heights = jnp.reshape(body_heights, (self._n, self._m))
        # Smooth the body heights field to reduce the amount of spikes and instabilities.
        for _ in range(2):
            padded_body_heights = jnp.pad(body_heights, 1, mode="constant")
            body_heights = jax.scipy.signal.convolve2d(
                padded_body_heights,
                jnp.ones((3, 3), dtype=body_heights.dtype) / 9.0,
                mode="valid",
                precision="highest",
            )

        previous_body_heights = jnp.reshape(state.body_heights, (self._n, self._m))
        water_heights = jnp.reshape(state.water_heights, (self._n, self._m))
        body_change = body_heights - previous_body_heights
        water_heights += self.ALPHA * body_change

        wave_speed = jnp.minimum(state.wave_speed, 0.5 * self._spacing / time_delta)

        c = jnp.square(wave_speed) / jnp.square(self._spacing)
        positional_damping = jnp.minimum(self.POSITIONAL_DAMPING * time_delta, 1.0)
        velocity_damping = jnp.maximum(0.0, 1.0 - self.VELOCITY_DAMPING * time_delta)

        water_velocities = state.water_velocities.reshape((self._n, self._m))

        padded_water_heights = jnp.pad(water_heights, 1, mode="edge")
        sums = jax.scipy.signal.convolve2d(
            padded_water_heights,
            jnp.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            mode="valid",
            precision="highest",
        )
        water_velocities += time_delta * c * (sums - 4.0 * water_heights)
        water_heights += (0.25 * sums - water_heights) * positional_damping

        water_velocities *= velocity_damping
        water_heights += time_delta * water_velocities

        sphere_restitution = 0.1

        # Now let us add some behaviour to the sphere.
        sphere_velocities += time_delta * jnp.array(
            [0.0, self.GRAVITY_CONSTANT, 0.0], dtype=sphere_velocities.dtype
        )

        sphere_center = spheres.center + time_delta * sphere_velocities

        sphere_touching_floor = sphere_center[:, 1] < spheres.radius

        sphere_center = sphere_center.at[:, 1].set(
            jnp.where(sphere_touching_floor, spheres.radius, sphere_center[:, 1])
        )
        sphere_velocities = sphere_velocities.at[:, 1].set(
            jnp.where(
                sphere_touching_floor,
                -sphere_restitution * sphere_velocities[:, 1],
                sphere_velocities[:, 1],
            )
        )

        sphere_touching_north_wall = (
            sphere_center[:, 2] + spheres.radius > self._tank_size[1]
        )
        sphere_touching_south_wall = (
            sphere_center[:, 2] - spheres.radius < -self._tank_size[1]
        )
        sphere_touching_west_wall = (
            sphere_center[:, 0] + spheres.radius > self._tank_size[0]
        )
        sphere_touching_east_wall = (
            sphere_center[:, 0] - spheres.radius < -self._tank_size[0]
        )

        sphere_center = sphere_center.at[:, 2].set(
            jnp.where(
                sphere_touching_north_wall,
                self._tank_size[1] - spheres.radius,
                sphere_center[:, 2],
            )
        )
        sphere_velocities = sphere_velocities.at[:, 2].set(
            jnp.where(
                sphere_touching_north_wall,
                -sphere_restitution * sphere_velocities[:, 2],
                sphere_velocities[:, 2],
            )
        )
        sphere_center = sphere_center.at[:, 2].set(
            jnp.where(
                sphere_touching_south_wall,
                -self._tank_size[1] + spheres.radius,
                sphere_center[:, 2],
            )
        )
        sphere_velocities = sphere_velocities.at[:, 2].set(
            jnp.where(
                sphere_touching_south_wall,
                -sphere_restitution * sphere_velocities[:, 2],
                sphere_velocities[:, 2],
            )
        )
        sphere_center = sphere_center.at[:, 0].set(
            jnp.where(
                sphere_touching_west_wall,
                self._tank_size[0] - spheres.radius,
                sphere_center[:, 0],
            )
        )
        sphere_velocities = sphere_velocities.at[:, 0].set(
            jnp.where(
                sphere_touching_west_wall,
                -sphere_restitution * sphere_velocities[:, 0],
                sphere_velocities[:, 0],
            )
        )
        sphere_center = sphere_center.at[:, 0].set(
            jnp.where(
                sphere_touching_east_wall,
                -self._tank_size[0] + spheres.radius,
                sphere_center[:, 0],
            )
        )
        sphere_velocities = sphere_velocities.at[:, 0].set(
            jnp.where(
                sphere_touching_east_wall,
                -sphere_restitution * sphere_velocities[:, 0],
                sphere_velocities[:, 0],
            )
        )

        # Now let us add the behaviour between the sphere and the walls.
        return State(
            time=state.time + time_delta,
            sphere_centers=sphere_center,
            water_heights=jnp.ravel(water_heights),
            sphere_velocities=sphere_velocities,
            water_velocities=jnp.ravel(water_velocities),
            wave_speed=wave_speed,
            body_heights=jnp.ravel(body_heights),
            time_delta=time_delta,
        )
