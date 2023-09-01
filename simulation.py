import collisions
from typing import Sequence
import jax
import jax.numpy as jnp
import typing


class State(typing.NamedTuple):
    time: jnp.float32
    sphere_center: jnp.ndarray
    water_heights: jnp.ndarray
    sphere_velocity: jnp.ndarray
    water_velocities: jnp.ndarray
    wave_speed: jnp.float32
    body_heights: jnp.ndarray


class Simulator:
    GRAVITY_CONSTANT = -10.0
    TIME_DELTA = jnp.float32(1.0 / 30.0)
    POSITIONAL_DAMPING = jnp.float32(1.0)
    VELOCITY_DAMPING = jnp.float32(0.3)
    ALPHA = 0.5

    def __init__(
        self,
        sphere: collisions.Sphere,
        rectanguloids: Sequence[collisions.Rectanguloid],
        n: int,
        m: int,
        spacing: float,
    ) -> None:
        self._sphere = sphere
        self._stacked_rectanguloid = jax.tree_map(
            lambda *x: jnp.stack(x), *rectanguloids
        )
        self.update = jax.jit(self.update)
        self._state = State(
            time=jnp.float32(0.0),
            sphere_center=self._sphere.center,
            water_heights=jnp.ones_like(self._stacked_rectanguloid.corner1[:, 1]),
            sphere_velocity=jnp.zeros_like(self._sphere.center),
            water_velocities=jnp.zeros_like(self._stacked_rectanguloid.corner1[:, 1]),
            wave_speed=jnp.float32(2.0),
            body_heights=jnp.zeros_like(self._stacked_rectanguloid.corner1[:, 1]),
        )
        self._n = n
        self._m = m
        self._spacing = spacing

    def simulate(self) -> tuple[jax.Array, jax.Array]:
        self._state = self.update(self._state)
        return self._state.sphere_center, self._state.water_heights

    def update(self, state: State) -> State:
        sphere = self._sphere._replace(center=state.sphere_center)

        # Now let us handle the behaviour between the sphere and the water.
        grid_centers = (
            self._stacked_rectanguloid.corner0[:, (0, 2)]
            + self._stacked_rectanguloid.corner1[:, (0, 2)]
        ) / 2.0

        r2 = jnp.sum(
            jnp.square(grid_centers - sphere.center[jnp.array([0, 2])]), axis=-1
        )

        collision_mask = r2 < jnp.square(self._sphere.radius)
        body_half_heights = jnp.sqrt(jnp.square(self._sphere.radius) - r2)
        min_body_heights = jnp.maximum(sphere.center[1] - body_half_heights, 0.0)
        max_body_heights = jnp.minimum(
            sphere.center[1] + body_half_heights, state.water_heights
        )
        body_heights = jnp.maximum(max_body_heights - min_body_heights, 0.0)
        body_heights = jnp.where(collision_mask, body_heights, 0.0)

        forces = -body_heights * jnp.square(self._spacing) * self.GRAVITY_CONSTANT
        force = jnp.sum(forces, axis=-1)

        sphere_density = 0.7
        sphere_mass = (
            4 * jnp.pi / 3 * jnp.power(self._sphere.radius, 3) * sphere_density
        )

        sphere_velocity = state.sphere_velocity.at[1].set(
            state.sphere_velocity[1] + self.TIME_DELTA * force / sphere_mass
        )
        sphere_velocity *= 0.999

        body_heights = jnp.reshape(body_heights, (self._n, self._m))
        for _ in range(2):
            padded_body_heights = jnp.pad(body_heights, 1, mode="constant")
            body_heights = jax.scipy.signal.convolve2d(
                padded_body_heights,
                jnp.array(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
                mode="valid",
            ) / (
                2.0
                * (
                    jnp.pad(
                        jnp.ones(
                            [size - 2 for size in body_heights.shape],
                            dtype=body_heights.dtype,
                        ),
                        1,
                        mode="constant",
                    )
                    + jnp.ones_like(body_heights)
                )
            )

        previous_body_heights = jnp.reshape(state.body_heights, (self._n, self._m))
        water_heights = jnp.reshape(state.water_heights, (self._n, self._m))
        body_change = body_heights - previous_body_heights
        water_heights += self.ALPHA * body_change

        wave_speed = jnp.minimum(
            state.wave_speed, 0.5 * self._spacing / self.TIME_DELTA
        )

        c = jnp.square(wave_speed) / jnp.square(self._spacing)
        positional_damping = jnp.minimum(self.POSITIONAL_DAMPING * self.TIME_DELTA, 1.0)
        velocity_damping = jnp.maximum(
            0.0, 1.0 - self.VELOCITY_DAMPING * self.TIME_DELTA
        )

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
        )
        water_velocities += self.TIME_DELTA * c * (sums - 4.0 * water_heights)
        water_heights += (0.25 * sums - water_heights) * positional_damping

        water_velocities *= velocity_damping
        water_heights += self.TIME_DELTA * water_velocities

        sphere_restitution = 0.1

        # Now let us add some behaviour to the sphere.
        sphere_velocity += self.TIME_DELTA * jnp.array(
            [0.0, self.GRAVITY_CONSTANT, 0.0], dtype=jnp.float32
        )

        sphere_center = sphere.center + self.TIME_DELTA * sphere_velocity

        sphere_touching_floor = sphere_center[1] < self._sphere.radius

        sphere_center = sphere_center.at[1].set(
            jnp.where(sphere_touching_floor, self._sphere.radius, sphere_center[1])
        )
        sphere_velocity = sphere_velocity.at[1].set(
            jnp.where(
                sphere_touching_floor,
                -sphere_restitution * sphere_velocity[1],
                sphere_velocity[1],
            )
        )

        # Now let us add the behaviour between the sphere and the walls.
        return State(
            time=state.time + self.TIME_DELTA,
            sphere_center=sphere_center,
            water_heights=jnp.ravel(water_heights),
            sphere_velocity=sphere_velocity,
            water_velocities=jnp.ravel(water_velocities),
            wave_speed=wave_speed,
            body_heights=jnp.ravel(body_heights),
        )
