import functools
import typing
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from water_simulator import collisions


class State(typing.NamedTuple):
    sphere_centers: jax.Array
    water_heights: jax.Array
    sphere_velocities: jax.Array
    water_velocities: jax.Array
    wave_speed: jax.Array
    body_heights: jax.Array
    time_delta: jax.Array


class Simulator:
    GRAVITY_CONSTANT = -9.81
    POSITIONAL_DAMPING = 1.0
    VELOCITY_DAMPING = 0.3
    ALPHA = 0.5

    def __init__(
        self,
        spheres: Sequence[collisions.Sphere],
        grid_field: np.ndarray,
        n: int,
        m: int,
        spacing: float,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self._n_spheres = len(spheres)
        self._spheres = jax.tree.map(lambda *x: jnp.stack(x), *spheres)
        self._n = n
        self._m = m
        self._dtype = dtype
        self._spacing = spacing
        self._grid_xz = grid_field[:, :, (0, 2)]  # .reshape((self._n * self._m, 2))
        self._grid_init_y = grid_field[:, :, 1]
        self._tank_size = (
            jnp.array([self._n, self._m], dtype=self._dtype) * spacing
        ) * 0.5
        self._jax_backend = jax.default_backend()

    def init_state(self) -> State:
        return State(
            sphere_centers=self._spheres.center.copy(),
            water_heights=self._grid_init_y.copy(),
            sphere_velocities=jnp.zeros_like(self._spheres.center),
            water_velocities=jnp.zeros((self._n, self._m), dtype=self._dtype),
            wave_speed=jnp.array(2.0, self._dtype),
            body_heights=jnp.zeros((self._n, self._m), dtype=self._dtype),
            time_delta=jnp.array(0.0, self._dtype),
        )

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def simulate(self, state: State) -> State:
        sphere_restitution = 0.1
        sphere_mass = (
            4.0
            * jnp.pi
            / 3.0
            * jnp.power(self._spheres.radius, 3)
            * self._spheres.density
        )

        (
            sphere_centers,
            sphere_velocities,
            water_heights,
            body_heights,
            water_velocities,
            wave_speed,
        ) = self._update_sphere_water_collision(
            state=state,
            sphere_mass=sphere_mass,
        )

        (
            sphere_centers,
            sphere_velocities,
        ) = self._update_sphere_floor_collision(
            sphere_centers,
            sphere_velocities,
            self._spheres.radius,
            sphere_restitution,
        )

        (
            sphere_centers,
            sphere_velocities,
        ) = self._update_sphere_wall_collision(
            sphere_centers,
            sphere_velocities,
            self._spheres.radius,
            sphere_restitution,
        )

        (
            sphere_collision_positional_correction,
            sphere_collision_velocity_correction,
        ) = self._calculate_spherical_collision_correction_updates(
            sphere_centers,
            sphere_velocities,
            self._spheres.radius,
            sphere_mass,
            sphere_restitution,
        )

        sphere_centers += sphere_collision_positional_correction
        sphere_velocities += sphere_collision_velocity_correction

        # Now let us add the behaviour between the sphere and the walls.
        return State(
            sphere_centers=sphere_centers,
            water_heights=water_heights,
            sphere_velocities=sphere_velocities,
            water_velocities=water_velocities,
            wave_speed=wave_speed,
            body_heights=body_heights,
            time_delta=state.time_delta,
        )

    def _update_sphere_water_collision(
        self,
        state: State,
        sphere_mass: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        spheres = self._spheres._replace(center=state.sphere_centers)

        sphere_body_heights = self._get_sphere_body_heights(state, spheres)
        sphere_body_heights = self._smooth_height_fields(sphere_body_heights)
        body_heights = jnp.sum(sphere_body_heights, axis=0)

        (
            water_heights,
            water_velocities,
            wave_speed,
        ) = self._update_water_by_body_height(state, body_heights)

        sphere_center, sphere_velocities = self._update_sphere_by_body_height(
            state, spheres, sphere_mass, sphere_body_heights
        )

        return (
            sphere_center,
            sphere_velocities,
            water_heights,
            body_heights,
            water_velocities,
            wave_speed,
        )

    def _get_sphere_body_heights(
        self, state: State, spheres: collisions.Sphere
    ) -> jax.Array:
        sphere_grid_distance_squared = jnp.sum(
            jnp.square(
                jnp.subtract(
                    spheres.center[:, jnp.array([0, 2])][:, None, None, :],
                    self._grid_xz[None, :, :, :],
                )
            ),
            axis=-1,
        )
        sphere_radius_squared = jnp.square(spheres.radius)
        body_half_heights_squared = jnp.subtract(
            sphere_radius_squared[:, None, None], sphere_grid_distance_squared
        )
        body_half_heights_squared = jnp.maximum(body_half_heights_squared, 0.0)
        body_half_heights = jnp.sqrt(body_half_heights_squared)
        min_body = jnp.maximum(
            spheres.center[:, [1]][:, :, None] - body_half_heights, 0.0
        )
        max_body = jnp.minimum(
            spheres.center[:, [1]][:, :, None] + body_half_heights,
            state.water_heights[None, :],
        )
        sphere_body_heights = jnp.maximum(max_body - min_body, 0.0)
        return sphere_body_heights

    def _update_water_by_body_height(
        self,
        state: State,
        body_heights: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        previous_body_heights = state.body_heights
        water_heights = state.water_heights
        body_change = body_heights - previous_body_heights
        water_heights += self.ALPHA * body_change
        wave_speed = jnp.minimum(
            state.wave_speed, 0.5 * self._spacing / state.time_delta
        )
        c = jnp.square(wave_speed / self._spacing)
        water_velocities = state.water_velocities
        padded_water_heights = jnp.pad(water_heights, 1, mode="edge")
        kernel = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        sums = jax.scipy.signal.convolve2d(
            padded_water_heights,
            kernel,
            mode="valid",
            precision="highest",
        )
        velocity_damping = jnp.maximum(
            0.0, 1.0 - self.VELOCITY_DAMPING * state.time_delta
        )
        water_velocities += state.time_delta * c * (sums - kernel.sum() * water_heights)
        water_velocities *= velocity_damping

        positional_damping = jnp.minimum(
            self.POSITIONAL_DAMPING * state.time_delta, 1.0
        )
        water_heights += (1 / kernel.sum() * sums - water_heights) * positional_damping
        water_heights += state.time_delta * water_velocities
        return (
            water_heights,
            water_velocities,
            wave_speed,
        )

    def _update_sphere_by_body_height(
        self,
        state: State,
        spheres: collisions.Sphere,
        sphere_mass: jax.Array,
        sphere_body_heights: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        forces = (
            -sphere_body_heights * jnp.square(self._spacing) * self.GRAVITY_CONSTANT
        )
        force = jnp.sum(forces, axis=(1, 2))
        acceleration = force / sphere_mass + self.GRAVITY_CONSTANT

        sphere_y_velocity_increment = state.time_delta * acceleration

        # JAX METAL does not support at[] correctly.
        if self._jax_backend in ["METAL"]:
            sphere_velocities = state.sphere_velocities + jnp.concatenate(
                [
                    jnp.zeros(
                        (self._n_spheres, 1), dtype=state.sphere_velocities.dtype
                    ),
                    jnp.expand_dims(sphere_y_velocity_increment, 1),
                    jnp.zeros(
                        (self._n_spheres, 1), dtype=state.sphere_velocities.dtype
                    ),
                ],
                axis=1,
            )
        else:
            sphere_velocities = state.sphere_velocities.at[:, 1].set(
                state.sphere_velocities[:, 1] + sphere_y_velocity_increment,
                indices_are_sorted=True,
                unique_indices=True,
            )

        sphere_velocities *= jnp.where(
            jnp.any(sphere_body_heights > 0.0, axis=(1, 2))[:, None], 0.93, 1.0
        )
        sphere_center = spheres.center + state.time_delta * sphere_velocities
        return sphere_center, sphere_velocities

    def _update_sphere_floor_collision(
        self,
        sphere_center: jax.Array,
        sphere_velocities: jax.Array,
        sphere_radius: jax.Array,
        sphere_restitution: float,
    ) -> tuple[jax.Array, jax.Array]:
        sphere_touching_floor = sphere_center[:, 1] < sphere_radius
        # JAX METAL does not support at[] correctly.
        if self._jax_backend in ["METAL"]:
            sphere_center = jnp.concatenate(
                [
                    sphere_center[:, 0, jnp.newaxis],
                    jnp.maximum(sphere_radius, sphere_center[:, 1])[:, jnp.newaxis],
                    sphere_center[:, 2, jnp.newaxis],
                ],
                axis=1,
            )
            sphere_velocities = jnp.concatenate(
                [
                    sphere_velocities[:, 0, jnp.newaxis],
                    jnp.where(
                        sphere_touching_floor,
                        -sphere_restitution * sphere_velocities[:, 1],
                        sphere_velocities[:, 1],
                    )[:, jnp.newaxis],
                    sphere_velocities[:, 2, jnp.newaxis],
                ],
                axis=1,
            )
        else:
            sphere_center = sphere_center.at[:, 1].set(
                jnp.where(sphere_touching_floor, sphere_radius, sphere_center[:, 1]),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_velocities = sphere_velocities.at[:, 1].set(
                jnp.where(
                    sphere_touching_floor,
                    -sphere_restitution * sphere_velocities[:, 1],
                    sphere_velocities[:, 1],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
        return sphere_center, sphere_velocities

    def _update_sphere_wall_collision(
        self,
        sphere_center: jax.Array,
        sphere_velocities: jax.Array,
        sphere_radius: jax.Array,
        sphere_restitution: float,
    ) -> tuple[jax.Array, jax.Array]:
        sphere_touching_north_wall = (
            sphere_center[:, 2] + sphere_radius > self._tank_size[1]
        )
        sphere_touching_south_wall = (
            sphere_center[:, 2] - sphere_radius < -self._tank_size[1]
        )
        sphere_touching_west_wall = (
            sphere_center[:, 0] + sphere_radius > self._tank_size[0]
        )
        sphere_touching_east_wall = (
            sphere_center[:, 0] - sphere_radius < -self._tank_size[0]
        )

        # JAX METAL does not support at[] correctly.
        if self._jax_backend in ["METAL"]:
            sphere_center = jnp.concatenate(
                [
                    sphere_center[:, 0, jnp.newaxis],
                    sphere_center[:, 1, jnp.newaxis],
                    jnp.where(
                        sphere_touching_north_wall,
                        self._tank_size[1] - sphere_radius,
                        sphere_center[:, 2],
                    )[:, jnp.newaxis],
                ],
                axis=1,
            )
            sphere_velocities = jnp.concatenate(
                [
                    sphere_velocities[:, 0, jnp.newaxis],
                    sphere_velocities[:, 1, jnp.newaxis],
                    jnp.where(
                        sphere_touching_north_wall,
                        -sphere_restitution * sphere_velocities[:, 2],
                        sphere_velocities[:, 2],
                    )[:, jnp.newaxis],
                ],
                axis=1,
            )

            sphere_center = jnp.concatenate(
                [
                    sphere_center[:, 0, jnp.newaxis],
                    sphere_center[:, 1, jnp.newaxis],
                    jnp.where(
                        sphere_touching_south_wall,
                        -self._tank_size[1] + sphere_radius,
                        sphere_center[:, 2],
                    )[:, jnp.newaxis],
                ],
                axis=1,
            )
            sphere_velocities = jnp.concatenate(
                [
                    sphere_velocities[:, 0, jnp.newaxis],
                    sphere_velocities[:, 1, jnp.newaxis],
                    jnp.where(
                        sphere_touching_south_wall,
                        -sphere_restitution * sphere_velocities[:, 2],
                        sphere_velocities[:, 2],
                    )[:, jnp.newaxis],
                ],
                axis=1,
            )

            sphere_center = jnp.concatenate(
                [
                    jnp.where(
                        sphere_touching_west_wall,
                        self._tank_size[0] - sphere_radius,
                        sphere_center[:, 0],
                    )[:, jnp.newaxis],
                    sphere_center[:, 1, jnp.newaxis],
                    sphere_center[:, 2, jnp.newaxis],
                ],
                axis=1,
            )
            sphere_velocities = jnp.concatenate(
                [
                    jnp.where(
                        sphere_touching_west_wall,
                        -sphere_restitution * sphere_velocities[:, 0],
                        sphere_velocities[:, 0],
                    )[:, jnp.newaxis],
                    sphere_velocities[:, 1, jnp.newaxis],
                    sphere_velocities[:, 2, jnp.newaxis],
                ],
                axis=1,
            )

            sphere_center = jnp.concatenate(
                [
                    jnp.where(
                        sphere_touching_east_wall,
                        -self._tank_size[0] + sphere_radius,
                        sphere_center[:, 0],
                    )[:, jnp.newaxis],
                    sphere_center[:, 1, jnp.newaxis],
                    sphere_center[:, 2, jnp.newaxis],
                ],
                axis=1,
            )
            sphere_velocities = jnp.concatenate(
                [
                    jnp.where(
                        sphere_touching_east_wall,
                        -sphere_restitution * sphere_velocities[:, 0],
                        sphere_velocities[:, 0],
                    )[:, jnp.newaxis],
                    sphere_velocities[:, 1, jnp.newaxis],
                    sphere_velocities[:, 2, jnp.newaxis],
                ],
                axis=1,
            )
        else:
            sphere_center = sphere_center.at[:, 2].set(
                jnp.where(
                    sphere_touching_north_wall,
                    self._tank_size[1] - sphere_radius,
                    sphere_center[:, 2],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_velocities = sphere_velocities.at[:, 2].set(
                jnp.where(
                    sphere_touching_north_wall,
                    -sphere_restitution * sphere_velocities[:, 2],
                    sphere_velocities[:, 2],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_center = sphere_center.at[:, 2].set(
                jnp.where(
                    sphere_touching_south_wall,
                    -self._tank_size[1] + sphere_radius,
                    sphere_center[:, 2],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_velocities = sphere_velocities.at[:, 2].set(
                jnp.where(
                    sphere_touching_south_wall,
                    -sphere_restitution * sphere_velocities[:, 2],
                    sphere_velocities[:, 2],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_center = sphere_center.at[:, 0].set(
                jnp.where(
                    sphere_touching_west_wall,
                    self._tank_size[0] - sphere_radius,
                    sphere_center[:, 0],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_velocities = sphere_velocities.at[:, 0].set(
                jnp.where(
                    sphere_touching_west_wall,
                    -sphere_restitution * sphere_velocities[:, 0],
                    sphere_velocities[:, 0],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_center = sphere_center.at[:, 0].set(
                jnp.where(
                    sphere_touching_east_wall,
                    -self._tank_size[0] + sphere_radius,
                    sphere_center[:, 0],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
            sphere_velocities = sphere_velocities.at[:, 0].set(
                jnp.where(
                    sphere_touching_east_wall,
                    -sphere_restitution * sphere_velocities[:, 0],
                    sphere_velocities[:, 0],
                ),
                indices_are_sorted=True,
                unique_indices=True,
            )
        return sphere_center, sphere_velocities

    def _calculate_spherical_collision_correction_updates(
        self,
        sphere_center: jax.Array,
        sphere_velocities: jax.Array,
        sphere_radius: jax.Array,
        sphere_mass: jax.Array,
        sphere_restitution: float,
    ) -> tuple[jax.Array, jax.Array]:
        centroid_directions = -1.0 * (
            sphere_center[:, None, :] - sphere_center[None, :, :]
        )
        centroid_distances = jnp.linalg.norm(centroid_directions, axis=2)
        added_radii = sphere_radius[:, None] + sphere_radius[None, :]
        intersection_mask = centroid_distances < added_radii
        intersection_mask = intersection_mask.at[jnp.diag_indices(self._n_spheres)].set(
            False,
            indices_are_sorted=True,
            unique_indices=True,
        )
        correction = (added_radii - centroid_distances) / 2.0
        collision_direction = centroid_directions / jnp.expand_dims(
            centroid_distances.at[jnp.diag_indices(self._n_spheres)].set(
                1.0,
                indices_are_sorted=True,
                unique_indices=True,
            ),
            2,
        )
        collision_direction *= intersection_mask[:, :, None]
        position_correction = jnp.sum(
            collision_direction * -1.0 * jnp.expand_dims(correction, 2),
            axis=1,
        )
        v = sphere_velocities[:, None, :] * collision_direction
        sphere_mass_mul_velocities = (sphere_mass[:, None] * sphere_velocities)[
            :, None, :
        ] * collision_direction
        new_v = (
            sphere_mass_mul_velocities
            + jnp.transpose(sphere_mass_mul_velocities, (1, 0, 2))
            - sphere_mass[:, None, None]
            * (v - jnp.transpose(v, (1, 0, 2)))
            * sphere_restitution
        ) / jnp.expand_dims((sphere_mass[:, None] + sphere_mass[None, :]), axis=2)
        collision_correction_velocities = jnp.sum(
            collision_direction * (new_v - v),
            axis=1,
        )
        return position_correction, collision_correction_velocities

    def _smooth_height_fields(self, height_fields: jax.Array) -> jax.Array:
        # Smooth the body heights field to reduce the amount of spikes and instabilities.
        height_fields = height_fields[:, :, :, None]
        height_fields = jnp.transpose(height_fields, [0, 3, 1, 2])
        kernel = np.ones((3, 3), dtype=height_fields.dtype) / 9.0
        kernel = kernel[:, :, None, None]
        kernel = np.transpose(kernel, [3, 2, 0, 1])
        for _ in range(2):
            if height_fields.shape[1:] != (1, self._n, self._m):
                raise AssertionError("Shape mismatch")
            # else...
            height_fields = jax.lax.conv(
                height_fields,
                jnp.asarray(kernel),
                (1, 1),
                "SAME",
                precision="highest",
            )
        return height_fields[:, 0, :, :]


class WaterVertexNormalUpdater:
    def __init__(self, n: int, m: int, xz: np.ndarray, indices: np.ndarray) -> None:
        self._n = n
        self._m = m
        self._n_faces = (self._n - 1) * (self._m - 1) * 2
        self._xz = np.reshape(xz, (self._n * self._m, 2))
        self._indices = indices
        (
            self._element_index_map,
            self._face_indices,
            self._vertex_normal_groups_size,
        ) = self._prepare()

    def __call__(self, water_heights: jax.Array) -> jax.Array:
        return self._compute(water_heights)

    def _prepare(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertex_normal_groups_size = np.zeros((self._n * self._m,), dtype=np.int32)
        element_index_map = np.zeros((self._n_faces * 3,), dtype=np.int32)
        face_indices = self._indices.reshape(self._n_faces, 3)

        for i, vertex_index in enumerate(self._indices):
            element_index_map[i] = vertex_normal_groups_size[vertex_index]
            vertex_normal_groups_size[vertex_index] += 1

        return (
            element_index_map.reshape(-1, 3),
            face_indices,
            vertex_normal_groups_size,
        )

    def _compute(self, water_heights: jax.Array) -> jax.Array:
        face_normals = self._calculate_face_normals(water_heights)
        vertex_normal_groups = self._vertex_face_normals(face_normals)
        vertex_normals = jnp.divide(
            jnp.sum(vertex_normal_groups, axis=1),
            self._vertex_normal_groups_size[:, None],
        )
        return vertex_normals.flatten()

    def _vertex_face_normals(self, face_normals: jax.Array) -> jax.Array:
        vertex_normal_groups = jnp.zeros(
            (self._n * self._m, max(self._vertex_normal_groups_size), 3),
            dtype=face_normals.dtype,
        )
        vertex_normal_groups = vertex_normal_groups.at[
            self._face_indices, self._element_index_map
        ].set(face_normals[:, None, :], unique_indices=True)
        return vertex_normal_groups

    def _calculate_face_normals(self, water_heights: jax.Array) -> jax.Array:
        faces_vertices = jnp.concatenate(
            (
                self._xz[self._face_indices, 0][:, :, None],
                water_heights[self._face_indices][:, :, None],
                self._xz[self._face_indices, 1][:, :, None],
            ),
            axis=2,
        )
        normals = jnp.cross(
            faces_vertices[:, 1, :] - faces_vertices[:, 0, :],
            faces_vertices[:, 2, :] - faces_vertices[:, 0, :],
        )
        normals /= jnp.linalg.norm(normals, axis=1)[:, None]
        return normals
