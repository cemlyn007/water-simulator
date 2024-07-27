import typing

import jax
import jax.numpy as jnp


class Sphere(typing.NamedTuple):
    center: jax.Array
    radius: jax.Array
    density: jax.Array


class Rectanguloid(typing.NamedTuple):
    corner0: jax.Array
    corner1: jax.Array

    def get_faces(self) -> jax.Array:
        faces = jnp.array(
            [
                [
                    [self.corner0[0], self.corner0[1], self.corner0[2]],
                    [self.corner0[0], self.corner1[1], self.corner0[2]],
                    [self.corner0[0], self.corner1[1], self.corner1[2]],
                    [self.corner0[0], self.corner0[1], self.corner1[2]],
                ],
                [
                    [self.corner1[0], self.corner0[1], self.corner0[2]],
                    [self.corner1[0], self.corner1[1], self.corner0[2]],
                    [self.corner1[0], self.corner1[1], self.corner1[2]],
                    [self.corner1[0], self.corner0[1], self.corner1[2]],
                ],
                [
                    [self.corner0[0], self.corner0[1], self.corner0[2]],
                    [self.corner1[0], self.corner0[1], self.corner0[2]],
                    [self.corner1[0], self.corner0[1], self.corner1[2]],
                    [self.corner0[0], self.corner0[1], self.corner1[2]],
                ],
                [
                    [self.corner0[0], self.corner1[1], self.corner0[2]],
                    [self.corner1[0], self.corner1[1], self.corner0[2]],
                    [self.corner1[0], self.corner1[1], self.corner1[2]],
                    [self.corner0[0], self.corner1[1], self.corner1[2]],
                ],
                [
                    [self.corner0[0], self.corner0[1], self.corner0[2]],
                    [self.corner1[0], self.corner0[1], self.corner0[2]],
                    [self.corner1[0], self.corner1[1], self.corner0[2]],
                    [self.corner0[0], self.corner1[1], self.corner0[2]],
                ],
                [
                    [self.corner0[0], self.corner0[1], self.corner1[2]],
                    [self.corner1[0], self.corner0[1], self.corner1[2]],
                    [self.corner1[0], self.corner1[1], self.corner1[2]],
                    [self.corner0[0], self.corner1[1], self.corner1[2]],
                ],
            ],
            dtype=jnp.float32,
        )
        return faces


def collide_sphere_with_bounded_plane(
    sphere: Sphere, bounded_plane: jax.typing.ArrayLike
) -> jax.Array:
    v = sphere.center - bounded_plane[0]
    n = jnp.cross(
        bounded_plane[1] - bounded_plane[0], bounded_plane[2] - bounded_plane[0]
    )
    v_dot_n = jnp.dot(v, n)
    signed_distance = v_dot_n / jnp.linalg.norm(n)
    intersection_to_plane = (sphere.center - signed_distance * n) / jnp.linalg.norm(n)
    face_min = jnp.min(bounded_plane, axis=0)
    face_max = jnp.max(bounded_plane, axis=0)
    possible_face_intersection = jnp.minimum(face_max, intersection_to_plane)
    possible_face_intersection = jnp.maximum(face_min, possible_face_intersection)
    distance = jnp.linalg.norm(sphere.center - possible_face_intersection)
    intersection = jnp.where(
        distance <= sphere.radius,
        possible_face_intersection,
        jnp.array([jnp.nan, jnp.nan, jnp.nan], dtype=jnp.float32),
    )
    return jnp.where(
        jnp.abs(signed_distance) <= sphere.radius,
        intersection,
        jnp.array([jnp.nan, jnp.nan, jnp.nan], dtype=jnp.float32),
    )
