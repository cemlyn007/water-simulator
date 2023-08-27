import collisions
from typing import Sequence
import collisions
import jax
import jax.numpy as jnp


class Simulator:
    def __init__(
        self,
        sphere: collisions.Sphere,
        rectanguloids: Sequence[collisions.Rectanguloid],
    ) -> None:
        self._sphere = sphere
        self._stacked_rectanguloid = jax.tree_map(
            lambda *x: jnp.stack(x), *rectanguloids
        )
        self.update = jax.jit(self.update)

    def update(
        self, time: float, sphere_center: jnp.ndarray, rectanguloid_heights: jnp.ndarray
    ) -> tuple[jax.Array, jax.Array]:
        r = 0.8
        sphere_center = sphere_center.at[0].set(r * jnp.cos(time))
        sphere_center = sphere_center.at[2].set(r * jnp.sin(time))
        sphere = self._sphere._replace(center=sphere_center)
        stacked_rectanguloid = self._stacked_rectanguloid._replace(
            corner1=self._stacked_rectanguloid.corner1.at[:, 1].set(
                rectanguloid_heights
            )
        )
        intersections = jax.vmap(
            collisions.collide_sphere_rectanguloid, in_axes=(None, 0)
        )(sphere, stacked_rectanguloid)
        collision_mask = jnp.logical_not(jnp.any(jnp.isnan(intersections), axis=(1, 2)))
        update_rectanguloid_heights = jnp.where(
            collision_mask,
            jnp.maximum(stacked_rectanguloid.corner1[:, 1] - 0.01, 0.01),
            jnp.minimum(stacked_rectanguloid.corner1[:, 1] + 0.0005, 1.0),
        )
        return sphere.center, update_rectanguloid_heights
