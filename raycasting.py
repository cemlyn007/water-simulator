import typing
import jax.numpy as jnp
import abc
import jax
from typing import Optional


class Ray(typing.NamedTuple):
    origin: jax.Array
    direction: jax.Array


class Intersections(typing.NamedTuple):
    positions: jax.Array
    valid: jax.Array


@jax.tree_util.register_pytree_node_class
class Object(abc.ABC):
    @abc.abstractmethod
    def intersect(self, ray: Ray) -> Intersections:
        pass

    @abc.abstractmethod
    def tree_flatten(self):
        pass

    @abc.abstractclassmethod
    def tree_unflatten(cls, aux_data, children):
        pass


@jax.tree_util.register_pytree_node_class
class Plane(Object):
    def __init__(self, normal: jax.Array, offset: jax.Array) -> None:
        self.normal = normal
        self.offset = offset

    def intersect(self, ray: Ray) -> Intersections:
        plane_m = jnp.dot(self.offset - ray.origin, self.normal) / jnp.dot(
            ray.direction, self.normal
        )
        valid = plane_m > 0.0
        positions = ray.origin + plane_m * ray.direction
        intersections = Intersections(
            jnp.reshape(positions, (1, 3)), jnp.reshape(valid, (1,))
        )
        return intersections

    def tree_flatten(self):
        return (self.normal, self.offset), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Sphere(Object):
    def __init__(self, center: jax.Array, radius: float) -> None:
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray) -> Intersections:
        x1 = ray.origin
        x2 = ray.origin + ray.direction
        a = jnp.sum(jnp.square(x2 - x1))
        b = 2 * jnp.sum((x2 - x1) * (x1 - self.center))
        c = jnp.sum(jnp.square(x1 - self.center)) - jnp.square(self.radius)
        discriminant = jnp.square(b) - 4.0 * a * c
        lambdas = jnp.array(
            [
                (b + jnp.sqrt(discriminant)) / (2.0 * a),
                (b - jnp.sqrt(discriminant)) / (2.0 * a),
            ],
            dtype=jnp.float32,
        )
        valid = jnp.logical_and(
            discriminant >= 0,
            jnp.array([True, jnp.reshape(discriminant == 0.0, ())]),
        )

        positions = ray.origin + lambdas[:, None] * ray.direction
        return Intersections(positions, valid)

    def tree_flatten(self):
        return (self.center, self.radius), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class BoundedPlane(Object):
    def __init__(
        self,
        normal: jax.Array,
        offset: float,
        min_point: jax.Array,
        max_point: jax.Array,
    ) -> None:
        self.normal = normal
        self.offset = offset
        self.min_point = min_point
        self.max_point = max_point

    def intersect(self, ray: Ray) -> Intersections:
        intersections = Plane(self.normal, self.offset).intersect(ray)
        valid = jnp.all(
            jnp.logical_and(
                intersections.positions >= self.min_point,
                intersections.positions <= self.max_point,
            )
        )
        return Intersections(
            intersections.positions, jnp.logical_and(intersections.valid, valid)
        )

    def tree_flatten(self):
        return (self.normal, self.offset, self.min_point, self.max_point), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Rectanguloid(Object):
    def __init__(self, corner0: jax.Array, corner1: jax.Array) -> None:
        self.corner0 = corner0
        self.corner1 = corner1

    def intersect(self, ray: Ray) -> Intersections:
        bounded_planes = (
            BoundedPlane(
                jnp.array([1.0, 0.0, 0.0]),
                self.corner0,
                self.corner0,
                jnp.array([self.corner0[0], self.corner1[1], self.corner1[2]]),
            ),
            BoundedPlane(
                jnp.array([0.0, 1.0, 0.0]),
                self.corner0,
                self.corner0,
                jnp.array([self.corner1[0], self.corner0[1], self.corner1[2]]),
            ),
            BoundedPlane(
                jnp.array([0.0, 0.0, 1.0]),
                self.corner0,
                self.corner0,
                jnp.array([self.corner1[0], self.corner1[1], self.corner0[2]]),
            ),
            BoundedPlane(
                jnp.array([1.0, 0.0, 0.0]),
                self.corner1,
                jnp.array([self.corner1[0], self.corner0[1], self.corner0[2]]),
                self.corner1,
            ),
            BoundedPlane(
                jnp.array([0.0, 1.0, 0.0]),
                self.corner1,
                jnp.array([self.corner0[0], self.corner1[1], self.corner0[2]]),
                self.corner1,
            ),
            BoundedPlane(
                jnp.array([0.0, 0.0, 1.0]),
                self.corner1,
                jnp.array([self.corner0[0], self.corner0[1], self.corner1[2]]),
                self.corner1,
            ),
        )
        intersections = jax.tree_map(
            lambda object: object.intersect(ray),
            bounded_planes,
            is_leaf=lambda o: isinstance(o, BoundedPlane),
        )
        intersections = jax.tree_map(lambda *xs: jnp.stack(xs), *intersections)
        return intersections._replace(
            positions=intersections.positions.reshape(len(bounded_planes), 3),
            valid=intersections.valid.reshape(
                len(bounded_planes),
            ),
        )

    def tree_flatten(self):
        return (self.corner0, self.corner1), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class Raycaster:
    def __init__(
        self,
        grouped_objects: dict[str, tuple[Object, ...]],
    ) -> None:
        self.grouped_objects = grouped_objects
        self._jit_cast = {group: jax.jit(self._cast) for group in self.grouped_objects}

    def cast(
        self, camera_position: jax.Array, ray_direction: jax.Array
    ) -> Optional[tuple[str, int]]:
        # https://antongerdelan.net/opengl/raycasting.html
        group_intersections = {
            group_name: self._jit_cast[group_name](
                camera_position, ray_direction, objects
            )
            for group_name, objects in self.grouped_objects.items()
        }
        results: list[str, int, float] = []
        for group, intersections in group_intersections.items():
            for i, intersection in enumerate(intersections):
                valid_positions = intersection.positions[intersection.valid]
                if valid_positions.shape[0] > 0:
                    results.append(
                        (
                            group,
                            i,
                            jnp.min(
                                jnp.linalg.norm(
                                    camera_position - valid_positions, axis=1
                                )
                            ),
                        )
                    )

        if len(results) == 0:
            return None
        # else...
        results.sort(key=lambda x: x[2])
        return (results[0][0], results[0][1])

    def _cast(
        self,
        camera_position: jax.Array,
        ray_direction: jax.Array,
        objects: tuple[Object, ...],
    ) -> list[Intersections]:
        ray = Ray(camera_position, ray_direction)
        intersections = jax.tree_map(
            lambda object: object.intersect(ray),
            objects,
            is_leaf=lambda o: isinstance(o, Object),
        )
        return intersections
