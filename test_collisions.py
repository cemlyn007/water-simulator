import collisions
import jax.numpy as jnp
import jax
import math


class TestSphereRectanguloidCollisions:
    def test_no_collision(self):
        sphere = collisions.Sphere(jnp.array([0, 0, 0], dtype=jnp.float32), 1.0)
        rectanguloid = collisions.Rectanguloid(
            jnp.array([2, 2, 2], dtype=jnp.float32),
            jnp.array([3, 3, 3], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        assert jnp.isnan(intersections).all()

    def test_no_collision(self):
        sphere = collisions.Sphere(jnp.array([0, 1.0, 0], dtype=jnp.float32), 0.49)
        rectanguloid = collisions.Rectanguloid(
            jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        assert jnp.isnan(intersections).all()

    def test_collision_just_touching(self):
        sphere = collisions.Sphere(jnp.array([0, 1.0, 0], dtype=jnp.float32), 0.5)
        rectanguloid = collisions.Rectanguloid(
            jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        expected_intersections = jnp.array(
            [
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
                [0.0, 0.5, 0.0],
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
            ],
            dtype=jnp.float32,
        )
        nan_mask = jnp.isnan(intersections)
        assert (nan_mask == jnp.isnan(expected_intersections)).all()
        assert jnp.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_a(self):
        sphere = collisions.Sphere(jnp.array([0, 1.0, 0], dtype=jnp.float32), 0.51)
        rectanguloid = collisions.Rectanguloid(
            jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        expected_intersections = jnp.array(
            [
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
                [0.0, 0.5, 0.0],
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
            ],
            dtype=jnp.float32,
        )
        nan_mask = jnp.isnan(intersections)
        assert (nan_mask == jnp.isnan(expected_intersections)).all()
        assert jnp.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_b(self):
        sphere = collisions.Sphere(
            jnp.array([0, 1.0, 0], dtype=jnp.float32),
            # Here I am essentially adding a small epsilon to the radius to guarantee intersection with the side faces.
            jnp.float32(math.sqrt(1 / 2.0) + 0.01),
        )
        rectanguloid = collisions.Rectanguloid(
            jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        expected_intersections = jnp.array(
            [
                [-0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [jnp.nan, jnp.nan, jnp.nan],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, -0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=jnp.float32,
        )
        nan_element_mask = jnp.isnan(intersections)
        assert (nan_element_mask == jnp.isnan(expected_intersections)).all()
        nan_mask = jnp.all(nan_element_mask, axis=1)
        assert jnp.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_sphere_under_rectanguloid(self):
        sphere = collisions.Sphere(
            jnp.array([0, -1.0, 0], dtype=jnp.float32),
            jnp.float32(math.sqrt(1 / 2.0) - 0.01),
        )
        rectanguloid = collisions.Rectanguloid(
            jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        expected_intersections = jnp.array(
            [
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
                [0.0, -0.5, 0.0],
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, jnp.nan],
            ],
            dtype=jnp.float32,
        )
        nan_element_mask = jnp.isnan(intersections)
        assert (nan_element_mask == jnp.isnan(expected_intersections)).all()
        nan_mask = jnp.all(nan_element_mask, axis=1)
        assert jnp.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_corner_collision_rectanguloid_off_center(self):
        sphere = collisions.Sphere(
            jnp.array([10.55, 15.3, 21.05], dtype=jnp.float32),
            jnp.float32(0.1),
        )
        rectanguloid = collisions.Rectanguloid(
            jnp.array([10.0, 15.0, 20.0], dtype=jnp.float32),
            jnp.array([10.5, 15.25, 21.0], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        expected_intersections = jnp.array(
            [
                [jnp.nan, jnp.nan, jnp.nan],
                [10.5, 15.25, 21.0],
                [jnp.nan, jnp.nan, jnp.nan],
                [10.5, 15.25, 21.0],
                [jnp.nan, jnp.nan, jnp.nan],
                [10.5, 15.25, 21.0],
            ],
            dtype=jnp.float32,
        )
        nan_element_mask = jnp.isnan(intersections)
        assert (nan_element_mask == jnp.isnan(expected_intersections)).all()
        nan_mask = jnp.all(nan_element_mask, axis=1)
        assert jnp.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_sphere_inside_rectanguloid(self):
        sphere = collisions.Sphere(
            jnp.array([0, 0.5, 0], dtype=jnp.float32),
            jnp.float32(0.1),
        )
        rectanguloid = collisions.Rectanguloid(
            jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
        )
        intersections = jax.jit(collisions.collide_sphere_rectanguloid)(
            sphere, rectanguloid
        )
        expected_intersections = jnp.array(
            [
                sphere.center,
                sphere.center,
                sphere.center,
                sphere.center,
                sphere.center,
                sphere.center,
            ],
            dtype=jnp.float32,
        )
        nan_element_mask = jnp.isnan(intersections)
        assert (nan_element_mask == jnp.isnan(expected_intersections)).all()
        nan_mask = jnp.all(nan_element_mask, axis=1)
        assert jnp.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()
