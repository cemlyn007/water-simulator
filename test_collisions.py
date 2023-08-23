import collisions
import numpy as np


class TestSphereRectanguloidCollisions:
    def test_no_collision(self):
        sphere = collisions.Sphere(np.array([0, 0, 0], dtype=np.float32), 1.0)
        rectanguloid = collisions.Rectanguloid(
            np.array([2, 2, 2], dtype=np.float32),
            np.array([3, 3, 3], dtype=np.float32),
        )
        intersections = collisions.collide_sphere_rectanguloid(sphere, rectanguloid)
        assert np.isnan(intersections).all()

    def test_no_collision(self):
        sphere = collisions.Sphere(np.array([0, 1.0, 0], dtype=np.float32), 0.49)
        rectanguloid = collisions.Rectanguloid(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        intersections = collisions.collide_sphere_rectanguloid(sphere, rectanguloid)
        assert np.isnan(intersections).all()

    def test_collision_just_touching(self):
        sphere = collisions.Sphere(np.array([0, 1.0, 0], dtype=np.float32), 0.5)
        rectanguloid = collisions.Rectanguloid(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        intersections = collisions.collide_sphere_rectanguloid(sphere, rectanguloid)
        expected_intersections = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [0.0, 0.5, 0.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )
        nan_mask = np.isnan(intersections)
        assert (nan_mask == np.isnan(expected_intersections)).all()
        assert np.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_a(self):
        sphere = collisions.Sphere(np.array([0, 1.0, 0], dtype=np.float32), 0.51)
        rectanguloid = collisions.Rectanguloid(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        intersections = collisions.collide_sphere_rectanguloid(sphere, rectanguloid)
        expected_intersections = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [0.0, 0.5, 0.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )
        nan_mask = np.isnan(intersections)
        assert (nan_mask == np.isnan(expected_intersections)).all()
        assert np.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_b(self):
        sphere = collisions.Sphere(
            np.array([0, 1.0, 0], dtype=np.float32),
            # Here I am essentially adding a small epsilon to the radius to guarantee intersection with the side faces.
            np.sqrt(1 / 2.0, dtype=np.float32) + np.float32(0.01),
        )
        rectanguloid = collisions.Rectanguloid(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        intersections = collisions.collide_sphere_rectanguloid(sphere, rectanguloid)
        expected_intersections = np.array(
            [
                [-0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [np.nan, np.nan, np.nan],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, -0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        nan_element_mask = np.isnan(intersections)
        assert (nan_element_mask == np.isnan(expected_intersections)).all()
        nan_mask = np.all(nan_element_mask, axis=1)
        assert np.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()

    def test_collision_sphere_inside_rectanguloid(self):
        sphere = collisions.Sphere(
            np.array([0, 0.5, 0], dtype=np.float32),
            np.float32(0.1),
        )
        rectanguloid = collisions.Rectanguloid(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        intersections = collisions.collide_sphere_rectanguloid(sphere, rectanguloid)
        expected_intersections = np.array(
            [
                sphere.center,
                sphere.center,
                sphere.center,
                sphere.center,
                sphere.center,
                sphere.center,
            ],
            dtype=np.float32,
        )
        nan_element_mask = np.isnan(intersections)
        assert (nan_element_mask == np.isnan(expected_intersections)).all()
        nan_mask = np.all(nan_element_mask, axis=1)
        assert np.equal(
            intersections[~nan_mask], expected_intersections[~nan_mask]
        ).all()
