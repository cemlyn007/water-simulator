from water_simulator import raycasting
import jax.numpy as jnp
import jax


class TestRaycasting:
    def test_hit_plane(self):
        plane = raycasting.Plane(
            normal=jnp.array([0.0, 1.0, 0.0]),
            offset=jnp.array([0.0, 0.0, 0.0]),
        )
        ray = raycasting.Ray(
            origin=jnp.array([0.0, 1.0, 0.0]),
            direction=jnp.array([0.0, -1.0, 0.0]),
        )
        intersections = plane.intersect(ray)
        assert jnp.all(intersections.valid)
        assert jnp.allclose(intersections.positions, jnp.array([[0.0, 0.0, 0.0]]))

    def test_jit_hit_planes(self):
        planes = [
            raycasting.Plane(
                normal=jnp.array([0.0, 1.0, 0.0]),
                offset=jnp.array([0.0, 0.0, 0.0]),
            ),
            raycasting.Plane(
                normal=jnp.array([0.0, 1.0, 0.0]),
                offset=jnp.array([0.0, 1.0, 0.0]),
            ),
        ]
        ray = raycasting.Ray(
            origin=jnp.array([0.0, 1.0, 0.0]),
            direction=jnp.array([0.0, -1.0, 0.0]),
        )
        intersections = jax.jit(
            lambda r: jax.tree.map(
                lambda plane: plane.intersect(r),
                planes,
                is_leaf=lambda o: isinstance(o, raycasting.Object),
            )
        )(ray)
        assert jnp.all(
            jnp.concatenate([intersection.valid for intersection in intersections])
            == jnp.array([True, False])
        )
        assert jnp.allclose(intersections[0].positions, jnp.array([[0.0, 0.0, 0.0]]))

    def test_intersect_bounded_plane(self):
        plane = raycasting.BoundedPlane(
            normal=jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
            offset=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            min_point=jnp.array([-1.0, 0.0, -1.0], dtype=jnp.float32),
            max_point=jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32),
        )
        ray_direction = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32) - jnp.array(
            [3.0, 7.0, 3.0], dtype=jnp.float32
        )
        ray_direction /= jnp.linalg.norm(ray_direction)
        intersection = plane.intersect(
            raycasting.Ray(
                jnp.array([3.0, 7.0, 3.0], dtype=jnp.float32),
                ray_direction,
            )
        )
        assert intersection.valid
        assert jnp.allclose(
            intersection.positions, jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        )

    def test_miss_bounded_plane(self):
        plane = raycasting.BoundedPlane(
            normal=jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
            offset=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            min_point=jnp.array([-1.0, 0.0, -1.0], dtype=jnp.float32),
            max_point=jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32),
        )
        ray_direction = jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32) - jnp.array(
            [3.0, 7.0, 3.0], dtype=jnp.float32
        )
        ray_direction /= jnp.linalg.norm(ray_direction)
        intersection = plane.intersect(
            raycasting.Ray(
                jnp.array([3.0, 7.0, 3.0], dtype=jnp.float32),
                ray_direction,
            )
        )
        assert not intersection.valid

    def test_miss_plane_looking_opposite_way(self):
        plane = raycasting.Plane(
            normal=jnp.array([0.0, 1.0, 0.0]),
            offset=jnp.array([0.0, 0.0, 0.0]),
        )
        ray = raycasting.Ray(
            origin=jnp.array([0.0, 1.0, 0.0]),
            direction=jnp.array([0.0, 1.0, 0.0]),
        )
        intersections = plane.intersect(ray)
        assert jnp.all(~intersections.valid)

    def test_jit_hit_plane_and_sphere(self):
        objects = [
            raycasting.Plane(
                normal=jnp.array([0.0, 1.0, 0.0]),
                offset=jnp.array([0.0, 0.0, 0.0]),
            ),
            raycasting.Plane(
                normal=jnp.array([0.0, 1.0, 0.0]),
                offset=jnp.array([0.0, 1.0, 0.0]),
            ),
            raycasting.Sphere(
                center=jnp.array([0.0, 1.0, 0.0]),
                radius=jnp.array([0.5]),
            ),
        ]
        ray = raycasting.Ray(
            origin=jnp.array([0.0, 1.0, 0.0]),
            direction=jnp.array([0.0, -1.0, 0.0]),
        )

        @jax.jit
        def forward(r):
            return [o.intersect(r) for o in objects]

        intersections = forward(ray)
        assert len(intersections) == 3
        assert jnp.all(
            jnp.concatenate([intersection.valid for intersection in intersections])
            == jnp.array([True, False, True, False])
        )
        assert jnp.allclose(intersections[0].positions, jnp.array([[0.0, 0.0, 0.0]]))
        assert jnp.allclose(intersections[2].positions[0], jnp.array([[0.0, 0.5, 0.0]]))

    def test_intersect_rectanguloid(self):
        rectanguloid = raycasting.Rectanguloid(
            corner0=jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
            corner1=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        )
        ray = raycasting.Ray(
            origin=jnp.array([0.0, 0.0, 2.0], dtype=jnp.float32),
            direction=jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32),
        )
        intersections = rectanguloid.intersect(ray)
        assert jnp.all(
            intersections.valid == jnp.array([False, False, True, False, False, True])
        )
        assert jnp.allclose(
            intersections.positions[jnp.array([2, 5]), :],
            jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=jnp.float32),
        )

    def test_intersect_floor_case_a(self):
        ray_origin = jnp.array([3.0, 7.0, 3.0], dtype=jnp.float32)
        ray_direction = jnp.array(
            [-0.41916183, -0.8319133, -0.36362556], dtype=jnp.float32
        )
        ray_direction /= jnp.linalg.norm(ray_direction)
        ray = raycasting.Ray(ray_origin, ray_direction)
        floor = raycasting.BoundedPlane(
            normal=jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
            offset=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            min_point=jnp.array([-1.0, 0.0, -1.0], dtype=jnp.float32),
            max_point=jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32),
        )
        intersections = floor.intersect(ray)
        assert intersections.valid
        assert jnp.allclose(
            intersections.positions,
            jnp.array([[-0.5269697, 0.0, -0.05966854]], dtype=jnp.float32),
        )
