import argparse
import contextlib
import math
import sys
import time
from typing import Sequence

import glfw
import jax
import jax.numpy as jnp
import numpy as np
from OpenGL.GL import *

from water_simulator import collisions, meshes, raycasting, simulation, textures


def update_orbit_camera_position(
    azimuth_radians: np.float32, elevation_radians: np.float32, radius: np.float32
) -> np.ndarray[any, np.float32]:
    x = radius * np.cos(azimuth_radians) * np.cos(elevation_radians)
    y = radius * np.sin(elevation_radians)
    z = radius * np.sin(azimuth_radians) * np.cos(elevation_radians)
    return np.array([x, y, z], dtype=np.float32)


class App:
    def __init__(self, n: int, m: int, cube_width: float) -> None:
        self._n = n
        self._m = m
        self._cube_width = cube_width
        self.current_cursor_position = np.array((0.0, 0.0), dtype=np.float32)
        self.last_cursor_position = np.array((0.0, 0.0), dtype=np.float32)
        self.current_scroll_offset = np.array((0.0, 0.0), dtype=np.float32)
        self.left_button_pressed = False
        self._framebuffer_size_changed = False

        self._jax_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        self._spheres = [
            collisions.Sphere(
                np.array([-0.5, 1.0, -0.5], dtype=np.float32),
                np.float32(0.2),
                np.float32(2.0),
            ),
            collisions.Sphere(
                np.array([0.5, 1.0, -0.5], dtype=np.float32),
                np.float32(0.3),
                np.float32(0.7),
            ),
            collisions.Sphere(
                np.array([0.5, 1.0, 0.5], dtype=np.float32),
                np.float32(0.25),
                np.float32(0.2),
            ),
        ]
        self._ball_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        grid_field = self._create_grid_field()
        self._simulator = simulation.Simulator(
            [jax.tree.map(self._jax_float, sphere) for sphere in self._spheres],
            grid_field,
            self._n,
            self._m,
            cube_width,
            dtype=self._jax_float,
        )
        self._jit_simulate = jax.jit(self._simulate, donate_argnums=(0, 1, 2))

    def _create_grid_field(self) -> np.ndarray:
        x_ticks = (
            np.linspace(0.0, self._n, num=self._n, endpoint=True) * self._cube_width
            - ((self._n - 1) * self._cube_width) / 2.0
        )
        z_ticks = (
            np.linspace(0.0, self._m, num=self._m, endpoint=True) * self._cube_width
            - ((self._m - 1) * self._cube_width) / 2.0
        )

        x = np.tile(x_ticks, (len(z_ticks), 1))
        z = np.tile(z_ticks, (len(x_ticks), 1))
        y = np.empty_like(x)
        y.fill(0.8)

        grid_field = np.stack((x.T, y, z), axis=-1)
        return grid_field

    def framebuffer_size_callback(self, window, width, height):
        self._framebuffer_size_changed = True
        if sys.platform == "darwin":
            self._width = width // 2
            self._height = height // 2
        else:
            self._width = width
            self._height = height
        self._framebuffer_width_size = width
        self._framebuffer_height_size = height

    def cursor_pos_callback(self, window, xpos: float, ypos: float) -> None:
        self.last_cursor_position[0] = self.current_cursor_position[0]
        self.last_cursor_position[1] = self.current_cursor_position[1]
        self.current_cursor_position[0] = xpos
        self.current_cursor_position[1] = ypos

    def mouse_button_callback(
        self, window, button: int, action: int, mods: int
    ) -> None:
        self.left_button_pressed = (
            button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
        )

    def scroll_callback(self, window, xoffset: float, yoffset: float) -> None:
        self.current_scroll_offset[0] = xoffset
        self.current_scroll_offset[1] = yoffset

    # TODO: Maybe we can try buffer donation, otherwise out of ideas...
    def _simulate(
        self,
        state: simulation.State,
        time_delta: float,
        donated: Sequence[jax.Array] = (),
    ) -> tuple[simulation.State, jax.Array, jax.Array, jax.Array]:
        state = state._replace(
            time_delta=jnp.asarray(time_delta, dtype=self._jax_float)
        )
        next_state = self._simulator.simulate(state=state)
        water_vertex_normal_updater = simulation.WaterVertexNormalUpdater(
            self._n, self._m, self._water.xz, self._water.indices
        )
        water_heights = next_state.water_heights.ravel()
        water_normals = water_vertex_normal_updater(water_heights)
        sphere_models = self._get_sphere_models(next_state.sphere_centers)
        return (
            next_state,
            water_heights.astype(jnp.float32),
            water_normals.astype(jnp.float32),
            sphere_models,
        )

    def render_until(
        self,
        elapsed_time: float = float("inf"),
        max_iterations: int = float("inf"),
        enable_profiling: bool = False,
    ) -> None:
        try:
            glfw.init()

            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

            self._width = 800
            self._height = 600

            if sys.platform == "darwin":
                self._framebuffer_width_size = self._width * 2
                self._framebuffer_height_size = self._height * 2
            else:
                self._framebuffer_width_size = self._width
                self._framebuffer_height_size = self._height

            window = glfw.create_window(
                self._width,
                self._height,
                "Water",
                None,
                None,
            )

            if window is None:
                print("Failed to create GLFW window", flush=True)
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window")
            # else...

            glfw.make_context_current(window)
            glfw.swap_interval(0)

            (
                self._framebuffer_width_size,
                self._framebuffer_height_size,
            ) = glfw.get_framebuffer_size(window)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)

            glfw.set_framebuffer_size_callback(window, self.framebuffer_size_callback)
            glfw.set_cursor_pos_callback(window, self.cursor_pos_callback)
            glfw.set_mouse_button_callback(window, self.mouse_button_callback)
            glfw.set_scroll_callback(window, self.scroll_callback)

            self._background_camera = textures.Camera(
                self._framebuffer_width_size, self._framebuffer_height_size
            )
            self._background_camera.bind()
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            self._background_camera.unbind()

            light = meshes.Light()
            self._water = meshes.Water(self._n, self._m, self._cube_width)
            wall_size = ((max(self._n, self._m) - 1) * self._cube_width) / 2.0
            wall_thickness = self._cube_width * 2
            container = meshes.Container(
                wall_size,
                wall_thickness,
            )
            balls = [meshes.Ball(sphere.radius.item()) for sphere in self._spheres]
            light_position = np.array((1.2, 4.0, 2.0), dtype=np.float32)

            camera_position = np.array((3.0, 7.0, 3.0), dtype=np.float32)
            camera_position /= np.linalg.norm(camera_position)
            camera_radians = np.array(
                (
                    math.atan(camera_position[2] / camera_position[0]),
                    math.atan(camera_position[1] / camera_position[0]),
                ),
                dtype=np.float32,
            )
            camera_position = update_orbit_camera_position(
                camera_radians[0],
                camera_radians[1],
                5.0,
            )

            view = look_at(
                camera_position,
                np.array((0.0, 0.5, 0.0), dtype=np.float32),
                np.array((0.0, 1.0, 0.0), dtype=np.float32),
            )
            projection = perspective(
                np.radians(60.0),
                self._framebuffer_width_size / self._framebuffer_height_size,
                0.01,
                100.0,
            )
            light.set_projection(projection)
            light.set_color(np.array((1.0, 1.0, 1.0), dtype=np.float32))
            light.set_view(view)
            model = scale(
                np.identity(4, dtype=np.float32),
                np.array((0.2, 0.2, 0.2), dtype=np.float32),
            )
            model = translate(model, light_position)
            light.set_model(model)

            container.set_projection(projection)
            container.set_color(np.array((0.7, 0.7, 0.7), dtype=np.float32))
            container.set_view(view)
            container.set_model(np.identity(4, dtype=np.float32))
            container.set_light_color(np.array((1.0, 1.0, 1.0), dtype=np.float32))
            container.set_view_position(camera_position)
            container.set_light_position(light_position)

            for sphere, ball, color in zip(self._spheres, balls, self._ball_colors):
                ball.set_projection(projection)
                ball.set_color(np.array(color, dtype=np.float32))
                ball.set_view(view)
                sphere_model = jax.device_get(
                    self._get_sphere_models(
                        sphere_centers=jnp.expand_dims(sphere.center, 0)
                    )[0]
                )
                ball.set_model(np.array(sphere_model).T)
                ball.set_light_color(np.array((1.0, 1.0, 1.0), dtype=np.float32))
                ball.set_view_position(camera_position)
                ball.set_light_position(light_position)

            self._water.set_light_color(np.array((1.0, 1.0, 1.0), dtype=np.float32))
            self._water.set_texture(self._background_camera.rendered_texture)

            self._water.set_view_position(camera_position)
            self._water.set_view(view)
            self._water.set_projection(projection)
            self._water.set_light_position(light_position)

            raycaster = raycasting.Raycaster(
                {
                    "floor": (
                        raycasting.BoundedPlane(
                            normal=jnp.array([0.0, 1.0, 0.0], dtype=self._jax_float),
                            offset=jnp.array([0.0, 0.0, 0.0], dtype=self._jax_float),
                            min_point=jnp.array(
                                [-1.0, 0.0, -1.0], dtype=self._jax_float
                            )
                            * (wall_size + wall_thickness),
                            max_point=jnp.array([1.0, 0.0, 1.0], dtype=self._jax_float)
                            * (wall_size + wall_thickness),
                        ),
                    ),
                    "walls": (
                        # West Wall.
                        raycasting.Rectanguloid(
                            jnp.array(
                                [
                                    -wall_size - wall_thickness,
                                    0.0,
                                    -wall_size - wall_thickness,
                                ],
                                dtype=self._jax_float,
                            ),
                            jnp.array(
                                [
                                    -wall_size,
                                    1.25,
                                    wall_size,
                                ],
                                dtype=self._jax_float,
                            ),
                        ),
                        # North Wall.
                        raycasting.Rectanguloid(
                            jnp.array(
                                [-wall_size - wall_thickness, 0.0, wall_size],
                                dtype=self._jax_float,
                            ),
                            jnp.array(
                                [
                                    wall_size + wall_thickness,
                                    1.25,
                                    wall_size + wall_thickness,
                                ],
                                dtype=self._jax_float,
                            ),
                        ),
                        # East Wall.
                        raycasting.Rectanguloid(
                            jnp.array(
                                [
                                    wall_size,
                                    0.0,
                                    -wall_size - wall_thickness,
                                ],
                                dtype=self._jax_float,
                            ),
                            jnp.array(
                                [
                                    wall_size + wall_thickness,
                                    1.25,
                                    wall_size + wall_thickness,
                                ],
                                dtype=self._jax_float,
                            ),
                        ),
                        # South Wall.
                        raycasting.Rectanguloid(
                            jnp.array(
                                [
                                    -wall_size - wall_thickness,
                                    0.0,
                                    -wall_size - wall_thickness,
                                ],
                                dtype=self._jax_float,
                            ),
                            jnp.array(
                                [
                                    wall_size + wall_thickness,
                                    1.25,
                                    -wall_size,
                                ],
                                dtype=self._jax_float,
                            ),
                        ),
                    ),
                    "spheres": tuple(
                        raycasting.Sphere(sphere.center, sphere.radius)
                        for sphere in self._spheres
                    ),
                },
            )
            simulator_state = self._simulator.init_state()
            water_heights = jnp.empty((self._n * self._m,), dtype=self._jax_float)
            water_normals = jnp.empty((self._n * self._m * 3,), dtype=self._jax_float)
            sphere_models = jnp.empty(
                (len(self._spheres), 4 * 4), dtype=self._jax_float
            )
            start = time.monotonic()
            if jax.config.jax_disable_jit:
                simulate = self._jit_simulate
            else:
                simulate = self._jit_simulate.lower(
                    state=simulator_state,
                    time_delta=1 / 30.0,
                    donated=(water_heights, water_normals, sphere_models),
                ).compile()
            end = time.monotonic()
            print(f"Compiling simulator step took: {end - start} seconds.", flush=True)
            current_selected_entity = None

            SMOOTHING = 0.1
            time_delta = 1 / 30.0
            ray_direction = jnp.empty((3,), dtype=self._jax_float)
            previous_left_button_pressed = False
            with (
                jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True)
                if enable_profiling
                else contextlib.nullcontext()
            ):
                iteration = 0
                end_loop = start_loop = start_iteration = glfw.get_time()
                try:
                    while (
                        not glfw.window_should_close(window)
                        and start_iteration - start_loop < elapsed_time
                        and iteration < max_iterations
                    ):
                        rotate_camera = self.left_button_pressed and (
                            current_selected_entity is None
                            or current_selected_entity[0] != "spheres"
                        )

                        if rotate_camera:
                            # TODO: Optimise?
                            cursor_position_change = (
                                self.current_cursor_position - self.last_cursor_position
                            )
                            camera_radians[0] += math.radians(
                                SMOOTHING * cursor_position_change[0]
                            )
                            camera_radians[0] %= 2.0 * math.pi
                            camera_radians[1] += math.radians(
                                SMOOTHING * cursor_position_change[1]
                            )
                            camera_radians[1] %= 2.0 * math.pi

                        camera_changed = (
                            rotate_camera
                            or self.current_scroll_offset[0] != 0
                            or self.current_scroll_offset[1] != 0
                        )

                        if camera_changed:
                            # TODO: Optimise?
                            camera_radius = (
                                np.linalg.norm(camera_position)
                                + 0.1 * self.current_scroll_offset[1]
                            )
                            camera_radius = np.clip(
                                camera_radius,
                                0,
                                25.0,
                            )
                            camera_position = update_orbit_camera_position(
                                camera_radians[0],
                                camera_radians[1],
                                camera_radius,
                            )

                        self.current_scroll_offset[0] = 0.0
                        self.current_scroll_offset[1] = 0.0
                        self.last_cursor_position[0] = self.current_cursor_position[0]
                        self.last_cursor_position[1] = self.current_cursor_position[1]

                        if camera_changed:
                            view = look_at(
                                camera_position,
                                np.array((0.0, 0.5, 0.0), dtype=np.float32),
                                np.array((0.0, 1.0, 0.0), dtype=np.float32),
                            )
                            view = np.array(view)
                            light.set_view(view)
                            for ball in balls:
                                ball.set_view(view)
                            container.set_view(view)
                            self._water.set_view(view)
                            for ball in balls:
                                ball.set_view_position(camera_position)
                            container.set_view_position(camera_position)
                            self._water.set_view_position(camera_position)

                        if self._framebuffer_size_changed:
                            projection = perspective(
                                np.radians(60.0),
                                self._framebuffer_width_size
                                / self._framebuffer_height_size,
                                0.01,
                                100.0,
                            )
                            light.set_projection(projection)
                            for ball in balls:
                                ball.set_projection(projection)
                            container.set_projection(projection)
                            self._water.set_projection(projection)
                            self._background_camera.resize(
                                self._framebuffer_width_size,
                                self._framebuffer_height_size,
                            )
                            self._framebuffer_size_changed = False

                        cursor_position = self.current_cursor_position
                        previous_selected_entity = current_selected_entity
                        if self.left_button_pressed:
                            ray_direction = self._get_cursor_ray(
                                cursor_position, projection, view
                            )
                            # ray_direction = ray_direction.T
                            ray_direction = jnp.array(
                                ray_direction, dtype=self._jax_float
                            )

                            jax_camera_position = jnp.array(
                                [
                                    camera_position[0],
                                    camera_position[1],
                                    camera_position[2],
                                ],
                                dtype=self._jax_float,
                            )

                            if (
                                not previous_left_button_pressed
                                and current_selected_entity is None
                            ):
                                for sphere_index, sphere in enumerate(self._spheres):
                                    raycaster.grouped_objects["spheres"][
                                        sphere_index
                                    ].center = sphere.center

                                current_selected_entity = raycaster.cast(
                                    jax_camera_position,
                                    ray_direction,
                                )

                                if current_selected_entity is not None:
                                    print(
                                        f"[GL] Intersection: {current_selected_entity[0]}, {current_selected_entity[1]}",
                                        flush=True,
                                    )

                        if current_selected_entity and not self.left_button_pressed:
                            current_selected_entity = None
                        time_delta = min(1.0 / 30.0, 2.0 * time_delta)
                        if current_selected_entity:
                            (
                                selected_entity_type,
                                selected_entity_index,
                                intersection_distance,
                            ) = current_selected_entity
                            if selected_entity_type == "spheres":
                                pos = (
                                    jax_camera_position
                                    + (ray_direction / jnp.linalg.norm(ray_direction))
                                    * intersection_distance
                                )
                                if (
                                    current_selected_entity is not None
                                    and previous_selected_entity is None
                                    or (
                                        current_selected_entity[:2]
                                        != previous_selected_entity[:2]
                                    )
                                ):
                                    selected_sphere_center = (
                                        simulator_state.sphere_centers[
                                            selected_entity_index
                                        ]
                                    )
                                    grab_position = pos
                                    selected_sphere_velocity = jnp.zeros(
                                        (3,), dtype=self._jax_float
                                    )
                                else:
                                    selected_sphere_velocity = (
                                        pos - grab_position
                                    ) / time_delta
                                    selected_sphere_center = pos
                                    grab_position = pos

                                simulator_state = simulator_state._replace(
                                    sphere_centers=simulator_state.sphere_centers.at[
                                        selected_entity_index
                                    ].set(selected_sphere_center),
                                    sphere_velocities=simulator_state.sphere_velocities.at[
                                        selected_entity_index
                                    ].set(selected_sphere_velocity),
                                )
                        previous_selected_entity = current_selected_entity

                        simulator_state, water_heights, water_normals, sphere_models = (
                            simulate(
                                state=simulator_state,
                                time_delta=time_delta,
                                donated=(water_heights, water_normals, sphere_models),
                            )
                        )
                        (
                            np_sphere_centers,
                            np_water_heights,
                            np_water_normals,
                            np_sphere_models,
                        ) = jax.device_get(
                            (
                                simulator_state.sphere_centers,
                                water_heights,
                                water_normals,
                                sphere_models,
                            )
                        )
                        for i in range(len(self._spheres)):
                            self._spheres[i] = self._spheres[i]._replace(
                                center=np_sphere_centers[i]
                            )
                            balls[i].set_model(np_sphere_models[i])

                        self._water.set_water_heights(np_water_heights)
                        self._water.set_water_normals(np_water_normals)

                        self._background_camera.bind()
                        glClearColor(0.1, 0.1, 0.1, 1.0)
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                        container.draw()
                        for ball in balls:
                            ball.draw()
                        self._background_camera.unbind()

                        glViewport(
                            0,
                            0,
                            self._framebuffer_width_size,
                            self._framebuffer_height_size,
                        )
                        glClearColor(0.1, 0.1, 0.1, 1.0)
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                        light.draw()
                        container.draw()
                        for ball in balls:
                            ball.draw()
                        self._water.draw()

                        glfw.swap_buffers(window)
                        previous_left_button_pressed = self.left_button_pressed
                        glfw.poll_events()
                        end_iteration = glfw.get_time()
                        time_delta = end_iteration - start_iteration
                        start_iteration = end_iteration
                        iteration += 1
                except KeyboardInterrupt:
                    pass
                finally:
                    end_loop = glfw.get_time()
                    average_latency_seconds = (
                        (end_loop - start_loop) / iteration
                        if iteration
                        else float("inf")
                    )
                    print(
                        f"Average FPS: {1.0 / average_latency_seconds} | Average Latency: {average_latency_seconds * 1000:.3f}ms.",
                        flush=True,
                    )
        finally:
            print("[GL] Terminating", flush=True)
            glfw.terminate()

    def _get_cursor_ray(
        self, cursor_position: np.ndarray, projection: np.ndarray, view: np.ndarray
    ) -> np.ndarray:
        x = (2.0 * cursor_position[0]) / self._width - 1.0
        y = 1.0 - (2.0 * cursor_position[1]) / self._height
        ray_clip = np.array((x, y, -1.0, 1.0))
        ray_eye = np.dot(np.linalg.inv(projection.T), ray_clip)
        ray_eye = np.array((ray_eye[0], ray_eye[1], -1.0, 0.0))
        ray_world = np.dot(np.linalg.inv(view.T), ray_eye)
        ray_world = np.array(ray_world[:3])
        ray_world /= np.linalg.norm(ray_world)
        return ray_world.astype(np.float32)

    def _get_sphere_models(self, sphere_centers: jax.Array) -> jax.Array:
        sphere_models = jnp.repeat(
            jnp.expand_dims(jnp.identity(4), 0), len(self._spheres), axis=0
        )
        sphere_models = sphere_models.at[:, :3, 3].set(sphere_centers)
        return sphere_models.reshape((sphere_models.shape[0], -1), order="F")


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = np.array(center - eye)
    f = f / np.linalg.norm(f)

    u = np.array(up)
    u = u / np.linalg.norm(u)

    s = np.cross(f, u)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    result = np.identity(4)

    result[0, 0] = s[0]
    result[1, 0] = s[1]
    result[2, 0] = s[2]

    result[0, 1] = u[0]
    result[1, 1] = u[1]
    result[2, 1] = u[2]

    result[0, 2] = -f[0]
    result[1, 2] = -f[1]
    result[2, 2] = -f[2]

    result[3, 0] = -np.dot(s, eye)
    result[3, 1] = -np.dot(u, eye)
    result[3, 2] = np.dot(f, eye)

    return result


def perspective(
    fov: np.ndarray, aspect: np.ndarray, near: np.ndarray, far: np.ndarray
) -> np.ndarray:
    """
    Create a perspective projection matrix.

    :param fov: Field of view in the y direction, in degrees.
    :param aspect: Aspect ratio, defined as width divided by height.
    :param near: Distance from the viewer to the near clipping plane.
    :param far: Distance from the viewer to the far clipping plane.
    :return: A 4x4 perspective projection matrix.
    """
    f = 1.0 / np.tan(fov / 2)
    nf = 1 / (near - far)

    result = np.zeros((4, 4))

    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) * nf
    result[2, 3] = -1.0
    result[3, 2] = (2 * far * near) * nf

    return result


def translate(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Apply a translation to a 4x4 matrix.

    :param matrix: The original 4x4 matrix.
    :param vector: A 3-element translation vector.
    :return: The translated 4x4 matrix.
    """
    translation_matrix = np.identity(4)
    translation_matrix[3, 0:3] = vector

    return np.dot(matrix, translation_matrix)


def scale(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Apply a scaling transformation to a 4x4 matrix.

    :param matrix: The original 4x4 matrix.
    :param vector: A 3-element scaling vector.
    :return: The scaled 4x4 matrix.
    """
    scaling_matrix = np.identity(4)
    scaling_matrix[0, 0] = vector[0]
    scaling_matrix[1, 1] = vector[1]
    scaling_matrix[2, 2] = vector[2]

    return np.dot(matrix, scaling_matrix)


def main():
    argument_parser = argparse.ArgumentParser(
        description="Water Simulator implemented using JAX and OpenGL."
    )
    argument_parser.add_argument(
        "--n", type=int, default=101, help="number of cubes in the x and z axis"
    )
    argument_parser.add_argument("--max_seconds", type=int, default=float("inf"))
    argument_parser.add_argument("--max_iterations", type=int, default=float("inf"))
    argument_parser.add_argument(
        "--enable_profiling", action="store_true", default=False
    )
    arguments = argument_parser.parse_args()
    n = arguments.n
    print(f"Using {n*n} instances", flush=True)
    app = App(n, n, 0.02)
    app.render_until(
        elapsed_time=arguments.max_seconds,
        max_iterations=arguments.max_iterations,
        enable_profiling=arguments.enable_profiling,
    )


if __name__ == "__main__":
    sys.exit(main())
