import glfw
from OpenGL.GL import *
import sys
import numpy as np
import glm
import threading
import math
import time
import meshes
import textures


def update_orbit_camera_position(
    azimuth_radians: np.float32, elevation_radians: np.float32, radius: np.float32
) -> np.ndarray[any, np.float32]:
    x = radius * np.cos(azimuth_radians) * np.cos(elevation_radians)
    y = radius * np.sin(elevation_radians)
    z = radius * np.sin(azimuth_radians) * np.cos(elevation_radians)
    return np.array([x, y, z], dtype=np.float32)


def get_y_scale_inplace(
    arr: np.ndarray[any, np.float32],
    i: np.ndarray[any, np.float32],
    j: np.ndarray[any, np.float32],
    t: float,
    cache: dict[str, np.ndarray[any, np.float32]],
) -> None:
    if not cache:
        cache["_ij"] = np.multiply(2 * np.pi * 0.0001, i) + np.multiply(
            2 * np.pi * 0.001, j
        )
        cache["_tmp"] = np.empty_like(i)

    np.multiply(-2 * np.pi * 0.1, t, out=cache["_tmp"])
    np.add(cache["_ij"], cache["_tmp"], out=arr)
    np.sin(arr, out=arr)
    np.multiply(0.25, arr, out=arr)
    np.add(arr, 1.0, out=arr)


class App:
    def __init__(self, n: int, m: int, cube_width: float) -> None:
        self._n = n
        self._m = m
        self._cube_width = cube_width
        self._instances = n * m
        self.current_cursor_position = glm.vec2(0.0, 0.0)
        self.last_cursor_position = glm.vec2(0.0, 0.0)
        self.current_scroll_offset = glm.vec2(0.0, 0.0)
        self.rotate_camera = False
        self._can_update_model = threading.Event()
        self._update_model = threading.Event()
        self._terminate = threading.Event()
        self._model_y = np.zeros(self._instances, dtype=np.float32)
        self._model_y_a = np.zeros(self._instances, dtype=np.float32)
        self._model_y_b = np.zeros(self._instances, dtype=np.float32)

    def framebuffer_size_callback(self, window, width, height):
        self._framebuffer_width_size = width
        self._framebuffer_height_size = height
        print(self._framebuffer_width_size, self._framebuffer_height_size, flush=True)
        glViewport(0, 0, width, height)
        self._background_camera.resize(width, height)

    def cursor_pos_callback(self, window, xpos: float, ypos: float) -> None:
        self.last_cursor_position.x = self.current_cursor_position.x
        self.last_cursor_position.y = self.current_cursor_position.y
        self.current_cursor_position.x = xpos
        self.current_cursor_position.y = ypos

    def mouse_button_callback(
        self, window, button: int, action: int, mods: int
    ) -> None:
        self.rotate_camera = button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS

    def scroll_callback(self, window, xoffset: float, yoffset: float) -> None:
        self.current_scroll_offset.x = xoffset
        self.current_scroll_offset.y = yoffset

    def simulation_thread(self) -> None:
        self._can_update_model.set()

        i = np.arange(self._n, dtype=np.float32)
        j = np.arange(self._m, dtype=np.float32)

        i, j = np.meshgrid(i, j, indexing="ij")
        i = i.flatten()
        j = j.flatten()

        y_scale = self._model_y_a

        cache = {}
        while True:
            t = time.monotonic()
            get_y_scale_inplace(y_scale, i, j, t, cache)
            self._can_update_model.wait()
            self._can_update_model.clear()
            if self._terminate.is_set():
                print("[SIM] Terminating", flush=True)
                break
            # else...
            self._model_y = y_scale
            self._update_model.set()

            y_scale = self._model_y_b if y_scale is self._model_y_a else self._model_y_a

    def render_until(self, elapsed_time: float = float("inf")) -> None:
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
                self._framebuffer_width_size = width
                self._framebuffer_height_size = height

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

            water = meshes.Water(self._n, self._m, self._cube_width)
            container = meshes.Container(
                (max(self._n, self._m) * self._cube_width * 1.1) / 2.0
            )

            light_position = glm.vec3(1.2, 4.0, 2.0)

            camera_position = glm.vec3(3.0, 3.0, 3.0)
            camera_radians = glm.vec2(0.0, 0.0)

            view = glm.lookAt(
                camera_position,
                glm.vec3(0.0, 0.5, 0.0),
                glm.vec3(0.0, 1.0, 0.0),
            )
            projection = glm.perspective(
                glm.radians(45.0),
                self._framebuffer_width_size / self._framebuffer_height_size,
                0.1,
                100.0,
            )

            light.set_projection(projection)
            light.set_color(glm.vec3(1.0, 1.0, 1.0))
            light.set_view(view)
            model = glm.mat4(1.0)
            model = glm.translate(
                model, glm.vec3(light_position[0], light_position[1], light_position[2])
            )
            model = glm.scale(model, glm.vec3(0.2))
            light.set_model(model)

            container.set_projection(projection)
            container.set_color(glm.vec3(0.7, 0.7, 0.7))
            container.set_view(view)
            container.set_model(glm.mat4(1.0))
            container.set_light_color(glm.vec3(1.0, 1.0, 1.0))
            container.set_view_position(camera_position)
            container.set_light_position(light_position)

            water.set_light_color(glm.vec3(1.0, 1.0, 1.0))
            water.set_texture(self._background_camera.rendered_texture)

            water.set_view_position(camera_position)
            water.set_view(view)
            water.set_projection(projection)
            water.set_light_position(light_position)

            smoothing = 0.1
            while (
                not glfw.window_should_close(window) and glfw.get_time() < elapsed_time
            ):
                start = glfw.get_time()
                cursor_position_change = (
                    self.current_cursor_position - self.last_cursor_position
                )
                if self.rotate_camera:
                    camera_radians.x += math.radians(
                        smoothing * cursor_position_change.x
                    )
                    camera_radians.x %= 2.0 * math.pi
                    camera_radians.y += math.radians(
                        smoothing * cursor_position_change.y
                    )
                    camera_radians.y %= 2.0 * math.pi

                camera_changed = (
                    self.rotate_camera
                    or self.current_scroll_offset.x != 0
                    or self.current_scroll_offset.y != 0
                )

                if camera_changed:
                    camera_radius = (
                        np.linalg.norm(camera_position)
                        + 0.1 * self.current_scroll_offset.y
                    )
                    camera_radius = np.clip(
                        camera_radius,
                        0,
                        25.0,
                    )
                    camera_position = glm.vec3(
                        *update_orbit_camera_position(
                            camera_radians[0],
                            camera_radians[1],
                            camera_radius,
                        )
                    )

                self.current_scroll_offset.x = 0.0
                self.current_scroll_offset.y = 0.0
                self.last_cursor_position.x = self.current_cursor_position.x
                self.last_cursor_position.y = self.current_cursor_position.y

                if camera_changed:
                    view = glm.lookAt(
                        camera_position,
                        glm.vec3(0.0, 0.5, 0.0),
                        glm.vec3(0.0, 1.0, 0.0),
                    )
                    light.set_view(view)
                    container.set_view(view)
                    water.set_view(view)
                    container.set_view_position(camera_position)
                    water.set_view_position(camera_position)

                if self._update_model.is_set():
                    self._update_model.clear()
                    water.set_water_heights(glm.array(self._model_y))
                    self._can_update_model.set()

                self._background_camera.bind()
                glClearColor(0.1, 0.1, 0.1, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                light.draw()
                container.draw()
                self._background_camera.unbind()

                glClearColor(0.1, 0.1, 0.1, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                light.draw()
                container.draw()
                water.draw()

                glfw.swap_buffers(window)
                glfw.poll_events()
                end = glfw.get_time()
                cost = end - start
                # print(f"[GL] Frame cost: {cost*1000.:.2f}ms")
        finally:
            print("[GL] Terminating", flush=True)
            self._can_update_model.set()
            self._terminate.set()
            glfw.terminate()


if __name__ == "__main__":
    n = 1000
    print(f"Using {n*n} instances", flush=True)
    app = App(n, n, 0.0025)
    simulation_thread = threading.Thread(target=app.simulation_thread)
    simulation_thread.start()
    app.render_until()
    simulation_thread.join()
