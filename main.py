import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import sys
import numpy as np
import glm
import threading
import math
import time


def update_orbit_camera_position(
    azimuth_radians: np.float32, elevation_radians: np.float32, radius: np.float32
) -> np.ndarray[any, np.float32]:
    x = radius * np.cos(azimuth_radians) * np.cos(elevation_radians)
    y = radius * np.sin(elevation_radians)
    z = radius * np.sin(azimuth_radians) * np.cos(elevation_radians)
    return np.array([x, y, z], dtype=np.float32)


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def get_xz_translate(n: int, m: int, cube_width: float) -> np.ndarray[any, np.float32]:
    full_width = 1.0 * cube_width
    half_width = 0.5 * cube_width
    translate = []
    index = 0
    for i in range(n):
        for j in range(m):
            translate.extend(
                [
                    # X coordinate.
                    full_width * i - ((full_width * n) / 2.0) + half_width,
                    # Z coordinate.
                    full_width * j - ((full_width * m) / 2.0) + half_width,
                ]
            )
            index += 1
    return np.array(translate, dtype=np.float32)


def get_y_scale_inplace(
    arr: np.ndarray[any, np.float32],
    i: np.ndarray[any, np.float32],
    j: np.ndarray[any, np.float32],
    t: float,
) -> None:
    np.radians(0.1 * (1 + i + j) * t, dtype=np.float32, out=arr)
    np.sin(arr, out=arr)
    np.add(arr, 1.0, out=arr)
    np.multiply(arr, 1.5, out=arr)


def cube_vertices_normals_and_indices():
    vertices = [
        # Front Face
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        # Back Face
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        # Top Face
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        # Bottom Face
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        # Right Face
        (0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
        # Left Face
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
    ]

    normals = [
        # Front Face
        (0, 0, 1),
        (0, 0, 1),
        (0, 0, 1),
        (0, 0, 1),
        # Back Face
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, -1),
        # Top Face
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 0),
        # Bottom Face
        (0, -1, 0),
        (0, -1, 0),
        (0, -1, 0),
        (0, -1, 0),
        # Right Face
        (1, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
        # Left Face
        (-1, 0, 0),
        (-1, 0, 0),
        (-1, 0, 0),
        (-1, 0, 0),
    ]

    # Indices to form triangles for GL_TRIANGLES
    indices = [
        # Front Face
        0,
        1,
        2,
        2,
        3,
        0,
        # Back Face
        4,
        5,
        6,
        6,
        7,
        4,
        # Top Face
        8,
        9,
        10,
        10,
        11,
        8,
        # Bottom Face
        12,
        13,
        14,
        14,
        15,
        12,
        # Right Face
        16,
        17,
        18,
        18,
        19,
        16,
        # Left Face
        20,
        21,
        22,
        22,
        23,
        20,
    ]

    return vertices, normals, indices


class App:
    def __init__(self, n: int, m: int):
        self._n = n
        self._m = m
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

        while True:
            t = time.monotonic()
            get_y_scale_inplace(y_scale, i, j, t)
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

            window = glfw.create_window(800, 600, "LearnOpenGL", None, None)

            if window is None:
                print("Failed to create GLFW window", flush=True)
                glfw.terminate()
                sys.exit(-1)

            glfw.make_context_current(window)
            glfw.swap_interval(0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_CULL_FACE)

            glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
            glfw.set_cursor_pos_callback(window, self.cursor_pos_callback)
            glfw.set_mouse_button_callback(window, self.mouse_button_callback)
            glfw.set_scroll_callback(window, self.scroll_callback)

            glEnable(GL_DEPTH_TEST)

            with open("shaders/basic_lighting.vs", "r") as file:
                vertex_shader_source = file.read()
                vertex_shader = shaders.compileShader(
                    vertex_shader_source, GL_VERTEX_SHADER
                )
            with open("shaders/basic_lighting.fs", "r") as file:
                fragment_shader_source = file.read()
                fragment_shader = shaders.compileShader(
                    fragment_shader_source, GL_FRAGMENT_SHADER
                )

            with open("shaders/light_cube.vs", "r") as file:
                vertex_shader_source = file.read()
                light_cube_vertex_shader = shaders.compileShader(
                    vertex_shader_source, GL_VERTEX_SHADER
                )
            with open("shaders/light_cube.fs", "r") as file:
                fragment_shader_source = file.read()
                light_cube_fragment_shader = shaders.compileShader(
                    fragment_shader_source, GL_FRAGMENT_SHADER
                )

            vertices, normals, indices = cube_vertices_normals_and_indices()

            vertex_data = []
            for vertex, normal in zip(vertices, normals):
                vertex_data.extend(vertex)
                vertex_data.extend(normal)
            vertex_data = glm.array(np.array(vertex_data, dtype=np.float32))

            indices = glm.array(np.array(indices, dtype=np.uint32), dtype=glm.uint32)
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER,
                np.array(indices, dtype=np.uint32).nbytes,
                np.array(indices, dtype=np.uint32),
                GL_STATIC_DRAW,
            )

            vbo = glGenBuffers(1)

            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                glm.sizeof(vertex_data),
                vertex_data.ptr,
                GL_STATIC_DRAW,
            )

            cube_width = 0.05

            model_scale_xz_vbo = glGenBuffers(1)
            xz_scale = np.array(cube_width, dtype=np.float32)
            xz_scale = np.tile(xz_scale, self._instances)
            glBindBuffer(GL_ARRAY_BUFFER, model_scale_xz_vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                xz_scale.nbytes,
                xz_scale,
                GL_STATIC_DRAW,
            )

            model_translate_xz_vbo = glGenBuffers(1)
            xz_translate = get_xz_translate(self._n, self._m, cube_width)
            glBindBuffer(GL_ARRAY_BUFFER, model_translate_xz_vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                xz_translate.nbytes,
                xz_translate,
                GL_STATIC_DRAW,
            )

            model_y_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, model_y_vbo)
            # First half will be y translations, second half will be y scales.
            glBufferData(
                GL_ARRAY_BUFFER,
                self._model_y.nbytes,
                self._model_y,
                GL_DYNAMIC_DRAW,
            )

            model_vao = glGenVertexArrays(1)
            glBindVertexArray(model_vao)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glVertexAttribPointer(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                6 * vertex_data.dt_size,
                ctypes.c_void_p(0),
            )
            glEnableVertexAttribArray(0)

            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                6 * vertex_data.dt_size,
                ctypes.c_void_p(3 * vertex_data.dt_size),
            )
            glEnableVertexAttribArray(1)

            # aInstanceTranslateX
            glBindBuffer(GL_ARRAY_BUFFER, model_translate_xz_vbo)
            glVertexAttribPointer(
                2,
                1,
                GL_FLOAT,
                GL_FALSE,
                2 * xz_translate.itemsize,
                None,
            )
            glEnableVertexAttribArray(2)
            glVertexAttribDivisor(2, 1)

            # aInstanceTranslateZ
            glBindBuffer(GL_ARRAY_BUFFER, model_translate_xz_vbo)
            glVertexAttribPointer(
                3,
                1,
                GL_FLOAT,
                GL_FALSE,
                2 * xz_translate.itemsize,
                ctypes.c_void_p(1 * xz_translate.itemsize),
            )
            glEnableVertexAttribArray(3)
            glVertexAttribDivisor(3, 1)

            # aInstanceScaleY
            glBindBuffer(GL_ARRAY_BUFFER, model_y_vbo)
            glVertexAttribPointer(
                4,
                1,
                GL_FLOAT,
                GL_FALSE,
                self._model_y.itemsize,
                None,
            )
            glEnableVertexAttribArray(4)
            glVertexAttribDivisor(4, 1)

            lighting_shader = shaders.compileProgram(
                vertex_shader, fragment_shader, validate=True
            )

            light_cube_vao = glGenVertexArrays(1)
            glBindVertexArray(light_cube_vao)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

            glBindBuffer(GL_ARRAY_BUFFER, vbo)

            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, 6 * vertex_data.dt_size, ctypes.c_void_p(0)
            )
            glEnableVertexAttribArray(0)

            light_cube_shader = shaders.compileProgram(
                light_cube_vertex_shader, light_cube_fragment_shader, validate=True
            )

            light_pos = glm.vec3(1.2, 4.0, 2.0)

            camera_position = glm.vec3(3.0, 1.0, 3.0)
            camera_radians = glm.vec2(0.0, 0.0)

            view = glm.lookAt(
                camera_position,
                glm.vec3(0.0, 0.0, 0.0),
                glm.vec3(0.0, 1.0, 0.0),
            )
            projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)

            glUseProgram(light_cube_shader)
            glUniformMatrix4fv(
                glGetUniformLocation(light_cube_shader, "projection"),
                1,
                GL_FALSE,
                glm.value_ptr(projection),
            )
            glUniform3f(
                glGetUniformLocation(light_cube_shader, "objectColor"),
                1.0,
                1.0,
                1.0,
            )
            glUniformMatrix4fv(
                glGetUniformLocation(light_cube_shader, "view"),
                1,
                GL_FALSE,
                glm.value_ptr(view),
            )
            model = glm.mat4(1.0)
            model = glm.translate(
                model, glm.vec3(light_pos[0], light_pos[1], light_pos[2])
            )
            model = glm.scale(model, glm.vec3(0.2))

            glUniformMatrix4fv(
                glGetUniformLocation(light_cube_shader, "model"),
                1,
                GL_FALSE,
                glm.value_ptr(model),
            )

            glUseProgram(lighting_shader)
            glUniform4f(
                glGetUniformLocation(lighting_shader, "objectColor"),
                0.0,
                0.2,
                1.0,
                1.0,
            )
            glUniform3f(
                glGetUniformLocation(lighting_shader, "lightColor"), 1.0, 1.0, 1.0
            )
            glUniform1f(glGetUniformLocation(lighting_shader, "cubeWidth"), cube_width)
            glUniform3f(
                glGetUniformLocation(lighting_shader, "viewPos"),
                camera_position.x,
                camera_position.y,
                camera_position.z,
            )
            glUniformMatrix4fv(
                glGetUniformLocation(lighting_shader, "view"),
                1,
                GL_FALSE,
                glm.value_ptr(view),
            )
            glUniformMatrix4fv(
                glGetUniformLocation(lighting_shader, "projection"),
                1,
                GL_FALSE,
                glm.value_ptr(projection),
            )
            glUniform3f(
                glGetUniformLocation(lighting_shader, "lightPos"),
                light_pos[0],
                light_pos[1],
                light_pos[2],
            )

            smoothing = 0.1
            while (
                not glfw.window_should_close(window) and glfw.get_time() < elapsed_time
            ):
                start = glfw.get_time()
                glClearColor(0.1, 0.1, 0.1, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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
                    np.clip(camera_radius, 0, 25.0, out=camera_radius)
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
                        glm.vec3(0.0, 0.0, 0.0),
                        glm.vec3(0.0, 1.0, 0.0),
                    )

                glUseProgram(light_cube_shader)
                if camera_changed:
                    glUniformMatrix4fv(
                        glGetUniformLocation(light_cube_shader, "view"),
                        1,
                        GL_FALSE,
                        glm.value_ptr(view),
                    )

                glBindVertexArray(light_cube_vao)
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

                glUseProgram(lighting_shader)
                if camera_changed:
                    glUniform3f(
                        glGetUniformLocation(lighting_shader, "viewPos"),
                        camera_position.x,
                        camera_position.y,
                        camera_position.z,
                    )
                    glUniformMatrix4fv(
                        glGetUniformLocation(lighting_shader, "view"),
                        1,
                        GL_FALSE,
                        glm.value_ptr(view),
                    )

                if self._update_model.is_set():
                    self._update_model.clear()
                    glBindBuffer(GL_ARRAY_BUFFER, model_y_vbo)
                    glBufferSubData(
                        GL_ARRAY_BUFFER,
                        0,
                        self._model_y.nbytes,
                        self._model_y,
                    )
                    self._can_update_model.set()

                glBindVertexArray(model_vao)
                glDrawElementsInstanced(
                    GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, self._instances
                )

                glfw.swap_buffers(window)
                glfw.poll_events()
                end = glfw.get_time()
                cost = end - start
                print(f"[GL] Frame cost: {cost*1000.:.2f}ms")
        finally:
            print("[GL] Terminating", flush=True)
            self._can_update_model.set()
            self._terminate.set()
            glfw.terminate()


if __name__ == "__main__":
    print(f"Using {n*n} instances", flush=True)
    app = App(n, n)
    simulation_thread = threading.Thread(target=app.simulation_thread)
    simulation_thread.start()
    app.render_until()
    simulation_thread.join()
