from OpenGL.GL import *
from OpenGL.GL import shaders

import numpy as np
import glm


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


class Light:
    def __init__(self) -> None:
        vertices, _, indices = cube_vertices_normals_and_indices()
        vertex_data = glm.array(np.array(vertices, dtype=np.float32))

        self._vbo = self._init_vbo(vertex_data)
        self._ebo = self._init_ebo(
            glm.array(np.array(indices, dtype=np.uint32), dtype=glm.uint32)
        )
        self._vao = self._init_vao(self._vbo, self._ebo, vertex_data)
        self._shader = self._init_shader(self._vao)

    def _init_vbo(self, vertex_data: glm.array) -> GLint:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                glm.sizeof(vertex_data),
                vertex_data.ptr,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_ebo(self, indices: glm.array) -> GLint:
        ebo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER,
                glm.sizeof(indices),
                indices.ptr,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, ebo)
            raise exception
        return ebo

    def _init_vao(self, vbo: GLint, ebo: GLint, vertex_data: glm.array) -> GLint:
        vao = glGenVertexArrays(1)
        try:
            glBindVertexArray(vao)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)

            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, 3 * vertex_data.dt_size, ctypes.c_void_p(0)
            )
            glEnableVertexAttribArray(0)
        except Exception as exception:
            glDeleteVertexArrays(1, vao)
            raise exception
        return vao

    def _init_shader(self, vao: GLint) -> GLint:
        with open("shaders/light_cube.vs", "r") as file:
            vertex_shader_source = file.read()
            light_cube_vertex_shader = shaders.compileShader(
                vertex_shader_source, GL_VERTEX_SHADER
            )
        try:
            with open("shaders/light_cube.fs", "r") as file:
                fragment_shader_source = file.read()
                light_cube_fragment_shader = shaders.compileShader(
                    fragment_shader_source, GL_FRAGMENT_SHADER
                )
            try:
                glBindVertexArray(vao)
                light_cube_shader = shaders.compileProgram(
                    light_cube_vertex_shader, light_cube_fragment_shader, validate=True
                )
            except Exception as exception:
                glDeleteShader(light_cube_fragment_shader)
                raise exception
        except Exception as exception:
            glDeleteShader(light_cube_vertex_shader)
            raise exception
        return light_cube_shader

    def set_view(self, view: glm.mat4) -> None:
        self._view = view
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "view"),
            1,
            GL_FALSE,
            glm.value_ptr(self._view),
        )

    def set_projection(self, projection: glm.mat4) -> None:
        self._projection = projection
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "projection"),
            1,
            GL_FALSE,
            glm.value_ptr(self._projection),
        )

    def set_color(self, color: glm.vec3) -> None:
        glUseProgram(self._shader)
        glUniform3fv(
            glGetUniformLocation(self._shader, "objectColor"),
            1,
            glm.value_ptr(color),
        )

    def set_model(self, model: glm.mat4) -> None:
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "model"),
            1,
            GL_FALSE,
            glm.value_ptr(model),
        )

    def draw(self) -> None:
        glUseProgram(self._shader)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

    def __del__(self) -> None:
        glDeleteProgram(self._shader)
        glDeleteBuffers(1, self._vbo)
        glDeleteBuffers(1, self._vao)
        glDeleteBuffers(1, self._ebo)


class Water:
    def __init__(self, n: int, m: int, cube_width: float) -> None:
        self._n = n
        self._m = m
        self._cube_width = cube_width
        vertices, normals, indices = cube_vertices_normals_and_indices()
        vertex_data = []
        for vertex, normal in zip(vertices, normals):
            vertex_data.extend(vertex)
            vertex_data.extend(normal)
        vertex_data = glm.array(np.array(vertex_data, dtype=np.float32))

        self._cube_vbo = self._init_vbo(vertex_data)
        self._ebo = self._init_ebo(
            glm.array(np.array(indices, dtype=np.uint32), dtype=glm.uint32)
        )

        self._model_scale_xz_vbo = self._init_model_scale_xz_vbo()
        self._water_positions_xz_vbo = self._init_model_translate_xz_vbo()
        self._water_height_vbo = self._init_model_y_vbo()

        self._vao = self._init_vao(
            self._cube_vbo,
            self._water_positions_xz_vbo,
            self._water_height_vbo,
            self._ebo,
            vertex_data,
        )

        self._shader = self._init_shader(self._vao)
        glUseProgram(self._shader)
        glUniform1f(glGetUniformLocation(self._shader, "cubeWidth"), self._cube_width)

    def _init_vbo(self, vertex_data: glm.array) -> GLint:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                glm.sizeof(vertex_data),
                vertex_data.ptr,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_model_scale_xz_vbo(self) -> GLint:
        vbo = glGenBuffers(1)
        try:
            xz_scale = np.array(self._cube_width, dtype=np.float32)
            xz_scale = np.tile(xz_scale, self._n * self._m)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                xz_scale.nbytes,
                xz_scale,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_model_translate_xz_vbo(self) -> GLint:
        vbo = glGenBuffers(1)
        try:
            xz_translate = self._get_xz_translate(self._n, self._m, self._cube_width)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                xz_translate.nbytes,
                xz_translate,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_model_y_vbo(self) -> GLint:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            # First half will be y translations, second half will be y scales.
            glBufferData(
                GL_ARRAY_BUFFER,
                self._n * self._m * glm.sizeof(glm.float32),
                None,
                GL_DYNAMIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_ebo(self, indices: glm.array) -> GLint:
        ebo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER,
                glm.sizeof(indices),
                indices.ptr,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, ebo)
            raise exception
        return ebo

    def _init_vao(
        self,
        cube_vbo: GLint,
        model_translate_xz_vbo: GLint,
        model_y_vbo: GLint,
        ebo: GLint,
        vertex_data: glm.array,
    ) -> GLint:
        vao = glGenVertexArrays(1)
        try:
            glBindVertexArray(vao)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

            glBindBuffer(GL_ARRAY_BUFFER, cube_vbo)
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
                2 * np.float32(0.0).itemsize,
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
                2 * np.float32(0.0).itemsize,
                ctypes.c_void_p(1 * np.float32(0.0).itemsize),
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
                np.float32(0.0).itemsize,
                None,
            )
            glEnableVertexAttribArray(4)
            glVertexAttribDivisor(4, 1)
        except Exception as exception:
            glDeleteVertexArrays(1, vao)
            raise exception
        return vao

    def _init_shader(self, vao: GLint) -> GLint:
        with open("shaders/basic_lighting.vs", "r") as file:
            vertex_shader_source = file.read()
            light_cube_vertex_shader = shaders.compileShader(
                vertex_shader_source, GL_VERTEX_SHADER
            )
        try:
            with open("shaders/basic_lighting.fs", "r") as file:
                fragment_shader_source = file.read()
                light_cube_fragment_shader = shaders.compileShader(
                    fragment_shader_source, GL_FRAGMENT_SHADER
                )
            try:
                glBindVertexArray(vao)
                light_cube_shader = shaders.compileProgram(
                    light_cube_vertex_shader, light_cube_fragment_shader, validate=True
                )
            except Exception as exception:
                glDeleteShader(light_cube_fragment_shader)
                raise exception
        except Exception as exception:
            glDeleteShader(light_cube_vertex_shader)
            raise exception
        return light_cube_shader

    def set_water_color(self, water_color: glm.vec4) -> None:
        self._water_color = water_color
        glUseProgram(self._shader)
        glUniform4f(
            glGetUniformLocation(self._shader, "objectColor"),
            water_color.x,
            water_color.y,
            water_color.z,
            water_color.w,
        )

    # TODO: Set methods.
    def set_light_color(self, light_color: glm.vec3) -> None:
        self._light_color = light_color
        glUseProgram(self._shader)
        glUniform3f(
            glGetUniformLocation(self._shader, "lightColor"),
            light_color.x,
            light_color.y,
            light_color.z,
        )

    def set_view_position(self, view_position: glm.vec3) -> None:
        self._view_position = view_position
        glUseProgram(self._shader)
        glUniform3f(
            glGetUniformLocation(self._shader, "viewPos"),
            view_position.x,
            view_position.y,
            view_position.z,
        )

    def set_view(self, view: glm.mat4) -> None:
        self._view = view
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "view"),
            1,
            GL_FALSE,
            glm.value_ptr(view),
        )

    def set_projection(self, projection: glm.mat4) -> None:
        self._projection = projection
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "projection"),
            1,
            GL_FALSE,
            glm.value_ptr(projection),
        )

    def set_light_position(self, light_position: glm.vec3) -> None:
        self._light_position = light_position
        glUseProgram(self._shader)
        glUniform3f(
            glGetUniformLocation(self._shader, "lightPos"),
            light_position.x,
            light_position.y,
            light_position.z,
        )

    def set_water_heights(self, water_heights: glm.array) -> None:
        self._water_heights = water_heights
        glBindBuffer(GL_ARRAY_BUFFER, self._water_height_vbo)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            glm.sizeof(water_heights),
            water_heights.ptr,
        )

    def draw(self) -> None:
        glUseProgram(self._shader)
        glBindVertexArray(self._vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self._n * self._m)

    def _get_xz_translate(
        self, n: int, m: int, cube_width: float
    ) -> np.ndarray[any, np.float32]:
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

    def __del__(self) -> None:
        glDeleteProgram(self._shader)
        glDeleteVertexArrays(1, self._vao)
        glDeleteBuffers(1, self._water_height_vbo)
        glDeleteBuffers(1, self._water_positions_xz_vbo)
        glDeleteBuffers(1, self._ebo)
        glDeleteBuffers(1, self._cube_vbo)
