from OpenGL.GL import *
from OpenGL.GL import shaders

import numpy as np
import glm
from meshes import geometry


class Light:
    def __init__(self) -> None:
        vertices, _, indices = geometry.cube_vertices_normals_and_indices()
        vertex_data = glm.array(np.array(vertices, dtype=np.float32))

        self._vbo = self._init_vbo(vertex_data)
        self._ebo = self._init_ebo(
            glm.array(np.array(indices, dtype=np.uint32), dtype=glm.uint32)
        )
        self._vao = self._init_vao(self._vbo, self._ebo, vertex_data)
        self._shader = self._init_shader(self._vao)
        glUseProgram(0)
        glBindVertexArray(0)

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
