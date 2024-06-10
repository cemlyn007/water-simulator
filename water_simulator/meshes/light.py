import os

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from water_simulator.meshes import geometry


class Light:
    def __init__(self) -> None:
        self._shader = None
        vertices, _, indices = geometry.cube_vertices_normals_and_indices()
        vertex_data = np.array(vertices, dtype=np.float32)

        self._vbo = self._init_vbo(vertex_data)
        self._ebo = self._init_ebo(np.array(indices, dtype=np.uint32))
        self._vao = self._init_vao(self._vbo, self._ebo, vertex_data)
        self._shader = self._init_shader(self._vao)
        glUseProgram(0)
        glBindVertexArray(0)

    def _init_vbo(self, vertex_data: np.ndarray) -> GLint:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                vertex_data.nbytes,
                vertex_data,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_ebo(self, indices: np.ndarray) -> GLint:
        ebo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER,
                indices.nbytes,
                indices,
                GL_STATIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, ebo)
            raise exception
        return ebo

    def _init_vao(self, vbo: GLint, ebo: GLint, vertex_data: np.ndarray) -> GLint:
        vao = glGenVertexArrays(1)
        try:
            glBindVertexArray(vao)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)

            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, 3 * vertex_data.itemsize, ctypes.c_void_p(0)
            )
            glEnableVertexAttribArray(0)
        except Exception as exception:
            glDeleteVertexArrays(1, vao)
            raise exception
        return vao

    def _init_shader(self, vao: GLint) -> GLint:
        shaders_directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders"
        )
        vertex_shader_filepath = os.path.join(shaders_directory, "light_cube.vs")
        with open(vertex_shader_filepath, "r") as file:
            vertex_shader_source = file.read()
            light_cube_vertex_shader = shaders.compileShader(
                vertex_shader_source, GL_VERTEX_SHADER
            )
        try:
            fragment_shader_filepath = os.path.join(shaders_directory, "light_cube.fs")
            with open(fragment_shader_filepath, "r") as file:
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

    def set_view(self, view: np.ndarray) -> None:
        if view.size != 16:
            raise ValueError("Expected 16 elements")
        # else...
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "view"),
            1,
            GL_FALSE,
            view,
        )

    def set_projection(self, projection: np.ndarray) -> None:
        if projection.size != 16:
            raise ValueError("Expected 16 elements")
        # else...
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "projection"),
            1,
            GL_FALSE,
            projection,
        )

    def set_color(self, color: np.ndarray) -> None:
        if color.size != 3:
            raise ValueError("Expected 3 elements")
        glUseProgram(self._shader)
        glUniform3fv(
            glGetUniformLocation(self._shader, "objectColor"),
            1,
            color,
        )

    def set_model(self, model: np.ndarray) -> None:
        if model.size != 16:
            raise ValueError("Expected 16 elements")
        # else...
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            glGetUniformLocation(self._shader, "model"),
            1,
            GL_FALSE,
            model,
        )

    def draw(self) -> None:
        glUseProgram(self._shader)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

    def __del__(self) -> None:
        if self._shader is not None:
            glDeleteProgram(self._shader)
        glDeleteBuffers(1, self._vbo)
        glDeleteBuffers(1, self._vao)
        glDeleteBuffers(1, self._ebo)
