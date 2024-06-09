import functools
import os

import glm
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders


class Ball:
    def __init__(self, radius: float) -> None:
        vertices, indices, normals = self._create_mesh(radius)
        vertex_data = []
        for vertex, normal in zip(vertices, normals):
            vertex_data.extend(vertex)
            vertex_data.extend(normal)
        vertex_data = np.array(vertex_data, dtype=np.float32)
        self._total_indices = len(indices)
        self._vbo = self._init_vbo(glm.array(vertex_data))
        self._ebo = self._init_ebo(glm.array(indices))
        self._vao = self._init_vao(self._vbo, self._ebo)
        self._shader = self._init_shader(self._vao)
        glUseProgram(0)
        glBindVertexArray(0)

    def _create_mesh(self, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sectors = 36
        stacks = 18

        # Generate sphere mesh data
        vertices = []
        indices = []
        normals = []

        for i in range(stacks + 1):
            V = i / stacks
            phi = V * np.pi

            for j in range(sectors + 1):
                U = j / sectors
                theta = U * 2 * np.pi

                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)

                vertices.append([x, y, z])

                normal = np.array([x, y, z])
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)

        for i in range(stacks):
            for j in range(sectors):
                first = i * (sectors + 1) + j
                second = first + sectors + 1

                indices.extend(
                    [
                        first + 1,
                        second,
                        first,
                    ]
                )
                indices.extend(
                    [
                        first + 1,
                        second + 1,
                        second,
                    ]
                )

        return (
            np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
            np.array(normals, dtype=np.float32),
        )

    def _init_vbo(self, vertex: glm.array) -> None:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER, glm.sizeof(vertex), vertex.ptr, GL_STATIC_DRAW
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_ebo(self, indices: glm.array) -> None:
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
            glDeleteBuffers(1, self._ebo)
            raise exception
        return ebo

    def _init_vao(self, vbo: GLint, ebo: GLint) -> None:
        vao = glGenVertexArrays(1)
        try:
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            # Positions.
            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None
            )
            glEnableVertexAttribArray(0)
            # Normals.
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                6 * glm.sizeof(glm.float32),
                ctypes.c_void_p(3 * glm.sizeof(glm.float32)),
            )
            glEnableVertexAttribArray(1)
        except Exception as exception:
            glDeleteVertexArrays(1, self._vao)
            raise exception
        return vao

    def _init_shader(self, vao: GLint) -> GLint:
        shaders_directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders"
        )
        vertex_shader_filepath = os.path.join(shaders_directory, "simple.vs")
        with open(vertex_shader_filepath, "r") as file:
            vertex_shader_source = file.read()
            vertex_shader = shaders.compileShader(
                vertex_shader_source, GL_VERTEX_SHADER
            )
        try:
            fragment_shader_filepath = os.path.join(shaders_directory, "simple.fs")
            with open(fragment_shader_filepath, "r") as file:
                fragment_shader_source = file.read()
                fragment_shader = shaders.compileShader(
                    fragment_shader_source, GL_FRAGMENT_SHADER
                )
            try:
                glBindVertexArray(vao)
                shader = shaders.compileProgram(
                    vertex_shader, fragment_shader, validate=True
                )
            except Exception as exception:
                glDeleteShader(fragment_shader)
                raise exception
        except Exception as exception:
            glDeleteShader(vertex_shader)
            raise exception
        return shader

    def set_light_color(self, light_color: glm.vec3) -> None:
        self._light_color = light_color
        glUseProgram(self._shader)
        glUniform3f(
            self._light_color_uniform_location,
            light_color.x,
            light_color.y,
            light_color.z,
        )

    def set_view_position(self, view_position: glm.vec3) -> None:
        self._view_position = view_position
        glUseProgram(self._shader)
        glUniform3f(
            self._view_position_uniform_location,
            view_position.x,
            view_position.y,
            view_position.z,
        )

    def set_view(self, view: glm.mat4) -> None:
        self._view = view
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            self._view_uniform_location,
            1,
            GL_FALSE,
            glm.value_ptr(self._view),
        )

    def set_projection(self, projection: glm.mat4) -> None:
        self._projection = projection
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            self._projection_uniform_location,
            1,
            GL_FALSE,
            glm.value_ptr(self._projection),
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

    def set_color(self, color: glm.vec3) -> None:
        glUseProgram(self._shader)
        glUniform3fv(
            glGetUniformLocation(self._shader, "objectColor"),
            1,
            glm.value_ptr(color),
        )

    def set_model(self, model: np.ndarray) -> None:
        glUseProgram(self._shader)
        glUniformMatrix4fv(
            self._model_location,
            1,
            GL_FALSE,
            model,
        )

    def draw(self) -> None:
        glUseProgram(self._shader)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._total_indices, GL_UNSIGNED_INT, None)

    def __del__(self) -> None:
        glDeleteBuffers(1, self._vbo)
        glDeleteBuffers(1, self._ebo)
        glDeleteVertexArrays(1, self._vao)

    @property
    @functools.cache
    def _light_color_uniform_location(self) -> int:
        return glGetUniformLocation(self._shader, "lightColor")

    @property
    @functools.cache
    def _view_position_uniform_location(self) -> int:
        return glGetUniformLocation(self._shader, "viewPos")

    @property
    @functools.cache
    def _view_uniform_location(self) -> int:
        return glGetUniformLocation(self._shader, "view")

    @property
    @functools.cache
    def _projection_uniform_location(self) -> int:
        return glGetUniformLocation(self._shader, "projection")

    @property
    @functools.cache
    def _model_location(self) -> int:
        return glGetUniformLocation(self._shader, "model")
