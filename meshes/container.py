from OpenGL.GL import *
from OpenGL.GL import shaders

import numpy as np
import glm
from meshes import geometry


class Container:
    def __init__(self, size: float, wall_thickness: float) -> None:
        vertex_data, indices = self._create_mesh(size, wall_thickness)
        self._total_indices = len(indices)
        self._vbo = self._init_vbo(vertex_data)
        self._ebo = self._init_ebo(indices)
        self._vao = self._init_vao(self._vbo, self._ebo)
        self._shader = self._init_shader(self._vao)
        glUseProgram(0)
        glBindVertexArray(0)

    def _create_mesh(
        self, size: float, wall_thickness: float
    ) -> tuple[glm.array, glm.array]:
        (
            cube_vertices,
            cube_normals,
            cube_indices,
        ) = geometry.cube_vertices_normals_and_indices()
        # Plane
        vertex_data = (
            np.float32(size + wall_thickness)
            * np.array(
                [
                    # Bottom left
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    # Bottom right
                    1.0,
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    # Top right
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    # Top left
                    -1,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        ).tolist()
        indices = [2, 1, 0, 0, 3, 2]

        height_scale = 1.25

        wall_length = 2.0 * (size + wall_thickness)

        cube_vertices_shaped = np.reshape(cube_vertices, (-1, 3))
        cube_vertices_shaped *= np.array([wall_length, height_scale, wall_thickness])
        cube_vertices_shaped += np.array(
            [0.0, height_scale / 2.0, size + wall_thickness / 2.0]
        )
        cube_normals_shaped = np.reshape(cube_normals, (-1, 3))
        for vertices, normals in zip(cube_vertices_shaped, cube_normals_shaped):
            vertex_data.extend(vertices)
            vertex_data.extend(normals)

        cube_vertices_shaped = np.reshape(cube_vertices, (-1, 3))
        cube_vertices_shaped *= np.array([wall_length, height_scale, wall_thickness])
        cube_vertices_shaped += np.array(
            [0.0, height_scale / 2.0, -size - wall_thickness / 2.0]
        )
        cube_normals_shaped = np.reshape(cube_normals, (-1, 3))
        for vertices, normals in zip(cube_vertices_shaped, cube_normals_shaped):
            vertex_data.extend(vertices)
            vertex_data.extend(normals)

        cube_vertices_shaped = np.reshape(cube_vertices, (-1, 3))
        cube_vertices_shaped *= np.array([wall_thickness, height_scale, wall_length])
        cube_vertices_shaped += np.array(
            [size + wall_thickness / 2.0, height_scale / 2.0, 0.0]
        )
        cube_normals_shaped = np.reshape(cube_normals, (-1, 3))
        for vertices, normals in zip(cube_vertices_shaped, cube_normals_shaped):
            vertex_data.extend(vertices)
            vertex_data.extend(normals)

        cube_vertices_shaped = np.reshape(cube_vertices, (-1, 3))
        cube_vertices_shaped *= np.array([wall_thickness, height_scale, wall_length])
        cube_vertices_shaped += np.array(
            [-size - wall_thickness / 2.0, height_scale / 2.0, 0.0]
        )
        cube_normals_shaped = np.reshape(cube_normals, (-1, 3))
        for vertices, normals in zip(cube_vertices_shaped, cube_normals_shaped):
            vertex_data.extend(vertices)
            vertex_data.extend(normals)

        offset = max(indices) + 1
        for i in range(4):
            indices.extend(
                [i * (max(cube_indices) + 1) + index + offset for index in cube_indices]
            )

        vertex_data = glm.array(
            np.array(vertex_data, dtype=np.float32), dtype=glm.float32
        )
        indices = glm.array(np.array(indices, dtype=np.uint32), dtype=glm.uint32)
        return vertex_data, indices

    def _init_vbo(self, vertex: glm.array) -> None:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER, glm.sizeof(vertex), vertex.ptr, GL_STATIC_DRAW
            )
        except Exception as exception:
            glDeleteBuffers(1, self._vbo)
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
        with open("shaders/container.vs", "r") as file:
            vertex_shader_source = file.read()
            vertex_shader = shaders.compileShader(
                vertex_shader_source, GL_VERTEX_SHADER
            )
        try:
            with open("shaders/container.fs", "r") as file:
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
        glDrawElements(GL_TRIANGLES, self._total_indices, GL_UNSIGNED_INT, None)

    def __del__(self) -> None:
        glDeleteBuffers(1, self._vbo)
        glDeleteBuffers(1, self._ebo)
        glDeleteVertexArrays(1, self._vao)
