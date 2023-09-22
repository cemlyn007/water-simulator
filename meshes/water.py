from OpenGL.GL import *
from OpenGL.GL import shaders

import numpy as np
import glm
from meshes import geometry
import jax
import jax.numpy as jnp


class Water:
    def __init__(self, n: int, m: int, cell_width: float) -> None:
        self._n = n
        self._m = m
        self._cell_width = cell_width
        vertices, normals, indices = geometry.grid_vertices_normals_and_indices(
            n, m, cell_width
        )
        self._n_vertices = len(vertices)
        vertex_data = []
        normal_data = []
        for vertex, normal in zip(vertices, normals):
            vertex_data.extend(vertex)
            normal_data.extend(normal)
        self._xz = np.array(vertex_data, dtype=np.float32)
        vertex_data = glm.array(np.array(vertex_data, dtype=np.float32))
        normal_data = glm.array(np.array(normal_data, dtype=np.float32))

        self._normal_vbo = self._init_normal_vbo(normal_data)
        self._vbo = self._init_vbo(vertex_data)
        self._indices = glm.array(indices, dtype=glm.uint32)
        self._ebo = self._init_ebo(self._indices)

        self._height_vbo = self._init_model_y_vbo()

        self._vao = self._init_vao(
            self._vbo,
            self._height_vbo,
            self._normal_vbo,
            self._ebo,
        )

        self._shader = self._init_shader(self._vao)
        glBindVertexArray(0)

        self._vertex_normal_updater = jax.jit(
            WaterVertexNormalUpdater(n, m, self._xz, self._indices)
        )

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

    def _init_model_y_vbo(self) -> GLint:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                self._n_vertices * glm.sizeof(glm.float32),
                None,
                GL_DYNAMIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_normal_vbo(self, normal_data: glm.array) -> GLint:
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                glm.sizeof(normal_data),
                normal_data.ptr,
                GL_DYNAMIC_DRAW,
            )
        except Exception as exception:
            glDeleteBuffers(1, vbo)
            raise exception
        return vbo

    def _init_ebo(self, indices: glm.array) -> GLint:
        self._len_ebo = len(indices)
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
        vbo: GLint,
        height_vbo: GLint,
        normal_vbo: GLint,
        ebo: GLint,
    ) -> GLint:
        vao = glGenVertexArrays(1)
        try:
            glBindVertexArray(vao)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

            # Contains X and Z coordinates.
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glVertexAttribPointer(
                0,
                2,
                GL_FLOAT,
                GL_FALSE,
                2 * np.float32(0.0).itemsize,
                None,
            )
            glEnableVertexAttribArray(0)

            glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                3 * np.float32(0.0).itemsize,
                None,
            )
            glEnableVertexAttribArray(1)

            # Contains Y coordinates.
            glBindBuffer(GL_ARRAY_BUFFER, height_vbo)
            glVertexAttribPointer(
                2,
                1,
                GL_FLOAT,
                GL_FALSE,
                np.float32(0.0).itemsize,
                None,
            )
            glEnableVertexAttribArray(2)
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

    def set_light_color(self, light_color: glm.vec3) -> None:
        self._light_color = light_color
        glUseProgram(self._shader)
        glUniform3f(
            glGetUniformLocation(self._shader, "lightColor"),
            light_color.x,
            light_color.y,
            light_color.z,
        )

    def set_texture(self, texture: int) -> None:
        glUseProgram(self._shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(glGetUniformLocation(self._shader, "background"), 0)

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
        if len(water_heights) != self._n_vertices:
            raise ValueError(
                f"Water heights array must have {self._n_vertices} elements."
            )
        # else...
        glBindBuffer(GL_ARRAY_BUFFER, self._height_vbo)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            glm.sizeof(water_heights),
            water_heights.ptr,
        )

        np_water_heights = np.array(water_heights)
        arr = self._vertex_normal_updater(np_water_heights)
        water_normals = glm.array(np.asarray(arr))
        glBindBuffer(GL_ARRAY_BUFFER, self._normal_vbo)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            glm.sizeof(water_normals),
            water_normals.ptr,
        )

    def draw(self) -> None:
        glUseProgram(self._shader)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._len_ebo, GL_UNSIGNED_INT, None)

    def __del__(self) -> None:
        glDeleteProgram(self._shader)
        glDeleteVertexArrays(1, self._vao)
        glDeleteBuffers(1, self._height_vbo)
        glDeleteBuffers(1, self._vbo)
        glDeleteBuffers(1, self._normal_vbo)
        glDeleteBuffers(1, self._ebo)


class WaterVertexNormalUpdater:
    def __init__(self, n: int, m: int, xz: jax.Array, indices: jax.Array) -> None:
        self._n = n
        self._m = m
        self._n_faces = (self._n - 1) * (self._m - 1) * 2
        self._xz = jnp.reshape(xz, (self._n * self._m, 2))
        self._indices = jnp.array(indices)
        (
            self._element_index_map,
            self._face_indices,
            self._vertex_normal_groups_size,
        ) = self._prepare()

    def __call__(self, water_heights: jax.Array) -> jax.Array:
        return self._compute(water_heights)

    def _prepare(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertex_normal_groups_size = np.zeros((self._n * self._m,), dtype=np.int32)
        element_index_map = np.zeros((self._n_faces * 3,), dtype=np.int32)
        face_indices = self._indices.reshape(self._n_faces, 3)

        for i, vertex_index in enumerate(self._indices):
            element_index_map[i] = vertex_normal_groups_size[vertex_index]
            vertex_normal_groups_size[vertex_index] += 1

        return (
            element_index_map.reshape(-1, 3),
            face_indices,
            vertex_normal_groups_size,
        )

    def _compute(self, water_heights: jax.Array) -> jax.Array:
        face_normals = self._calculate_face_normals(water_heights)
        vertex_normal_groups = self._vertex_face_normals(face_normals)
        vertex_normals = jnp.divide(
            jnp.sum(vertex_normal_groups, axis=1),
            self._vertex_normal_groups_size[:, None],
        )
        return vertex_normals.flatten()

    def _vertex_face_normals(self, face_normals: jax.Array) -> jax.Array:
        vertex_normal_groups = jnp.zeros(
            (self._n * self._m, max(self._vertex_normal_groups_size), 3),
            dtype=jnp.float32,
        )
        vertex_normal_groups = vertex_normal_groups.at[
            self._face_indices, self._element_index_map
        ].set(face_normals[:, None, :])
        return vertex_normal_groups

    def _calculate_face_normals(self, water_heights: jax.Array) -> jax.Array:
        faces_vertices = jnp.empty((self._n_faces, 3, 3), dtype=jnp.float32)
        faces_vertices = faces_vertices.at[:, :, jnp.array([0, 2])].set(
            self._xz[self._face_indices]
        )
        faces_vertices = faces_vertices.at[:, :, 1].set(
            water_heights[self._face_indices]
        )
        normals = jnp.cross(
            faces_vertices[:, 1, :] - faces_vertices[:, 0, :],
            faces_vertices[:, 2, :] - faces_vertices[:, 0, :],
        )
        normals /= jnp.linalg.norm(normals, axis=1, keepdims=True)
        return normals
