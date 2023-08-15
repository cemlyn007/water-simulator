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


def grid_vertices_normals_and_indices(n: int, m: int, cell_size: float):
    vertices = []
    for i in range(n):
        for j in range(m):
            vertices.append(
                (
                    i * cell_size,
                    j * cell_size,
                )
            )

    triangle_vertices = np.array(vertices, dtype=np.float32)
    triangle_vertices = triangle_vertices.reshape((n, m, 2))

    triangle_indices = []
    for i in range(n - 1):
        for j in range(m - 1):
            triangle_indices.extend(
                [
                    i * m + j,
                    i * m + j + 1,
                    (i + 1) * m + j + 1,
                    (i + 1) * m + j + 1,
                    (i + 1) * m + j,
                    i * m + j,
                ]
            )

    triangle_vertices = triangle_vertices.reshape((-1, 2))
    triangle_vertices -= np.array(
        [(n - 1) * cell_size / 2.0, (m - 1) * cell_size / 2.0], dtype=np.float32
    )
    triangle_indices = np.array(triangle_indices, dtype=np.uint32)
    triangle_normals = np.array([[0, 1, 0]] * len(triangle_vertices), dtype=np.float32)
    return triangle_vertices, triangle_normals, triangle_indices


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


class Water:
    def __init__(self, n: int, m: int, cell_width: float) -> None:
        self._n = n
        self._m = m
        self._cell_width = cell_width
        vertices, normals, indices = grid_vertices_normals_and_indices(n, m, cell_width)
        self._n_vertices = len(vertices)
        vertex_data = []
        for vertex, normal in zip(vertices, normals):
            vertex_data.extend(vertex)
            vertex_data.extend(normal)
        vertex_data = glm.array(np.array(vertex_data, dtype=np.float32))

        self._vbo = self._init_vbo(vertex_data)
        self._indices = glm.array(indices, dtype=glm.uint32)
        self._ebo = self._init_ebo(self._indices)

        self._height_vbo = self._init_model_y_vbo()

        self._vao = self._init_vao(
            self._vbo,
            self._height_vbo,
            self._ebo,
        )

        self._shader = self._init_shader(self._vao)
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
                5 * np.float32(0.0).itemsize,
                ctypes.c_void_p(0),
            )
            glEnableVertexAttribArray(0)

            # Contains normals.
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                5 * np.float32(0.0).itemsize,
                ctypes.c_void_p(2 * np.float32(0.0).itemsize),
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

    def draw(self) -> None:
        glUseProgram(self._shader)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._len_ebo, GL_UNSIGNED_INT, None)

    def __del__(self) -> None:
        glDeleteProgram(self._shader)
        glDeleteVertexArrays(1, self._vao)
        glDeleteBuffers(1, self._height_vbo)
        glDeleteBuffers(1, self._vbo)
        glDeleteBuffers(1, self._ebo)


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
        cube_vertices, cube_normals, cube_indices = cube_vertices_normals_and_indices()
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
