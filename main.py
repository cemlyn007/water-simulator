import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import sys
import numpy as np
import glm


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def get_instance_model(i: int, j: int, t: float, n: int, m: int) -> glm.mat4:
    smoothing = 0.1
    scale_width = 0.05
    full_width = 1.0 * scale_width
    half_width = 0.5 * scale_width
    scale_vector = glm.vec3(
        scale_width,
        1.5 * np.sin(np.radians(smoothing * (1 + i + j) * t, dtype=np.float32)),
        scale_width,
    )
    model = glm.translate(
        glm.vec3(
            full_width * i - ((full_width * n) / 2.0) + half_width,
            scale_vector.y / 2,
            full_width * j - ((full_width * m) / 2.0) + half_width,
        ),
    )
    model = glm.scale(
        model,
        scale_vector,
    )
    return model


if __name__ == "__main__":
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(800, 600, "LearnOpenGL", None, None)

    if window is None:
        print("Failed to create GLFW window")
        glfw.terminate()
        sys.exit(-1)

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    # glfw.set_cursor_pos_callback(window, mouse_callback)
    # glfw.set_scroll_callback(window, scroll_callback)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glEnable(GL_DEPTH_TEST)

    with open("shaders/basic_lighting.vs", "r") as file:
        vertex_shader_source = file.read()
        vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
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

    vertices = np.array(
        [
            -0.5,
            -0.5,
            -0.5,
            0.0,
            0.0,
            -1.0,
            0.5,
            -0.5,
            -0.5,
            0.0,
            0.0,
            -1.0,
            0.5,
            0.5,
            -0.5,
            0.0,
            0.0,
            -1.0,
            0.5,
            0.5,
            -0.5,
            0.0,
            0.0,
            -1.0,
            -0.5,
            0.5,
            -0.5,
            0.0,
            0.0,
            -1.0,
            -0.5,
            -0.5,
            -0.5,
            0.0,
            0.0,
            -1.0,
            -0.5,
            -0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            0.5,
            -0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            -0.5,
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            -0.5,
            -0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            -0.5,
            0.5,
            0.5,
            -1.0,
            0.0,
            0.0,
            -0.5,
            0.5,
            -0.5,
            -1.0,
            0.0,
            0.0,
            -0.5,
            -0.5,
            -0.5,
            -1.0,
            0.0,
            0.0,
            -0.5,
            -0.5,
            -0.5,
            -1.0,
            0.0,
            0.0,
            -0.5,
            -0.5,
            0.5,
            -1.0,
            0.0,
            0.0,
            -0.5,
            0.5,
            0.5,
            -1.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            0.5,
            -0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            -0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            -0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            1.0,
            0.0,
            0.0,
            -0.5,
            -0.5,
            -0.5,
            0.0,
            -1.0,
            0.0,
            0.5,
            -0.5,
            -0.5,
            0.0,
            -1.0,
            0.0,
            0.5,
            -0.5,
            0.5,
            0.0,
            -1.0,
            0.0,
            0.5,
            -0.5,
            0.5,
            0.0,
            -1.0,
            0.0,
            -0.5,
            -0.5,
            0.5,
            0.0,
            -1.0,
            0.0,
            -0.5,
            -0.5,
            -0.5,
            0.0,
            -1.0,
            0.0,
            -0.5,
            0.5,
            -0.5,
            0.0,
            1.0,
            0.0,
            0.5,
            0.5,
            -0.5,
            0.0,
            1.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.0,
            1.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.0,
            1.0,
            0.0,
            -0.5,
            0.5,
            0.5,
            0.0,
            1.0,
            0.0,
            -0.5,
            0.5,
            -0.5,
            0.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    vbo = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    n = 10
    m = 10
    n_instances = n * m

    model_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, model_vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        glm.sizeof(glm.array([glm.mat4(1.0)] * n_instances, dtype=glm.mat4)),
        None,
        GL_DYNAMIC_DRAW,
    )

    instance_models = []
    model_vaos = []
    for instance_index in range(n_instances):
        model_vao = glGenVertexArrays(1)
        glBindVertexArray(model_vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0)
        )
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(
            1,
            3,
            GL_FLOAT,
            GL_FALSE,
            6 * vertices.itemsize,
            ctypes.c_void_p(3 * vertices.itemsize),
        )
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, model_vbo)
        for i in range(4):
            index = 2 + i
            glVertexAttribPointer(
                index,
                4,
                GL_FLOAT,
                GL_FALSE,
                glm.sizeof(glm.mat4(1.0)),
                ctypes.c_void_p(i * glm.sizeof(glm.vec4(1.0))),
            )
            glVertexAttribDivisor(index, 1)
            glEnableVertexAttribArray(index)
        instance_models.append(glm.mat4(1.0))
        model_vaos.append(model_vao)

    instance_models_data = glm.array(instance_models, dtype=glm.mat4)

    lighting_shader = shaders.compileProgram(
        vertex_shader, fragment_shader, validate=True
    )

    light_cube_vao = glGenVertexArrays(1)
    glBindVertexArray(light_cube_vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0)
    )
    glEnableVertexAttribArray(0)

    light_cube_shader = shaders.compileProgram(
        light_cube_vertex_shader, light_cube_fragment_shader, validate=True
    )

    delta_time = 0.0
    last_frame = 0.0
    light_pos = glm.vec3(1.2, 1.0, 2.0)

    camera_position = glm.vec3(3.0, 1.0, 3.0)

    while not glfw.window_should_close(window):
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        # process_input(window)

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)
        view = glm.lookAt(
            camera_position,
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0),
        )

        glUseProgram(light_cube_shader)
        glUniformMatrix4fv(
            glGetUniformLocation(light_cube_shader, "projection"),
            1,
            GL_FALSE,
            glm.value_ptr(projection),
        )
        glUniformMatrix4fv(
            glGetUniformLocation(light_cube_shader, "view"),
            1,
            GL_FALSE,
            glm.value_ptr(view),
        )
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(light_pos[0], light_pos[1], light_pos[2]))
        model = glm.scale(model, glm.vec3(0.2))

        glUniformMatrix4fv(
            glGetUniformLocation(light_cube_shader, "model"),
            1,
            GL_FALSE,
            glm.value_ptr(model),
        )
        glUniform3f(
            glGetUniformLocation(light_cube_shader, "objectColor"), 1.0, 1.0, 1.0
        )

        glBindVertexArray(light_cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        glUseProgram(lighting_shader)
        glUniform3f(glGetUniformLocation(lighting_shader, "objectColor"), 0.0, 0.2, 1.0)
        glUniform3f(glGetUniformLocation(lighting_shader, "lightColor"), 1.0, 1.0, 1.0)
        glUniform3f(
            glGetUniformLocation(lighting_shader, "lightPos"),
            light_pos[0],
            light_pos[1],
            light_pos[2],
        )

        glUniformMatrix4fv(
            glGetUniformLocation(lighting_shader, "projection"),
            1,
            GL_FALSE,
            glm.value_ptr(projection),
        )
        glUniformMatrix4fv(
            glGetUniformLocation(lighting_shader, "view"),
            1,
            GL_FALSE,
            glm.value_ptr(view),
        )
        instance_index = 0
        instance_models = []
        for i in range(n):
            for j in range(m):
                model = get_instance_model(i, j, current_frame, n, m)
                instance_models.append(model)
                instance_index += 1
        glBindBuffer(GL_ARRAY_BUFFER, model_vbo)
        instance_models_data = glm.array(instance_models, dtype=glm.mat4)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            glm.sizeof(instance_models_data),
            instance_models_data.ptr,
        )

        for instance_vao in model_vaos:
            glBindVertexArray(instance_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, 36, n_instances)

        glfw.swap_buffers(window)
        glfw.poll_events()
