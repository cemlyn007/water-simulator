from OpenGL.GL import *


class Camera:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffer)
        self.rendered_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.rendered_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self._width,
            self._height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        self._depth_render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_render_buffer)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self._width, self._height
        )
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER,
            self._depth_render_buffer,
        )

        glFramebufferTexture(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.rendered_texture, 0
        )
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Failed to create camera")
        # else...
        self.unbind()

    def resize(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffer)
        glBindTexture(GL_TEXTURE_2D, self.rendered_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self._width,
            self._height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        self._depth_render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_render_buffer)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self._width, self._height
        )
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER,
            self._depth_render_buffer,
        )

        glFramebufferTexture(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.rendered_texture, 0
        )

    def bind(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffer)
        glViewport(0, 0, self._width, self._height)

    def unbind(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def __del__(self) -> None:
        glDeleteRenderbuffers(1, [self._depth_render_buffer])
        glDeleteFramebuffers(1, [self._framebuffer])
        glDeleteTextures(1, [self.rendered_texture])
