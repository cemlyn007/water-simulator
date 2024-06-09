#version 330 core
layout(location = 0) in vec2 aPosXZ;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aTranslateY;

out vec3 FragPos;
out vec3 Normal;
out vec2 ScreenPos;

uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(aPosXZ.x, aTranslateY, aPosXZ.y);
    Normal = aNormal;

    gl_Position = projection * view * vec4(FragPos, 1.0);
    ScreenPos = vec2(0.5, 0.5) + 0.5 * vec2(gl_Position) / gl_Position.z;
}
