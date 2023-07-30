#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in float aInstanceTranslateX;
layout (location = 3) in float aInstanceTranslateZ;
layout (location = 4) in float aInstanceScaleY;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 projection;
uniform float cubeWidth;

void main()
{
    float aInstanceTranslateY = aInstanceScaleY / 2.0;
    FragPos = vec3(
        (aPos.x * cubeWidth) + aInstanceTranslateX,
        (aPos.y * aInstanceScaleY) + aInstanceTranslateY,
        (aPos.z * cubeWidth) + aInstanceTranslateZ
    );
    Normal = aNormal;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}