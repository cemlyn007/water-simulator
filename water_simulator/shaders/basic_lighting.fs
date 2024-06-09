#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 ScreenPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform sampler2D background;

void main() {
    // refraction effect
    float r = 0.02;	// todo: should be distance dependent!
    vec2 uv = ScreenPos + r * vec2(Normal.x, Normal.z);
    vec3 color = texture(background, uv).xyz;
    color.z = min(color.z + 0.2, 1.0);
    vec3 L = normalize(vec3(10.0, 10.0, 10.0) - FragPos);
    float s = max(dot(Normal, L), 0.0);
    color *= (0.5 + 0.5 * s);

    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular
    float specularStrength = 0.9;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 256);
    vec3 specular = specularStrength * spec * lightColor;

    vec4 result = vec4(ambient + diffuse + specular, 1.0) * vec4(color, 1.0);
    FragColor = result;
}
