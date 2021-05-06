#version 330 core

layout (location = 0) in vec3 lPos;

out VS_OUT
{
    vec3 fragWorldPos;
    vec3 bbMinWorld;
    vec3 bbMaxWorld;
} vs_out;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
uniform vec3 bbMin;
uniform vec3 bbMax;

void main()
{
    gl_Position = proj * view * model * vec4(lPos, 1.f);
    vs_out.fragWorldPos = vec3(model * vec4(lPos, 1.f));
    vs_out.bbMinWorld = vec3(model * vec4(bbMin, 1.f));
    vs_out.bbMaxWorld = vec3(model * vec4(bbMax, 1.f));
}