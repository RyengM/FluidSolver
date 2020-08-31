#version 430 core
layout (location = 0) out vec4 color;
uniform sampler2D rayDir; 
 
void main(void) 
{
	color = texture(rayDir, vec2(gl_FragCoord.xy) / vec2(textureSize(rayDir, 0)));
}