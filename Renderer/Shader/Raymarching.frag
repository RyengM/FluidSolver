#version 330 core

out vec4 FragColor;

in VS_OUT
{
    vec3 fragWorldPos;
    vec3 bbMinWorld;
    vec3 bbMaxWorld;
} ps_in;

struct Light
{
	vec3 pos;
	vec3 strength;
	vec3 dir;
};

struct Ray
{
	vec3 pos;
	vec3 dir;
};

Ray CreateRay(vec3 pos, vec3 dir)
{
	Ray ray;
	ray.pos = pos;
	ray.dir = dir;
	return ray;
}

uniform Light light;
uniform mat4 view;
uniform mat4 proj;
uniform vec3 eyePos;
uniform vec3 lookDir;
uniform float nearPlane, farPlane;
uniform float xResolution, yResolution;

uniform sampler3D densityTexture;
uniform sampler2D sceneDepthTexture;

#define E 2.718281828459

float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0;
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));	
}

void main()
{
	Ray ray = CreateRay(eyePos, normalize(ps_in.fragWorldPos - eyePos));

	// ray intersection
	vec3 firstIntersection = (ps_in.bbMinWorld - eyePos) / ray.dir;
	vec3 secondIntersection = (ps_in.bbMaxWorld - eyePos) / ray.dir;
	vec3 closest = min(firstIntersection, secondIntersection);
	vec3 furthest = max(firstIntersection, secondIntersection);

	// the distance between camera and hit point of the nearst and furthest plane
	float t0 = max(closest.x, max(closest.y, closest.z));
	float t1 = min(furthest.x, min(furthest.y, furthest.z));
	float tFirstHit = 0.0;

	t0 = max(0, t0);

	float sceneDepth = texture(sceneDepthTexture, vec2(gl_FragCoord.x / xResolution, gl_FragCoord.y / yResolution)).r;
	sceneDepth = LinearizeDepth(sceneDepth);

//	t1 = min(t1, sceneDepth / dot(normalize(lookDir), ray.dir));

	// the distance between entry point and out point
	float boxThickness = max(0.0, t1 - t0);
	vec3 entryPos = eyePos + t0 * ray.dir;

	if (boxThickness == 0.0) discard;

	float shadowSteps = 64.0;
	float raymarchStep = 256.0;
    vec3 bbOppsite = ps_in.bbMaxWorld - ps_in.bbMinWorld;
	// ensure ray can walk through the volume
	int maxSteps = int(boxThickness * raymarchStep) + 1;

    // assume the smoke is influenced by main light only
	vec3 lightVec = -normalize(light.dir);
	lightVec *= 1.0 / shadowSteps;
	vec3 localCamVec = ray.dir / raymarchStep / bbOppsite;

	// convert the sample range to 0-1
	vec3 curPos = (entryPos - (ps_in.bbMinWorld + ps_in.bbMaxWorld - bbOppsite) * 0.5) / bbOppsite;
	float lightEnergy = 0.0;
	float transmittance = 1.0;

	for (int i = 0; i < maxSteps; ++i)
	{	
		float cursample = texture(densityTexture, curPos).x;

		if(cursample > 0.001)
		{
			vec3 lightPos = curPos;
			float shadow = 0.0;

			// for (int s = 0; s < shadowSteps; ++s)
			// {
			// 	lightPos += lightVec;
			// 	float lightSample = texture(densityTexture, lightPos).x;

			// 	if (lightSample < 0.005f)
			// 		continue;

			// 	shadow += lightSample;
			// }

			float curdensity = clamp(cursample / 64.0, 0, 1);
			float shadowterm = exp(-shadow * 0.001);
			float absorbedlight = shadowterm * curdensity;
			lightEnergy += absorbedlight * transmittance;
			transmittance *= 1.0 - curdensity;
		}
     
		curPos += localCamVec;
	}

	vec3 color = (lightEnergy + 0.8) * 0.5 * light.strength;
	float alpha = lightEnergy * 2.0;

	FragColor = vec4(vec3(color), alpha);
}