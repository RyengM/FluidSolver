#version 330 core
out vec4 FragColor;

in vec3 fragWorldPos;

uniform vec3 camPos;

struct Ray
{
	vec3 pos;
	vec3 dir;
};

struct RayHit
{
	vec3 pos;
	vec3 normal;
};

Ray CreateRay(vec3 pos, vec3 dir)
{
	Ray ray;
	ray.pos = pos;
	ray.dir = dir;
	return ray;
}

RayHit CreateRayHit()
{
	RayHit rayHit;
	rayHit.pos = vec3(0, 0, 0);
	rayHit.normal = vec3(0, 0, 0);
	return rayHit;
}

float sdfSphere(vec3 pos, float radius)
{
	return length(pos) - radius;
}

float sdfField(vec3 pos)
{
	return sdfSphere(pos, 0.1);
}

vec3 getNormal(vec3 pos)
{
	vec2 offset = vec2(0.001f,0.0f);
    vec3 n= vec3(
        sdfField(pos+offset.xyy)-sdfField(pos.xyy),
        sdfField(pos+offset.yxy)-sdfField(pos.yxy),
        sdfField(pos+offset.yyx)-sdfField(pos.yyx)
    );
    return normalize(n);
}

void main()
{
	Ray ray = CreateRay(camPos, normalize(fragWorldPos - camPos));
	RayHit rayHit = CreateRayHit();

	float len = 0;
	vec3 res = vec3(0, 0, 0);
	for (int i = 0; i < 20; ++i)
	{
		vec3 p = ray.pos + ray.dir * len;
		float d = sdfField(p);

		if (d < 0.05)
		{
			rayHit.pos = p;
			rayHit.normal = getNormal(p);
			res = vec3(1, 1, 1);
			break;
		}

		len += d;
	}

	FragColor = vec4(res, 1.f);
}