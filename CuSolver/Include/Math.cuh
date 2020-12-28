#include "CudaUtility.h"

#define PI 3.1415926535

static __device__ int3 combine_int3(int a, int b, int c)
{
	int3 res;
	res.x = a, res.y = b, res.z = c;
	return res;
}

static __device__ float3 combine_float3(float a, float b, float c)
{
	float3 res;
	res.x = a, res.y = b, res.z = c;
	return res;
}

template<typename T>
static __device__ T min(T a, T b)
{
	return a < b ? a : b;
}

template<typename T>
static __device__ T max(T a, T b)
{
	return a > b ? a : b;
}

static __device__ int3 minmax(int3 pos, int3 max_pos)
{
	pos.x = max(0, min(pos.x, max_pos.x - 1));
	pos.y = max(0, min(pos.y, max_pos.y - 1));
	pos.z = max(0, min(pos.z, max_pos.z - 1));
	return pos;
}

static __device__ float length(float3 f)
{
	return sqrt(pow(f.x, 2) + pow(f.y, 2) + pow(f.z, 2));
}

static __device__ float3 normalize(float3 f)
{
	float len = length(f) + 1e-5;
	return combine_float3(f.x / len, f.y / len, f.z / len);
}

static __device__ float3 cross(float3 a, float3 b)
{
	return combine_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static __device__ float lerp(float a, float b, float s)
{
	return a * (1 - s) + b * s;
}

static __device__ float getRandom(unsigned int *seed0, unsigned int *seed1) {
	// hash the seeds using bitwise AND and bitshifts
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	// convert to float
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;

	return (res.f - 2.f) / 2.f;
}

static __device__ float3 operator*(float a, float3 b)
{
	b.x *= a;
	b.y *= a;
	b.z *= a;
	return b;
}

static __device__ float3 operator*(float3 a, float3 b)
{
	b.x *= a.x;
	b.y *= a.y;
	b.z *= a.z;
	return b;
}

static __device__ float3 operator-(float3 a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

static __device__ float3 operator+(float3 a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

static __device__ float3 operator/(float3 a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}


static __device__ int3 operator*(int3 a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

static __device__ int3 operator/(int3 a, int b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}

static __device__ int3 operator+(int3 a, int3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

static __device__ float3 floor(float3 a)
{
	return combine_float3(floor(a.x), floor(a.y), floor(a.z));
}