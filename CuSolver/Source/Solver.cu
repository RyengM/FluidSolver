#define SOLVER_EXPORTS

#include "../Include/Solver.h"
#include "../Include/Math.cuh"
#include <device_launch_parameters.h>
#include <curand.h>

#pragma comment(lib, "curand.lib")

#define MGPCG 1
#define MACCORMACK 1
#define REFLECT 0
#define VORT_CONFINE 1
#define IVOCK 0
#define BURST 0

/////////////////////////////
//                         //
//    y(j)|                //
//        |                //
//        | __ __ __ x(i)  //
//       /                 //
//      / z(k)             //
//                         //
/////////////////////////////

// u: velocity, v: vorticity, l: left, r: right, bo: bottom, t: top, bh: behind, f: front

// voxels out of bound will be assigned as border value
static __device__ float sample(float* field, int3 pos, int3 max_pos)
{
	pos = minmax(pos, max_pos);
	return field[pos.x + pos.y * max_pos.x + pos.z * max_pos.x * max_pos.y];
}

// voxels out of bound will be regarded as 0
static __device__ float cg_sample(float* field, int3 pos, int3 max_pos)
{
	if (pos.x < 0 || pos.x >= max_pos.x || pos.y < 0 || pos.y >= max_pos.y || pos.z < 0 || pos.z >= max_pos.z)
		return 0;
	return field[pos.x + pos.y * max_pos.x + pos.z * max_pos.x * max_pos.y];
}

// sum of six neighbor sample
static __device__ float neibor_sum(float* field, size_t i, size_t j, size_t k, int3 max_pos)
{
	return cg_sample(field, combine_int3(i - 1, j, k), max_pos) + cg_sample(field, combine_int3(i + 1, j, k), max_pos)
		+ cg_sample(field, combine_int3(i, j - 1, k), max_pos) + cg_sample(field, combine_int3(i, j + 1, k), max_pos)
		+ cg_sample(field, combine_int3(i, j, k - 1), max_pos) + cg_sample(field, combine_int3(i, j, k + 1), max_pos);
}

static __device__ float trilerp(float* field, float3 pos, float fpos_x, float fpos_y, float fpos_z, int3 max_pos, bool b_minmax_sampe)
{
	float x = pos.x - fpos_x;
	float y = pos.y - fpos_y;
	float z = pos.z - fpos_z;

	int ix = int(x);
	int iy = int(y);
	int iz = int(z);

	float fx = x - floor(x);
	float fy = y - floor(y);
	float fz = z - floor(z);

	if (x < -1e-6)
	{
		ix -= 1;
		fx = x - float(ix);
	}

	if (y < -1e-6)
	{
		iy -= 1;
		fy = y - float(iy);
	}

	if (z < -1e-6)
	{
		iz -= 1;
		fz = z - float(iz);
	}

	float a, b, c, d, e, f, g, h;

	if (b_minmax_sampe)
	{
		a = sample(field, combine_int3(ix, iy, iz), max_pos);
		b = sample(field, combine_int3(ix + 1, iy, iz), max_pos);
		c = sample(field, combine_int3(ix, iy + 1, iz), max_pos);
		d = sample(field, combine_int3(ix + 1, iy + 1, iz), max_pos);
		e = sample(field, combine_int3(ix, iy, iz + 1), max_pos);
		f = sample(field, combine_int3(ix + 1, iy, iz + 1), max_pos);
		g = sample(field, combine_int3(ix, iy + 1, iz + 1), max_pos);
		h = sample(field, combine_int3(ix + 1, iy + 1, iz + 1), max_pos);
	}
	else
	{
		a = cg_sample(field, combine_int3(ix, iy, iz), max_pos);
		b = cg_sample(field, combine_int3(ix + 1, iy, iz), max_pos);
		c = cg_sample(field, combine_int3(ix, iy + 1, iz), max_pos);
		d = cg_sample(field, combine_int3(ix + 1, iy + 1, iz), max_pos);
		e = cg_sample(field, combine_int3(ix, iy, iz + 1), max_pos);
		f = cg_sample(field, combine_int3(ix + 1, iy, iz + 1), max_pos);
		g = cg_sample(field, combine_int3(ix, iy + 1, iz + 1), max_pos);
		h = cg_sample(field, combine_int3(ix + 1, iy + 1, iz + 1), max_pos);
	}

	float lerp1 = lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
	float lerp2 = lerp(lerp(e, f, fx), lerp(g, h, fx), fy);

	return lerp(lerp1, lerp2, fz);
}

// Runge-Kutta
static __device__ float3 RK1(float* ux, float* uy, float* uz, float3 pos, float dt, int max_pos_x, int max_pos_y, int max_pos_z)
{
	float3 u;
	u.x = trilerp(ux, pos, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u.y = trilerp(uy, pos, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u.z = trilerp(uz, pos, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);
	return pos - dt * u;
}

static __device__ float3 RK2(float* ux, float* uy, float* uz, float3 pos, float dt, int max_pos_x, int max_pos_y, int max_pos_z)
{
	float3 u;
	u.x = trilerp(ux, pos, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u.y = trilerp(uy, pos, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u.z = trilerp(uz, pos, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);
	float3 mid = pos - 0.5f * dt * u;
	u.x = trilerp(ux, mid, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u.y = trilerp(uy, mid, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u.z = trilerp(uz, mid, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);

	return pos - dt * u;
}

static __device__ float3 RK4(float* ux, float* uy, float* uz, float3 pos, float dt, int max_pos_x, int max_pos_y, int max_pos_z)
{
	float3 u1;
	u1.x = trilerp(ux, pos, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u1.y = trilerp(uy, pos, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u1.z = trilerp(uz, pos, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);
	float3 p1 = pos - 0.5f * dt * u1;
	float3 u2;
	u2.x = trilerp(ux, p1, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u2.y = trilerp(uy, p1, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u2.z = trilerp(uz, p1, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);
	float3 p2 = pos - 0.5f * dt * u2;
	float3 u3;
	u3.x = trilerp(ux, p2, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u3.y = trilerp(uy, p2, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u3.z = trilerp(uz, p2, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);
	float3 p3 = pos - dt * u3;
	float3 u4;
	u4.x = trilerp(ux, p3, 0.f, 0.5f, 0.5f, combine_int3(max_pos_x + 1, max_pos_y, max_pos_z), false);
	u4.y = trilerp(uy, p3, 0.5f, 0.f, 0.5f, combine_int3(max_pos_x, max_pos_y + 1, max_pos_z), false);
	u4.z = trilerp(uz, p3, 0.5f, 0.5f, 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z + 1), false);
	return pos - dt * (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0;
}

static __device__ float3 MacCormack(float* ux, float* uy, float* uz, float3 pos, float dt, int max_pos_x, int max_pos_y, int max_pos_z)
{
	float3 coord_predict = RK4(ux, uy, uz, pos, dt, max_pos_x, max_pos_y, max_pos_z);
	float3 coord_reflect = RK4(ux, uy, uz, coord_predict, -dt, max_pos_x, max_pos_y, max_pos_z);

	return coord_predict + (pos - coord_reflect) / 2.f;
}

static __device__ float3 BFECC(float* ux, float* uy, float* uz, float3 pos, float dt, int max_pos_x, int max_pos_y, int max_pos_z)
{
	float3 coord_predict = RK4(ux, uy, uz, pos, dt, max_pos_x, max_pos_y, max_pos_z);
	float3 coord_reflect = RK4(ux, uy, uz, coord_predict, -dt, max_pos_x, max_pos_y, max_pos_z);

	float3 coord_mid = pos + (pos - coord_reflect) / 2.f;

	return RK4(ux, uy, uz, coord_mid, dt, max_pos_x, max_pos_y, max_pos_z);
}

void swap(float** a, float** b)
{
	float* temp = *a;
	*a = *b;
	*b = temp;
}

static __global__ void CopyFrom(float* dst, float* src)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	dst[ind] = src[ind];
}

static __global__ void Fill(float* field, float fill)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	field[ind] = fill;
}

// if there are more collocated values, can be reset in the same kernel
static __global__ void ResetGridCollocatedValue(float* rho, float* w_rho)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	rho[ind] = 0.f;
	w_rho[ind] = 0.f;
}

static __global__ void ResetGridStaggeredValue(float* field, float* w_field)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	field[ind] = 0.f;
	w_field[ind] = 0.f;
}

static __global__ void Mul(float* a, float* b, float* res)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	res[ind] = a[ind] * b[ind];
}

static __global__ void GlobalReduce(float* res)
{
	__shared__ float sdata[1024];
	size_t tid = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = tid + j * blockDim.x + k * blockDim.x * gridDim.x;

	sdata[tid] = res[ind];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	//// when threads <=32, there is only one wrap is working, no synchonization is required in a wrap
	//// there are still some optimization, maybe applied later
	//if (tid < 32)
	//{
	//	sdata[tid] += sdata[tid + 32]; sdata[tid] += sdata[tid + 16];
	//	sdata[tid] += sdata[tid + 8]; sdata[tid] += sdata[tid + 4];
	//	sdata[tid] += sdata[tid + 2]; sdata[tid] += sdata[tid + 1];
	//}

	// write result to global memory
	if (tid == 0)
		res[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];
}

static __global__ void BlockReduce(float* a)
{
	size_t ind = threadIdx.x;

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (ind < s)
			a[ind] += a[ind + s];
		__syncthreads();
	}
}

static __global__ void GridInitKernel(float* temperature, float temperature_env)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	temperature[ind] = temperature_env;
}

static __global__ void SourceKernel(float source_x, float source_y, float source_z, float radius, float* rho, float* temperature, float rho0, float temperature0)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	if (pow((float(i) - source_x), 2) + pow(float(j) - source_y, 2) + pow((float(k) - source_z), 2) <= radius * radius)
	{
		rho[ind] = rho0;
		temperature[ind] = temperature0;
	}
}

static __global__ void BurstSourceKernel(float* pos_x, float* pos_y, float* pos_z, int p_num, float radius, float* rho, float* temperature, float rho0, float temperature0, float temperature_env, float density_decay, float temperature_decay)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	for (int p = 0; p < p_num; p++)
	{
		if (pow((float(i) - pos_x[p]), 2) + pow(float(j) - pos_y[p], 2) + pow((float(k) - pos_z[p]), 2) <= radius * radius)
		{
			rho[ind] = rho0;
			temperature[ind] = temperature0;
		}
	}

	rho[ind] *= density_decay;
	temperature[ind] = (temperature[ind] - temperature_env) * temperature_decay + temperature_env;

	// trick, prevent umbrella stay too long
	if (j > gridDim.x - 20.f) rho[ind] *= 0.95f;
}

static __global__ void UpdateBurstSource(float* pos_x, float* pos_y, float* pos_z, float* vel_x, float* vel_y, float* vel_z, float vel_decay, float gravity, float dt)
{
	size_t i = threadIdx.x;

	vel_y[i] -= gravity * dt * 0.05f;
	vel_x[i] *= vel_decay;
	vel_y[i] *= vel_decay;
	vel_z[i] *= vel_decay;

	pos_x[i] += vel_x[i] * dt;
	pos_y[i] += vel_y[i] * dt;
	pos_z[i] += vel_z[i] * dt;
}

static __global__ void ForceKernelUy(float* ux, float* uy, float* uz, float* f_density, float* f_temperature, float dt, float temperature_env, float gravity, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float3 pos;
	// absolute position  center: 0, ux: 1, uy: 2, uz: 3
	pos.x = float(i) + 0.5f;
	pos.y = float(j);
	pos.z = float(k) + 0.5f;

	float temperature = trilerp(f_temperature, pos, 0.5f, 0.5f, 0.5f, max_pos, true);

	float buoyancy = (temperature - temperature_env) * dt;

	uy[ind] += buoyancy;

#if BURST
	uy[ind] -= gravity * 0.001f;
#endif
}

static __global__ void SemiLagKernel(float* field, float* new_field, float* ux, float* uy, float* uz, float dt, int max_pos_x, int max_pos_y, int max_pos_z, int dir)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float3 pos;
	// absolute position  center: 0, ux: 1, uy: 2, uz: 3, vortx: 4, vorty: 5, vortz: 6
	if (dir >= 0 && dir < 4)
	{
		pos.x = float(i) + (dir == 1 ? 0.f : 0.5f);
		pos.y = float(j) + (dir == 2 ? 0.f : 0.5f);
		pos.z = float(k) + (dir == 3 ? 0.f : 0.5f);
	}
	else
	{
		pos.x = float(i) + (dir == 4 ? 0.5f : 0.f);
		pos.y = float(j) + (dir == 5 ? 0.5f : 0.f);
		pos.z = float(k) + (dir == 6 ? 0.5f : 0.f);
	}

	// trace back
#if MACCORMACK
	float3 coord = MacCormack(ux, uy, uz, pos, dt, max_pos_x, max_pos_y, max_pos_z);
#else
	float3 coord = RK4(ux, uy, uz, pos, dt, max_pos_x, max_pos_y, max_pos_z);
#endif
	if (dir >= 0 && dir < 4)
		new_field[ind] = trilerp(field, coord, dir == 1 ? 0.f : 0.5f, dir == 2 ? 0.f : 0.5f, dir == 3 ? 0.f : 0.5f,
			combine_int3(dir == 1 ? (max_pos_x + 1) : max_pos_x, dir == 2 ? (max_pos_y + 1) : max_pos_y, dir == 3 ? (max_pos_z + 1) : max_pos_z), false);
	else
		new_field[ind] = trilerp(field, coord, dir == 4 ? 0.5f : 0.f, dir == 5 ? 0.5f : 0.f, dir == 6 ? 0.5f : 0.f, combine_int3(max_pos_x, max_pos_y, max_pos_z), false);
}

// ▽·u
static __global__ void DivergenceKernel(float* field, float* ux, float* uy, float* uz, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float ul = sample(ux, combine_int3(i, j, k), combine_int3(max_pos.x + 1, max_pos.y, max_pos.z));
	float ur = sample(ux, combine_int3(i + 1, j, k), combine_int3(max_pos.x + 1, max_pos.y, max_pos.z));
	float ubo = sample(uy, combine_int3(i, j, k), combine_int3(max_pos.x, max_pos.y + 1, max_pos.z));
	float ut = sample(uy, combine_int3(i, j + 1, k), combine_int3(max_pos.x, max_pos.y + 1, max_pos.z));
	float ubh = sample(uz, combine_int3(i, j, k), combine_int3(max_pos.x, max_pos.y, max_pos.z + 1));
	float uf = sample(uz, combine_int3(i, j, k + 1), combine_int3(max_pos.x, max_pos.y, max_pos.z + 1));

	float div = ur + uf + ut - ul - ubh - ubo;

	field[ind] = div;
}

// -6p + Σp_neibor = ▽·u
static __global__ void JacobiKernel(float* field, float* new_field, float* div_field, float* r, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pl = sample(field, combine_int3(i - 1, j, k), max_pos);
	float pr = sample(field, combine_int3(i + 1, j, k), max_pos);
	float pbo = sample(field, combine_int3(i, j - 1, k), max_pos);
	float pt = sample(field, combine_int3(i, j + 1, k), max_pos);
	float pbh = sample(field, combine_int3(i, j, k - 1), max_pos);
	float pf = sample(field, combine_int3(i, j, k + 1), max_pos);
	float div = sample(div_field, combine_int3(i, j, k), max_pos);

	new_field[ind] = (pl + pr + pbh + pf + pbo + pt - div) / 6.f;
	r[ind] = div + 6 * field[ind] - pl - pr - pbh - pf - pbo - pt;
}

// ▽·u = 0, apply pressure
static __global__ void ApplyGradientUx(float* ux, float* pressure_field, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pl = sample(pressure_field, combine_int3(i - 1, j, k), max_pos);
	float pr = sample(pressure_field, combine_int3(i, j, k), max_pos);

	ux[ind] -= pr - pl;
}

static __global__ void ApplyGradientUy(float* uy, float* pressure_field, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pbo = sample(pressure_field, combine_int3(i, j - 1, k), max_pos);
	float pt = sample(pressure_field, combine_int3(i, j, k), max_pos);

	uy[ind] -= pt - pbo;
}

static __global__ void ApplyGradientUz(float* uz, float* pressure_field, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pbh = sample(pressure_field, combine_int3(i, j, k - 1), max_pos);
	float pf = sample(pressure_field, combine_int3(i, j, k), max_pos);

	uz[ind] -= pf - pbh;
}

static __global__ void FindVortx(float* vortx, float* uy, float* uz, int3 max_pos_uy, int3 max_pos_uz)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float ubh = sample(uy, combine_int3(i, j, k - 1), max_pos_uy);
	float uf = sample(uy, combine_int3(i, j, k), max_pos_uy);
	float ubo = sample(uz, combine_int3(i, j - 1, k), max_pos_uz);
	float ut = sample(uz, combine_int3(i, j, k), max_pos_uz);
	vortx[ind] = uf - ubh - ut + ubo;
}

static __global__ void FindVorty(float* vorty, float* ux, float* uz, int3 max_pos_ux, int3 max_pos_uz)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float ul = sample(uz, combine_int3(i - 1, j, k), max_pos_uz);
	float ur = sample(uz, combine_int3(i, j, k), max_pos_uz);
	float ubh = sample(ux, combine_int3(i, j, k - 1), max_pos_ux);
	float uf = sample(ux, combine_int3(i, j, k), max_pos_ux);
	vorty[ind] = ur - ul - uf + ubh;
}

static __global__ void FindVortz(float* vortz, float* ux, float* uy, int3 max_pos_ux, int3 max_pos_uy)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float ul = sample(uy, combine_int3(i - 1, j, k), max_pos_uy);
	float ur = sample(uy, combine_int3(i, j, k), max_pos_uy);
	float ubo = sample(ux, combine_int3(i, j - 1, k), max_pos_ux);
	float ut = sample(ux, combine_int3(i, j, k), max_pos_ux);
	vortz[ind] = ut - ubo - ur + ul;
}

static __global__ void FindDeltaVort(float* advected_vortx, float* advected_vorty, float* advected_vortz, float* vortx, float* vorty, float* vortz)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// return delta vort
	advected_vortx[ind] -= vortx[ind];
	advected_vorty[ind] -= vorty[ind];
	advected_vortz[ind] -= vortz[ind];
}

static __global__ void ApplyVortConfinement(float* u, float* vortx, float* vorty, float* vortz, int3 max_pos, int dir, float coeff)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// vort does not cover the whole space
	if (i > 2 && i < blockDim.x - 2 && j > 2 && j < gridDim.x - 2 && k > 2 && k < gridDim.y - 2)
	{
		float3 pos;
		// absolute position ux: 1, uy: 2, uz: 3
		pos.x = float(i) + (dir == 1 ? 0.f : 0.5f);
		pos.y = float(j) + (dir == 2 ? 0.f : 0.5f);
		pos.z = float(k) + (dir == 3 ? 0.f : 0.5f);

		float vxt = trilerp(vortx, combine_float3(pos.x, pos.y + 0.5, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vyt = trilerp(vorty, combine_float3(pos.x, pos.y + 0.5, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzt = trilerp(vortz, combine_float3(pos.x, pos.y + 0.5, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float vt = sqrt(vxt * vxt + vyt * vyt + vzt * vzt);

		float vxbo = trilerp(vortx, combine_float3(pos.x, pos.y - 0.5, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vybo = trilerp(vorty, combine_float3(pos.x, pos.y - 0.5, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzbo = trilerp(vortz, combine_float3(pos.x, pos.y - 0.5, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float vbo = sqrt(vxbo * vxbo + vybo * vybo + vzbo * vzbo);

		float vxr = trilerp(vortx, combine_float3(pos.x + 0.5, pos.y, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vyr = trilerp(vorty, combine_float3(pos.x + 0.5, pos.y, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzr = trilerp(vortz, combine_float3(pos.x + 0.5, pos.y, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float vr = sqrt(vxr * vxr + vyr * vyr + vzr * vzr);

		float vxl = trilerp(vortx, combine_float3(pos.x - 0.5, pos.y, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vyl = trilerp(vorty, combine_float3(pos.x - 0.5, pos.y, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzl = trilerp(vortz, combine_float3(pos.x - 0.5, pos.y, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float vl = sqrt(vxl * vxl + vyl * vyl + vzl * vzl);

		float vxf = trilerp(vortx, combine_float3(pos.x, pos.y, pos.z + 0.5), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vyf = trilerp(vorty, combine_float3(pos.x, pos.y, pos.z + 0.5), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzf = trilerp(vortz, combine_float3(pos.x, pos.y, pos.z + 0.5), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float vf = sqrt(vxf * vxf + vyf * vyf + vzf * vzf);

		float vxbh = trilerp(vortx, combine_float3(pos.x, pos.y, pos.z - 0.5), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vybh = trilerp(vorty, combine_float3(pos.x, pos.y, pos.z - 0.5), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzbh = trilerp(vortz, combine_float3(pos.x, pos.y, pos.z - 0.5), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float vbh = sqrt(vxbh * vxbh + vybh * vybh + vzbh * vzbh);

		float vxc = trilerp(vortx, combine_float3(pos.x, pos.y, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float vyc = trilerp(vorty, combine_float3(pos.x, pos.y, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float vzc = trilerp(vortz, combine_float3(pos.x, pos.y, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);

		float nx = vr - vl;
		float ny = vt - vbo;
		float nz = vf - vbh;
		float n = sqrt(nx * nx + ny * ny + nz * nz + 1e-5);
		nx /= n;
		ny /= n;
		nz /= n;

		if (dir == 1)
			u[ind] += coeff * (ny * vzc - nz * vyc);
		else if (dir == 2)
			u[ind] += coeff * (nz * vxc - nx * vzc);
		else if (dir == 3)
			u[ind] += coeff * (nx * vyc - ny * vxc);
	}
}

static __global__ void ApplyVelocityxConfinement(float* ux, float* psiy, float* psiz, int3 max_pos, float scale)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// vort does not cover the whole space
	if (i > 2 && i < blockDim.x - 2 && j > 2 && j < gridDim.x - 2 && k > 2 && k < gridDim.y - 2)
	{
		float3 pos;
		pos.x = float(i);
		pos.y = float(j) + 0.5f;
		pos.z = float(k) + 0.5f;

		float psit = trilerp(psiz, combine_float3(pos.x, pos.y + 0.5f, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float psibo = trilerp(psiz, combine_float3(pos.x, pos.y - 0.5f, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float psif = trilerp(psiy, combine_float3(pos.x, pos.y, pos.z + 0.5f), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float psibh = trilerp(psiy, combine_float3(pos.x, pos.y, pos.z - 0.5f), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		ux[ind] += (psif - psibh - psit + psibo) * scale;
	}
}

static __global__ void ApplyVelocityyConfinement(float* uy, float* psix, float* psiz, int3 max_pos, float scale)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// vort does not cover the whole space
	if (i > 2 && i < blockDim.x - 2 && j > 2 && j < gridDim.x - 2 && k > 2 && k < gridDim.y - 2)
	{
		float3 pos;
		pos.x = float(i) + 0.5f;
		pos.y = float(j);
		pos.z = float(k) + 0.5f;

		float psif = trilerp(psix, combine_float3(pos.x, pos.y, pos.z + 0.5f), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float psibh = trilerp(psix, combine_float3(pos.x, pos.y, pos.z - 0.5f), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float psir = trilerp(psiz, combine_float3(pos.x + 0.5f, pos.y, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		float psil = trilerp(psiz, combine_float3(pos.x - 0.5f, pos.y, pos.z), 0.f, 0.f, 0.5f, combine_int3(max_pos.x + 1, max_pos.y + 1, max_pos.z), true);
		uy[ind] += (psir - psil - psif + psibh) * scale;
	}
}

static __global__ void ApplyVelocityzConfinement(float* uz, float* psix, float* psiy, int3 max_pos, float scale)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// vort does not cover the whole space
	if (i > 2 && i < blockDim.x - 2 && j > 2 && j < gridDim.x - 2 && k > 2 && k < gridDim.y - 2)
	{
		float3 pos;
		pos.x = float(i) + 0.5f;
		pos.y = float(j) + 0.5f;
		pos.z = float(k);

		float psir = trilerp(psiy, combine_float3(pos.x + 0.5f, pos.y, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float psil = trilerp(psiy, combine_float3(pos.x - 0.5f, pos.y, pos.z), 0.f, 0.5f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z + 1), true);
		float psit = trilerp(psix, combine_float3(pos.x, pos.y + 0.5f, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);
		float psibo = trilerp(psix, combine_float3(pos.x, pos.y - 0.5f, pos.z), 0.5f, 0.f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z + 1), true);

		uz[ind] += (psit - psibo - psir + psil) * scale;
	}
}

static __global__ void ApplyStretch(float* vortx, float* vorty, float* vortz, float* ux, float* uy, float* uz, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// x-axis
	float3 pos;
	pos.x = float(i) + 0.5f;
	pos.y = float(j);
	pos.z = float(k);

	float vx = vortx[ind];
	float vy = trilerp(vorty, pos, 0.f, 0.5f, 0.f, max_pos, true);
	float vz = trilerp(vortz, pos, 0.f, 0.f, 0.5f, max_pos, true);
	// ω·▽u = lim ε->0 (u(x + 0.5εω) - u(x - 0.5εω)) / ε
	float n = sqrt(vx * vx + vy * vy + vz * vz + 1e-5);
	vx /= n * 2.f;
	vy /= n * 2.f;
	vz /= n * 2.f;

	float dudvx = trilerp(ux, pos + combine_float3(vx, vy, vz), 0.5f, 0.f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z), true)
		- trilerp(ux, pos - combine_float3(vx, vy, vz), 0.5f, 0.f, 0.f, combine_int3(max_pos.x + 1, max_pos.y, max_pos.z), true);
	vortx[ind] += dudvx;

	// y-axis
	pos.x = float(i);
	pos.y = float(j) + 0.5f;

	vx = trilerp(vortx, pos, 0.5f, 0.f, 0.f, max_pos, true);
	vy = vorty[ind];
	// ω·▽u = lim ε->0 (u(x + 0.5εω) - u(x - 0.5εω)) / ε
	n = sqrt(vx * vx + vy * vy + vz * vz + 1e-5);
	vx /= n * 2.f;
	vy /= n * 2.f;
	vz /= n * 2.f;

	float dudvy = trilerp(uy, pos + combine_float3(vx, vy, vz), 0.f, 0.5f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z), true)
		- trilerp(uy, pos - combine_float3(vx, vy, vz), 0.f, 0.5f, 0.f, combine_int3(max_pos.x, max_pos.y + 1, max_pos.z), true);
	vorty[ind] += dudvy;

	// z-axis
	pos.y = float(j);
	pos.z = float(k) + 0.5f;

	vy = trilerp(vorty, pos, 0.f, 0.5f, 0.f, max_pos, true);
	vz = vorty[ind];
	// ω·▽u = lim ε->0 (u(x + 0.5εω) - u(x - 0.5εω)) / ε
	n = sqrt(vx * vx + vy * vy + vz * vz + 1e-5);
	vx /= n * 2.f;
	vy /= n * 2.f;
	vz /= n * 2.f;

	float dudvz = trilerp(uz, pos + combine_float3(vx, vy, vz), 0.f, 0.f, 0.5f, combine_int3(max_pos.x, max_pos.y, max_pos.z + 1), true)
		- trilerp(uz, pos - combine_float3(vx, vy, vz), 0.f, 0.f, 0.5f, combine_int3(max_pos.x, max_pos.y, max_pos.z + 1), true);
	vortz[ind] += dudvz;
}

// reflect u = 2 * projected u - origin u
static __global__ void VelocityReflect(float* r_u, float* u)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	r_u[ind] = 2 * r_u[ind] - u[ind];
}

// collocated attributes should be invoked once in order to reduce kernel invocation
static __global__ void CollocatedWeightDivision(float* rho, float* w_rho)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	if (w_rho[ind] > 0)
	{
		rho[ind] /= w_rho[ind];
	}
}

static __global__ void StaggeredWeightDivision(float* attr, float* w_attr)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	if (w_attr[ind] > 0)
	{
		attr[ind] /= w_attr[ind];
	}
}

// -6p + Σp_neibor = ▽·u
// Ap = -▽·u, Ap = 6p - Σp_neibor, b = -▽·u
// r = b - Ax, assume x = 0 in the beginning
static __global__ void InitConjugate(float* r, float* field, float* x)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	r[ind] = -field[ind];
	x[ind] = 0.f;
}

// p here is conjugate gradient, not pressure
static __global__ void ComputeAp(float* Ap, float* p, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pc = cg_sample(p, combine_int3(i, j, k), max_pos);

	Ap[ind] = 6.f * pc - neibor_sum(p, i, j, k, max_pos);
}

static __global__ void UpdateResidual(float* r, float* p, float* Ap, float* x, float alpha)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	x[ind] += alpha * p[ind];
	r[ind] -= alpha * Ap[ind];
}

static __global__ void UpdateP(float* p, float* z, float beta)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	p[ind] = z[ind] + beta * p[ind];
}

static __global__ void SubRestrict(float* r, float* z, int offset, int3 max_pos, int phase)
{
	size_t new_ind = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	size_t i = threadIdx.x * 2 + ((phase & 1) ? 1 : 0);
	size_t j = blockIdx.x * 2 + ((phase & 2) ? 1 : 0);
	size_t k = blockIdx.y * 2 + ((phase & 4) ? 1 : 0);
	size_t ind = i + j * max_pos.x + k * max_pos.x * max_pos.y;

	float res = r[offset + ind] - (6 * z[offset + ind] - neibor_sum(z + offset, i, j, k, max_pos));
	// r[l+1][pos//2] += res * 0.5
	r[offset + max_pos.x * max_pos.y * max_pos.z + new_ind] += res * 0.5;
}

static __global__ void Prolongate(float* z, int offset, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// r[l][pos] += r[l+1][pos//2]
	size_t new_ind = (i >> 1) + (j >> 1) * (max_pos.x >> 1) + (k >> 1) * (max_pos.x >> 1) * (max_pos.y >> 1);
	z[offset + ind] += z[offset + max_pos.x * max_pos.y * max_pos.z + new_ind];
}

static __global__ void Smooth(float* r, float* z, int offset, bool phase, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// red/black Gauss Seidel
	if (bool((i + j + k) & 1) == phase)
	{
		z[offset + ind] = (r[offset + ind] + neibor_sum(z + offset, i, j, k, max_pos)) / 6.f;
	}
}

void Solver::Restrict(int offset, int max_pos_x, int max_pos_y, int max_pos_z)
{
	int3 max_pos;
	max_pos.x = max_pos_x;
	max_pos.y = max_pos_y;
	max_pos.z = max_pos_z;
	// void write conflict
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 0);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 1);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 2);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 3);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 4);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 5);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 6);
	SubRestrict << <dim3(max_pos.y / 2, max_pos.z / 2), max_pos.x / 2 >> > (r, z, offset, max_pos, 7);
}

void Solver::InitCuda()
{
	checkCudaErrors(cudaMalloc((void**)&f_ux, (nx + 1) * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_uy, nx * (ny + 1) * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_uz, nx * ny * (nz + 1) * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_ux, (nx + 1) * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_uy, nx * (ny + 1) * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_uz, nx * ny * (nz + 1) * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_rho, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_rho, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_temperature, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_temperature, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_pressure, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_pressure, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_div, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_vortx, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_vorty, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_vortz, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_vortx, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_vorty, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_vortz, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_psix, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_psiy, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_psiz, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&r, mg_space * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&z, mg_space * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&Ap, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&x, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&temp, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_temp_res, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p_posx, particle_num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p_posy, particle_num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p_posz, particle_num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p_velx, particle_num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p_vely, particle_num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p_velz, particle_num * sizeof(float)));

	checkCudaErrors(cudaMemset(f_ux, 0, (nx + 1) * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_uy, 0, nx * (ny + 1) * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_uz, 0, nx * ny * (nz + 1) * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_ux, 0, (nx + 1) * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_uy, 0, nx * (ny + 1) * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_uz, 0, nx * ny * (nz + 1) * sizeof(float)));
	checkCudaErrors(cudaMemset(f_rho, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_rho, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_temperature, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_temperature, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_pressure, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_pressure, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_div, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_vortx, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_vorty, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_vortz, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_vortx, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_vorty, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_vortz, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_psix, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_psiy, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_psiz, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(r, 0, mg_space * sizeof(float)));
	checkCudaErrors(cudaMemset(z, 0, mg_space * sizeof(float)));
	checkCudaErrors(cudaMemset(p, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(Ap, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(x, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(temp, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(d_temp_res, 0, sizeof(float)));
	checkCudaErrors(cudaMemset(p_posx, 0, particle_num * sizeof(float)));
	checkCudaErrors(cudaMemset(p_posy, 0, particle_num * sizeof(float)));
	checkCudaErrors(cudaMemset(p_posz, 0, particle_num * sizeof(float)));
	checkCudaErrors(cudaMemset(p_velx, 0, particle_num * sizeof(float)));
	checkCudaErrors(cudaMemset(p_vely, 0, particle_num * sizeof(float)));
	checkCudaErrors(cudaMemset(p_velz, 0, particle_num * sizeof(float)));
}

void Solver::FreeCuda()
{
	checkCudaErrors(cudaFree(f_ux));
	checkCudaErrors(cudaFree(f_uy));
	checkCudaErrors(cudaFree(f_uz));
	checkCudaErrors(cudaFree(f_new_ux));
	checkCudaErrors(cudaFree(f_new_uy));
	checkCudaErrors(cudaFree(f_new_uz));
	checkCudaErrors(cudaFree(f_rho));
	checkCudaErrors(cudaFree(f_new_rho));
	checkCudaErrors(cudaFree(f_temperature));
	checkCudaErrors(cudaFree(f_new_temperature));
	checkCudaErrors(cudaFree(f_pressure));
	checkCudaErrors(cudaFree(f_new_pressure));
	checkCudaErrors(cudaFree(f_div));
	checkCudaErrors(cudaFree(f_vortx));
	checkCudaErrors(cudaFree(f_vorty));
	checkCudaErrors(cudaFree(f_vortz));
	checkCudaErrors(cudaFree(f_new_vortx));
	checkCudaErrors(cudaFree(f_new_vorty));
	checkCudaErrors(cudaFree(f_new_vortz));
	checkCudaErrors(cudaFree(f_psix));
	checkCudaErrors(cudaFree(f_psiy));
	checkCudaErrors(cudaFree(f_psiz));
	checkCudaErrors(cudaFree(r));
	checkCudaErrors(cudaFree(z));
	checkCudaErrors(cudaFree(p));
	checkCudaErrors(cudaFree(Ap));
	checkCudaErrors(cudaFree(x));
	checkCudaErrors(cudaFree(temp));
	checkCudaErrors(cudaFree(d_temp_res));
	checkCudaErrors(cudaFree(p_posx));
	checkCudaErrors(cudaFree(p_posy));
	checkCudaErrors(cudaFree(p_posz));
	checkCudaErrors(cudaFree(p_velx));
	checkCudaErrors(cudaFree(p_vely));
	checkCudaErrors(cudaFree(p_velz));
}

void Solver::InitParam()
{
	GridInitKernel << <dim3(ny, nz), nx >> > (f_temperature, temperature_env);

#if BURST
	curandGenerator_t gen;
	curandStatus_t status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	curandGenerateNormal(gen, p_posx, particle_num, burst_pos_x, init_normal_radius);
	curandGenerateNormal(gen, p_posy, particle_num, burst_pos_y, init_normal_radius);
	curandGenerateNormal(gen, p_posz, particle_num, burst_pos_z, init_normal_radius);
	curandGenerateNormal(gen, p_velx, particle_num, 0.f, init_normal_velocity);
	curandGenerateNormal(gen, p_vely, particle_num, 0.f, init_normal_velocity);
	curandGenerateNormal(gen, p_velz, particle_num, 0.f, init_normal_velocity);
#endif
}

void Solver::UpdateCuda()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	int3 max_pos_ux = max_pos;
	max_pos_ux.x = nx + 1;
	int3 max_pos_uy = max_pos;
	max_pos_uy.y = ny + 1;
	int3 max_pos_uz = max_pos;
	max_pos_uz.z = nz + 1;

	// add source
#if BURST
	BurstSourceKernel << <dim3(ny, nz), nx >> > (p_posx, p_posy, p_posz, particle_num, particle_radius, f_rho, f_temperature, rho, temperature, temperature_env, density_decay, temperature_decay);
	UpdateBurstSource << <dim3(1, 1), particle_num >> > (p_posx, p_posy, p_posz, p_velx, p_vely, p_velz, vel_decay, gravity, dt);
#else
	SourceKernel << <dim3(ny, nz), nx >> > (source_pos_x, source_pos_y, source_pos_z, source_radius, f_rho, f_temperature, rho, temperature);
#endif

	ForceKernelUy << <dim3(ny + 1, nz), nx >> > (f_ux, f_uy, f_uz, f_rho, f_temperature, dt, temperature_env, gravity, max_pos);

#if VORT_CONFINE || IVOCK
	// u -> ω
	FindVortx << <dim3(ny, nz), nx >> > (f_vortx, f_uy, f_uz, max_pos_uy, max_pos_uz);
	FindVorty << <dim3(ny, nz), nx >> > (f_vorty, f_ux, f_uz, max_pos_ux, max_pos_uz);
	FindVortz << <dim3(ny, nz), nx >> > (f_vortz, f_ux, f_uy, max_pos_ux, max_pos_uy);
#if IVOCK
	// ω -> ω~, stretch
	ApplyStretch << <dim3(ny, nz), nx >> > (f_vortx, f_vorty, f_vortz, f_ux, f_uy, f_uz, max_pos);
	// ω~ -> ω~, advect
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_vortx, f_new_vortx, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 4);
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_vorty, f_new_vorty, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 5);
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_vortz, f_new_vortz, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 6);
#endif
#endif
	
#if REFLECT		
	// u -> u1/2*
	SemiLagKernel << <dim3(ny, nz), nx + 1 >> > (f_ux, f_new_ux, f_ux, f_uy, f_uz, dt / 2.f, max_pos.x, max_pos.y, max_pos.z, 1);
	SemiLagKernel << <dim3(ny + 1, nz), nx >> > (f_uy, f_new_uy, f_ux, f_uy, f_uz, dt / 2.f, max_pos.x, max_pos.y, max_pos.z, 2);
	SemiLagKernel << <dim3(ny, nz + 1), nx >> > (f_uz, f_new_uz, f_ux, f_uy, f_uz, dt / 2.f, max_pos.x, max_pos.y, max_pos.z, 3);
	// record u1/2*
	CopyFrom << <dim3(ny, nz), nx + 1 >> > (f_ux, f_new_ux);
	CopyFrom << <dim3(ny + 1, nz), nx >> > (f_uy, f_new_uy);
	CopyFrom << <dim3(ny, nz + 1), nx >> > (f_uz, f_new_uz);
#else
	// u -> u*
	Advect();
#endif

#if IVOCK
	// u* -> ω*
	FindVortx << <dim3(ny, nz), nx >> > (f_vortx, f_uy, f_uz, max_pos_uy, max_pos_uz);
	FindVorty << <dim3(ny, nz), nx >> > (f_vorty, f_ux, f_uz, max_pos_ux, max_pos_uz);
	FindVortz << <dim3(ny, nz), nx >> > (f_vortz, f_ux, f_uy, max_pos_ux, max_pos_uy);
	// δω = ω~ - ω*, return f_new_vort as delta vort
	FindDeltaVort << <dim3(ny, nz), nx >> > (f_new_vortx, f_new_vorty, f_new_vortz, f_vortx, f_vorty, f_vortz);
	// δω -> ψ
	PsiVCycle();
	// ψ -> δu
	ApplyVelocityxConfinement << <dim3(ny, nz), nx + 1 >> > (f_ux, f_psiy, f_psiz, max_pos, ivock_scale);
	ApplyVelocityyConfinement << <dim3(ny + 1, nz), nx >> > (f_uy, f_psix, f_psiz, max_pos, ivock_scale);
	ApplyVelocityzConfinement << <dim3(ny, nz + 1), nx >> > (f_uz, f_psix, f_psiy, max_pos, ivock_scale);
#endif

#if VORT_CONFINE
	ApplyVortConfinement << <dim3(ny, nz), nx + 1 >> > (f_ux, f_vortx, f_vorty, f_vortz, max_pos, 1, curl_strength);
	ApplyVortConfinement << <dim3(ny + 1, nz), nx >> > (f_uy, f_vortx, f_vorty, f_vortz, max_pos, 2, curl_strength);
	ApplyVortConfinement << <dim3(ny, nz + 1), nx >> > (f_uz, f_vortx, f_vorty, f_vortz, max_pos, 3, curl_strength);
#endif
	Project();

#if REFLECT
	// update velocity
	ApplyGradientUx << <dim3(ny, nz), nx + 1 >> > (f_ux, f_pressure, max_pos);
	ApplyGradientUy << <dim3(ny + 1, nz), nx >> > (f_uy, f_pressure, max_pos);
	ApplyGradientUz << <dim3(ny, nz + 1), nx >> > (f_uz, f_pressure, max_pos);
	// velocity reflect, u1/2^ = 2u1/2 - u1/2*
	VelocityReflect << <dim3(ny, nz), nx + 1 >> > (f_ux, f_new_ux);
	VelocityReflect << <dim3(ny + 1, nz), nx >> > (f_uy, f_new_uy);
	VelocityReflect << <dim3(ny, nz + 1), nx >> > (f_uz, f_new_uz);

	// u1/2^ -> u1
	SemiLagKernel << <dim3(ny, nz), nx + 1 >> > (f_ux, f_new_ux, f_ux, f_uy, f_uz, dt / 2.f, max_pos.x, max_pos.y, max_pos.z, 1);
	SemiLagKernel << <dim3(ny + 1, nz), nx >> > (f_uy, f_new_uy, f_ux, f_uy, f_uz, dt / 2.f, max_pos.x, max_pos.y, max_pos.z, 2);
	SemiLagKernel << <dim3(ny, nz + 1), nx >> > (f_uz, f_new_uz, f_ux, f_uy, f_uz, dt / 2.f, max_pos.x, max_pos.y, max_pos.z, 3);
	swap(&f_ux, &f_new_ux);
	swap(&f_uy, &f_new_uy);
	swap(&f_uz, &f_new_uz);
	// temperature advection
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_temperature, f_new_temperature, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 0);
	swap(&f_temperature, &f_new_temperature);
	// density advection
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_rho, f_new_rho, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 0);
	swap(&f_rho, &f_new_rho);
	Project();
#endif

	// update velocity
	ApplyGradientUx << <dim3(ny, nz), nx + 1 >> > (f_ux, f_pressure, max_pos);
	ApplyGradientUy << <dim3(ny + 1, nz), nx >> > (f_uy, f_pressure, max_pos);
	ApplyGradientUz << <dim3(ny, nz + 1), nx >> > (f_uz, f_pressure, max_pos);
}

void Solver::Initialize()
{
	f_density = (float*)malloc(nx * ny * nz * sizeof(float));

	// note that the layout of our multi level grid is [level][z][y][x]
	int temp_space = nx * ny * nz;
	for (int i = 0; i < mg_level; ++i)
	{
		mg_space += temp_space;
		temp_space /= 8;
	}

	InitCuda();
	InitParam();
}

void Solver::Update()
{
	printf("frame: %d\n", current_frame);
	UpdateCuda();
	current_frame++;
}

Solver::~Solver()
{
	free(f_density);
	FreeCuda();
}

void Solver::Advect()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	// velocity advection
	SemiLagKernel << <dim3(ny, nz), nx + 1 >> > (f_ux, f_new_ux, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 1);
	SemiLagKernel << <dim3(ny + 1, nz), nx >> > (f_uy, f_new_uy, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 2);
	SemiLagKernel << <dim3(ny, nz + 1), nx >> > (f_uz, f_new_uz, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 3);
	swap(&f_ux, &f_new_ux);
	swap(&f_uy, &f_new_uy);
	swap(&f_uz, &f_new_uz);
	// temperature advection
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_temperature, f_new_temperature, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 0);
	swap(&f_temperature, &f_new_temperature);
	// density advection
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_rho, f_new_rho, f_ux, f_uy, f_uz, dt, max_pos.x, max_pos.y, max_pos.z, 0);
	swap(&f_rho, &f_new_rho);
}

// calc pressure
void Solver::Project()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	// divergence
	DivergenceKernel << <dim3(ny, nz), nx >> > (f_div, f_ux, f_uy, f_uz, max_pos);

#if 0
	// jacobi iteration
	for (int i = 0; i < max_iter; ++i)
	{
		JacobiKernel << <dim3(ny, nz), nx >> > (f_pressure, f_new_pressure, f_div, r, max_pos);
		swap(&f_pressure, &f_new_pressure);

		Mul << <dim3(ny, nz), nx >> > (r, r, temp);
		Reduce();
		checkCudaErrors(cudaMemcpy(&init_rTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "iter " << i << " residual: " << init_rTr << std::endl;
	}
#else
	Conjugate(f_pressure, f_div);
#endif
}

void Solver::Scatter()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

}

void Solver::ResetGridValue()
{
	ResetGridCollocatedValue << <dim3(ny, nz), nx >> > (f_rho, f_new_rho);
	ResetGridStaggeredValue << <dim3(ny, nz), nx + 1 >> > (f_ux, f_new_ux);
	ResetGridStaggeredValue << <dim3(ny + 1, nz), nx >> > (f_uy, f_new_uy);
	ResetGridStaggeredValue << <dim3(ny, nz + 1), nx >> > (f_uz, f_new_uz);
}

void Solver::Reduce()
{
	int resolution = nx * ny * nz;
	int blockResolution = resolution / 1024;
	int dimResolution = sqrt(blockResolution);
	GlobalReduce << <dim3(dimResolution, dimResolution), 1024 >> > (temp);
	if (blockResolution > 1024)
	{
		blockResolution /= 1024;
		GlobalReduce << <dim3(blockResolution, 1), 1024 >> > (temp);
	}
	BlockReduce << <1, blockResolution >> > (temp);
}

void Solver::Conjugate(float* res, float* field)
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	InitConjugate << <dim3(ny, nz), nx >> > (r, field, x);

	// aTb operator, calc the sum of each block and then reduce all the data
	// note that the number of thread in each block cannot exceed 1024
	Mul << <dim3(ny, nz), nx >> > (r, r, temp);
	Reduce();
	checkCudaErrors(cudaMemcpy(&init_rTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "init rTr: " << init_rTr << std::endl;

#if MGPCG
	MG_Preconditioner();
#else
	CopyFrom << <dim3(ny, nz), nx >> > (z, r);
#endif

	// p(0) = M^-1 r(0)
	UpdateP << <dim3(ny, nz), nx >> > (p, z, 0);

	Mul << <dim3(ny, nz), nx >> > (z, r, temp);
	Reduce();
	checkCudaErrors(cudaMemcpy(&old_zTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < max_iter; ++i)
	{
		// α(k) = r(k)Tr(k) / p(k)TAp(k)
		ComputeAp << <dim3(ny, nz), nx >> > (Ap, p, max_pos);
		Mul << <dim3(ny, nz), nx >> > (p, Ap, temp);
		Reduce();
		checkCudaErrors(cudaMemcpy(&pAp, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		float alpha = old_zTr / pAp;

		// x(k+1) = x(k) + α(k)p(k), r(k+1) = r(k) - α(k)Ap(k)
		UpdateResidual << <dim3(ny, nz), nx >> > (r, p, Ap, x, alpha);

		// if ||r(k+1)|| is sufficient enough small, break
		Mul << <dim3(ny, nz), nx >> > (r, r, temp);
		Reduce();
		checkCudaErrors(cudaMemcpy(&rTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "iter " << i << " rTr: " << rTr << std::endl;

		// early stop
		if (rTr < iter_precision || rTr == 0)
			break;

#if MGPCG
		MG_Preconditioner();
#else
		CopyFrom << <dim3(ny, nz), nx >> > (z, r);
#endif

		// β(k) = r(k+1)Tr(k+1)/r(k)Tr(k)
		Mul << <dim3(ny, nz), nx >> > (z, r, temp);
		Reduce();
		checkCudaErrors(cudaMemcpy(&new_zTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		float beta = new_zTr / old_zTr;
		// p(k+1) = r(k+1) + β(k)p(k)
		UpdateP << <dim3(ny, nz), nx >> > (p, z, beta);

		old_zTr = new_zTr;
		last_rTr = rTr;
	}
	swap(&res, &x);
	//CopyFrom << <dim3(ny, nz), nx >> > (res, x);
}

void Solver::PsiVCycle()
{
	InitConjugate << <dim3(ny, nz), nx >> > (r, f_new_vortx, x);
	MG_Preconditioner();
	CopyFrom << <dim3(ny, nz), nx >> > (f_psix, z);
	InitConjugate << <dim3(ny, nz), nx >> > (r, f_new_vorty, x);
	MG_Preconditioner();
	CopyFrom << <dim3(ny, nz), nx >> > (f_psiy, z);
	InitConjugate << <dim3(ny, nz), nx >> > (r, f_new_vortz, x);
	MG_Preconditioner();
	CopyFrom << <dim3(ny, nz), nx >> > (f_psiz, z);
}

void Solver::MG_Preconditioner()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	size_t r_offset = nx * ny * nz;
	size_t offset = 0;

	// initialize z[l] and r[l] with 0 except r[0]
	int blockSize = 1024;
	int gridFullSize = mg_space / blockSize;
	int gridResSize = (mg_space - r_offset) / blockSize;
	Fill << < gridFullSize, 1024 >> > (z, 0);
	Fill << < gridResSize, 1024 >> > (r + r_offset, 0);

	// downsample
	for (int l = 0; l < mg_level - 1; ++l)
	{
		for (int i = 0; i < init_smooth_steps << l; ++i)
		{
			Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 0, max_pos);
			Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 1, max_pos);
		}
		Restrict(offset, max_pos.x, max_pos.y, max_pos.z);

		offset += max_pos.x * max_pos.y * max_pos.z;
		max_pos.x = max_pos.x >> 1;
		max_pos.y = max_pos.y >> 1;
		max_pos.z = max_pos.z >> 1;
	}

	// bottom smoothing
	for (int i = 0; i < bottom_smooth_steps; ++i)
	{
		Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 0, max_pos);
		Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 1, max_pos);
	}
	// upsample
	for (int l = mg_level - 2; l >= 0; --l)
	{
		max_pos.x = max_pos.x << 1;
		max_pos.y = max_pos.y << 1;
		max_pos.z = max_pos.z << 1;
		offset -= max_pos.x * max_pos.y * max_pos.z;
		Prolongate << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (z, offset, max_pos);

		for (int i = 0; i < init_smooth_steps << l; ++i)
		{
			Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 0, max_pos);
			Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 1, max_pos);
		}
	}
}

float* Solver::GetDensityField()
{
	checkCudaErrors(cudaMemcpy(f_density, f_rho, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
	return f_density;
}