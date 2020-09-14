#define SOLVER_EXPORTS

#include "Solver.h"
#include "CudaUnitility.h"
#include <device_launch_parameters.h>

#define MGPCG 1

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

static __device__ float sample(float* field, int3 pos, int3 max_pos)
{
	pos = minmax(pos, max_pos);
	return field[pos.x + pos.y * max_pos.x + pos.z * max_pos.x * max_pos.y];
}

static __device__ float neibor_sum(float* field, int pos, int offset, int3 max_pos)
{
	int i = pos % max_pos.x;
	int j = pos % (max_pos.x * max_pos.y) / max_pos.x;
	int k = pos / (max_pos.x * max_pos.y);
	return i > 0 ? field[offset + pos - 1] : 0 + i < max_pos.x - 1 ? field[offset + pos + 1] : 0 +
		j > 0 ? field[offset + pos - max_pos.x] : 0 + j < max_pos.y - 1 ? field[offset + pos + max_pos.x] : 0 +
		k > 0 ? field[offset + pos - max_pos.x * max_pos.y] : 0 + k < max_pos.z - 1 ? field[offset + pos + max_pos.x * max_pos.y] : 0;
}

static __device__ float cg_sample(float* field, int3 pos, int3 max_pos)
{
	if (pos.x < 0 || pos.x >= max_pos.x || pos.y < 0 || pos.y >= max_pos.y || pos.z < 0 || pos.z >= max_pos.z)
		return 0;
	return field[pos.x + pos.y * max_pos.x + pos.z * max_pos.x * max_pos.y];
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

static __device__ float trilerp(float* field, float3 pos, int3 max_pos)
{
	float x = pos.x;
	float y = pos.y;
	float z = pos.z;

	int ix = int(x);
	int iy = int(y);
	int iz = int(z);

	float fx = x - floor(x);
	float fy = y - floor(y);
	float fz = z - floor(z);

	float a = sample(field, combine_int3(ix, iy, iz), max_pos);
	float b = sample(field, combine_int3(ix + 1, iy, iz), max_pos);
	float c = sample(field, combine_int3(ix, iy + 1, iz), max_pos);
	float d = sample(field, combine_int3(ix + 1, iy + 1, iz), max_pos);
	float e = sample(field, combine_int3(ix, iy, iz + 1), max_pos);
	float f = sample(field, combine_int3(ix + 1, iy, iz + 1), max_pos);
	float g = sample(field, combine_int3(ix, iy + 1, iz + 1), max_pos);
	float h = sample(field, combine_int3(ix + 1, iy + 1, iz + 1), max_pos);

	float lerp1 = lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
	float lerp2 = lerp(lerp(e, f, fx), lerp(g, h, fx), fy);

	return lerp(lerp1, lerp2, fz);
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

static __device__ float3 RK2(float* ux, float* uy, float* uz, float3 pos, float dt, int3 max_pos)
{
	float3 u;
	u.x = trilerp(ux, pos, max_pos);
	u.y = trilerp(uy, pos, max_pos);
	u.z = trilerp(uz, pos, max_pos);
	float3 mid = pos - 0.5f * dt * u;
	u.x = trilerp(ux, mid, max_pos);
	u.y = trilerp(uy, mid, max_pos);
	u.z = trilerp(uz, mid, max_pos);
	// here may exist out of range problem
	return pos - dt * u;
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

static __global__ void Fill(float* field, int offset, float fill)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	field[offset + ind] = fill;
}

static __global__ void GlobalReduce(float* a, float* b, float* res)
{
	 __shared__ float sdata[1024];
	size_t tid = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = tid + j * blockDim.x + k * blockDim.x * gridDim.x;

	sdata[tid] = a[ind] * b[ind];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// when threads <=32, there is only one wrap is working, no synchonization is required in a wrap
	// there are still some optimization, maybe applied later
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32]; sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8]; sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2]; sdata[tid] += sdata[tid + 1];
	}

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

static __global__ void SourceKernel(float* rho, float* ux, float* uy, float* uz, float rho0, float3 u0)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	if (i > blockDim.x / 2 - 2 && i < blockDim.x / 2 + 2 && j > gridDim.x / 2 - 2 && j < gridDim.x / 2 + 2 && k > 1 && k < 3)
	{
		rho[ind] = rho0;
		ux[ind] = u0.x;
		uy[ind] = u0.y;
		uz[ind] = u0.z;
	}
}

static __global__ void SemiLagKernel(float* field, float* new_field, float* ux, float* uy, float* uz, float dt, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float3 pos;
	pos.x = float(i);
	pos.y = float(j);
	pos.z = float(k);

	float3 coord = RK2(ux, uy, uz, pos, dt, max_pos);
	new_field[ind] = trilerp(field, coord, max_pos);
}

static __global__ void DivergenceKernel(float* field, float* ux, float* uy, float* uz, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float ul = sample(ux, combine_int3(i - 1, j, k), max_pos);
	float ur = sample(ux, combine_int3(i + 1, j, k), max_pos);
	float ubh = sample(uy, combine_int3(i, j - 1, k), max_pos);
	float uf = sample(uy, combine_int3(i, j + 1, k), max_pos);
	float ubo = sample(uz, combine_int3(i, j, k - 1), max_pos);
	float ut = sample(uz, combine_int3(i, j, k + 1), max_pos);

#if 0
	// box boundary
	float ucx = sample(ux, combine_int3(i, j, k), max_pos);
	float ucy = sample(uy, combine_int3(i, j, k), max_pos);
	float ucz = sample(uz, combine_int3(i, j, k), max_pos);
	if (i == 0)
		ul = -ucx;
	if (i == max_pos.x - 1)
		ur = -ucx;
	if (j == 0)
		ubh = -ucy;
	if (j == max_pos.y - 1)
		uf = -ucy;
	if (k == 0)
		ubo = -ucz;
	if (k == max_pos.z - 1)
		ut = -ucz;
#endif

	float div = (ur + uf + ut - ul - ubh - ubo) * 0.5;

	field[ind] = div;
}

static __global__ void JacobiKernel(float* field, float* new_field, float* div_field, float* r, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pl = sample(field, combine_int3(i - 1, j, k), max_pos);
	float pr = sample(field, combine_int3(i + 1, j, k), max_pos);
	float pbh = sample(field, combine_int3(i, j - 1, k), max_pos);
	float pf = sample(field, combine_int3(i, j + 1, k), max_pos);
	float pbo = sample(field, combine_int3(i, j, k - 1), max_pos);
	float pt = sample(field, combine_int3(i, j, k + 1), max_pos);
	float div = sample(div_field, combine_int3(i, j, k), max_pos);

	new_field[ind] = (pl + pr + pbh + pf + pbo + pt - div) / 6.f;
	r[ind] = div + 6 * field[ind] - pl - pr - pbh - pf - pbo - pt;
}

static __global__ void ApplyGradient(float* f_ux, float* f_uy, float* f_uz, float* pressure_field, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pl = sample(pressure_field, combine_int3(i - 1, j, k), max_pos);
	float pr = sample(pressure_field, combine_int3(i + 1, j, k), max_pos);
	float pbh = sample(pressure_field, combine_int3(i, j - 1, k), max_pos);
	float pf = sample(pressure_field, combine_int3(i, j + 1, k), max_pos);
	float pbo = sample(pressure_field, combine_int3(i, j, k - 1), max_pos);
	float pt = sample(pressure_field, combine_int3(i, j, k + 1), max_pos);

	f_ux[ind] -= 0.5 * (pr - pl);
	f_uy[ind] -= 0.5 * (pf - pbh);
	f_uz[ind] -= 0.5 * (pt - pbo);
}

static __global__ void VorticityKernel(float* f_vortx, float* f_vorty, float* f_vortz, float* ux, float* uy, float* uz, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float ul = sample(ux, combine_int3(i - 1, j, k), max_pos);
	float ur = sample(ux, combine_int3(i + 1, j, k), max_pos);
	float ubh = sample(uy, combine_int3(i, j - 1, k), max_pos);
	float uf = sample(uy, combine_int3(i, j + 1, k), max_pos);
	float ubo = sample(uz, combine_int3(i, j, k - 1), max_pos);
	float ut = sample(uz, combine_int3(i, j, k + 1), max_pos);

	f_vortx[ind] = (uf - ubh - ut + ubo) * 0.5;
	f_vorty[ind] = (ut - ubo - ur + ul) * 0.5;
	f_vortz[ind] = (ur - ul - uf + ubh) * 0.5;
}

static __global__ void ForceKernel(float* f_ux, float* f_uy, float* f_uz, float* f_vortx, float* f_vorty, float* f_vortz, float dt, float curl_strength, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float3 vl = combine_float3(sample(f_vortx, combine_int3(i - 1, j, k), max_pos),
		sample(f_vorty, combine_int3(i - 1, j, k), max_pos),
		sample(f_vortz, combine_int3(i - 1, j, k), max_pos));
	float3 vr = combine_float3(sample(f_vortx, combine_int3(i + 1, j, k), max_pos),
		sample(f_vorty, combine_int3(i + 1, j, k), max_pos),
		sample(f_vortz, combine_int3(i + 1, j, k), max_pos));
	float3 vbh = combine_float3(sample(f_vortx, combine_int3(i, j - 1, k), max_pos),
		sample(f_vorty, combine_int3(i, j - 1, k), max_pos),
		sample(f_vortz, combine_int3(i, j - 1, k), max_pos));
	float3 vf = combine_float3(sample(f_vortx, combine_int3(i, j + 1, k), max_pos),
		sample(f_vorty, combine_int3(i, j + 1, k), max_pos),
		sample(f_vortz, combine_int3(i, j + 1, k), max_pos));
	float3 vbo = combine_float3(sample(f_vortx, combine_int3(i, j, k - 1), max_pos),
		sample(f_vorty, combine_int3(i, j, k - 1), max_pos),
		sample(f_vortz, combine_int3(i, j, k - 1), max_pos));
	float3 vt = combine_float3(sample(f_vortx, combine_int3(i, j, k + 1), max_pos),
		sample(f_vorty, combine_int3(i, j, k + 1), max_pos),
		sample(f_vortz, combine_int3(i, j, k + 1), max_pos));
	float3 vc = combine_float3(sample(f_vortx, combine_int3(i, j, k), max_pos),
		sample(f_vorty, combine_int3(i, j, k), max_pos),
		sample(f_vortz, combine_int3(i, j, k), max_pos));

	// �� = ��|��|, N = ��/|��|
	float3 force = normalize(combine_float3(abs(length(vr)) - abs(length(vl)), abs(length(vf)) - abs(length(vbh)), abs(length(vt)) - abs(length(vbo))));
	// f_conf(vort) = ��h(N����)
	float3 fvort = curl_strength * cross(force, vc);

	f_ux[ind] += fvort.x * dt;
	f_uy[ind] += fvort.y * dt;
	f_uz[ind] += fvort.z * dt;
}

// -Ax = -b, r0 = -b = -����u
static __global__ void InitConjugate(float* r, float* f_div, float* x)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	r[ind] = -f_div[ind];
	x[ind] = 0;
}

// p here is conjugate gradient, not pressure
static __global__ void ComputeAp(float* Ap, float* p, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float pc = cg_sample(p, combine_int3(i, j, k), max_pos);

	Ap[ind] = 6.f * pc - neibor_sum(p, ind, 0, max_pos);
	//printf("ap %f %f\n", pc, neibor_sum(p, ind, 0, max_pos));
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

static __global__ void Restrict(float* r, float* z, int offset, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	float res = r[offset + ind] - (6 * z[offset + ind] - neibor_sum(z, ind, offset, max_pos));
	// r[l+1][pos//2] += res * 0.5
	offset += max_pos.x * max_pos.y * max_pos.z;
	int new_ind = i >> 2 + (j >> 2) * (max_pos.x >> 2) + (k >> 2) * (max_pos.x >> 2) * (max_pos.y >> 2);
	r[offset + new_ind] += res * 0.5;
}

static __global__ void Prolongate(float* z, int offset, int3 max_pos)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t k = blockIdx.y;
	size_t ind = i + j * blockDim.x + k * blockDim.x * gridDim.x;

	// r[l][pos] = r[l+1][pos//2]
	int new_ind = i >> 2 + (j >> 2) * (max_pos.x >> 2) + (k >> 2) * (max_pos.x >> 2) * (max_pos.y >> 2);
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
		z[offset + ind] = (r[offset + ind] + neibor_sum(z, ind, offset, max_pos)) / 6.f;
	}
}

void Solver::InitCuda()
{
	checkCudaErrors(cudaMalloc((void**)&f_ux, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_uy, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_uz, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_ux, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_uy, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_uz, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_rho, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_rho, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_pressure, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_new_pressure, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_div, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_vortx, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_vorty, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&f_vortz, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&r, mg_space * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&z, mg_space * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&new_z, mg_space * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&p, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&Ap, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&x, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&temp, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_temp_res, sizeof(float)));

	checkCudaErrors(cudaMemset(f_ux, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_uy, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_uz, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_ux, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_uy, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_uz, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_rho, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_rho, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_pressure, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_new_pressure, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_div, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_vortx, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_vorty, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(f_vortz, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(r, 0, mg_space * sizeof(float)));
	checkCudaErrors(cudaMemset(z, 0, mg_space * sizeof(float)));
	checkCudaErrors(cudaMemset(new_z, 0, mg_space * sizeof(float)));
	checkCudaErrors(cudaMemset(p, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(Ap, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(x, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(temp, 0, nx * ny * nz * sizeof(float)));
	checkCudaErrors(cudaMemset(d_temp_res, 0, sizeof(float)));
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
	checkCudaErrors(cudaFree(f_pressure));
	checkCudaErrors(cudaFree(f_new_pressure));
	checkCudaErrors(cudaFree(f_div));
	checkCudaErrors(cudaFree(f_vortx));
	checkCudaErrors(cudaFree(f_vorty));
	checkCudaErrors(cudaFree(f_vortz));
	checkCudaErrors(cudaFree(r));
	checkCudaErrors(cudaFree(z));
	checkCudaErrors(cudaFree(new_z));
	checkCudaErrors(cudaFree(p));
	checkCudaErrors(cudaFree(Ap));
	checkCudaErrors(cudaFree(x));
	checkCudaErrors(cudaFree(temp));
	checkCudaErrors(cudaFree(d_temp_res));
}

void Solver::UpdateCuda()
{
	float3 u;
	u.x = vel_x;
	u.y = vel_y;
	u.z = vel_z;

	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	// add source
	SourceKernel << <dim3(ny, nz), nx >> > (f_rho, f_ux, f_uy, f_uz, rho, u);
	// add force
	VorticityKernel << <dim3(ny, nz), nx >> > (f_vortx, f_vorty, f_vortz, f_ux, f_uy, f_uz, max_pos);
	ForceKernel << <dim3(ny, nz), nx >> > (f_ux, f_uy, f_uz, f_vortx, f_vorty, f_vortz, dt, curl_strength, max_pos);
	// velocity advection
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_ux, f_new_ux, f_ux, f_uy, f_uz, dt, max_pos);
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_uy, f_new_uy, f_ux, f_uy, f_uz, dt, max_pos);
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_uz, f_new_uz, f_ux, f_uy, f_uz, dt, max_pos);
	swap(&f_ux, &f_new_ux);
	swap(&f_uy, &f_new_uy);
	swap(&f_uz, &f_new_uz);
	// density advection
	SemiLagKernel << <dim3(ny, nz), nx >> > (f_rho, f_new_rho, f_ux, f_uy, f_uz, dt, max_pos);
	swap(&f_rho, &f_new_rho);
	// divergence
	DivergenceKernel << <dim3(ny, nz), nx >> > (f_div, f_ux, f_uy, f_uz, max_pos);
#if 0
	// jacobi iteration
	for (int i = 0; i < max_iter; ++i)
	{
		JacobiKernel << <dim3(ny, nz), nx >> > (f_pressure, f_new_pressure, f_div, r, max_pos);
		swap(&f_pressure, &f_new_pressure);
		//aTb << <1, 1 >> > (r, r,d_temp_res, max_pos);
		//checkCudaErrors(cudaMemcpy(&rTr, d_temp_res, sizeof(float), cudaMemcpyDeviceToHost));
		//std::cout << "iter " << i << " rTr: " << rTr << std::endl;
	}
#else
	Conjugate();
#endif
	// update velocity
	ApplyGradient << <dim3(ny, nz), nx >> > (f_ux, f_uy, f_uz, f_pressure, max_pos);
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

void Solver::Conjugate()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	int n = nx * ny * nz;

	InitConjugate << <dim3(ny, nz), nx >> > (r, f_div, x);

	// aTb operator, calc the sum of each block and then reduce all the data
	// note that the number of thread in each block cannot exceed 1024
	// the number here should self-adapte to the amount of euler girds, here is 64*64*256, should be modified whenever grid size is changed
	GlobalReduce << <dim3(32, 32), 1024 >> > (r, r, temp);
	BlockReduce << <1, 1024 >> > (temp);
	checkCudaErrors(cudaMemcpy(&init_rTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "init rTr: " << init_rTr << std::endl;

#if MGPCG
	MG_Preconditioner();
#else
	CopyFrom << <dim3(ny, nz), nx >> > (z, r);
#endif

	// p(0) = M^-1 r(0)
	UpdateP << <dim3(ny, nz), nx >> > (p, z, 0);

	GlobalReduce << <dim3(32, 32), 1024 >> > (z, r, temp);
	BlockReduce << <1, 1024 >> > (temp);
	checkCudaErrors(cudaMemcpy(&old_zTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 30; ++i)
	{
		// ��(k) = r(k)Tr(k) / p(k)TAp(k)
		ComputeAp << <dim3(ny, nz), nx >> > (Ap, p, max_pos);
		GlobalReduce << <dim3(32, 32), 1024 >> > (p, Ap, temp);
		BlockReduce << <1, 1024 >> > (temp);
		checkCudaErrors(cudaMemcpy(&pAp, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		float alpha = old_zTr / pAp;

		// x(k+1) = x(k) + ��(k)p(k), r(k+1) = r(k) - ��(k)Ap(k)
		UpdateResidual << <dim3(ny, nz), nx >> > (r, p, Ap, x, alpha);

		// if ||r(k+1)|| is sufficient enough small, break
		GlobalReduce << <dim3(32, 32), 1024 >> > (r, r, temp);
		BlockReduce << <1, 1024 >> > (temp);
		checkCudaErrors(cudaMemcpy(&rTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "iter " << i << " rTr: " << rTr << std::endl;

		// early stop
		if (rTr < init_rTr * 1e-14 || rTr * 10 > last_rTr || rTr == 0)
			break;

#if MGPCG
		MG_Preconditioner();
#else
		CopyFrom << <dim3(ny, nz), nx >> > (z, r);
#endif

		// ��(k) = r(k+1)Tr(k+1)/r(k)Tr(k)
		GlobalReduce << <dim3(32, 32), 1024 >> > (z, r, temp);
		BlockReduce << <1, 1024 >> > (temp);
		checkCudaErrors(cudaMemcpy(&new_zTr, &temp[0], sizeof(float), cudaMemcpyDeviceToHost));
		float beta = new_zTr / old_zTr;
		// p(k+1) = r(k+1) + ��(k)p(k)
		UpdateP << <dim3(ny, nz), nx >> > (p, z, beta);

		old_zTr = new_zTr;
		last_rTr = rTr;
	}

	CopyFrom << <dim3(ny, nz), nx >> > (f_pressure, x);
}

void Solver::MG_Preconditioner()
{
	int3 max_pos;
	max_pos.x = nx;
	max_pos.y = ny;
	max_pos.z = nz;

	int r_offset = nx * ny * nz;
	int offset = 0;

	// initialize z[l] and r[l] with 0 except r[0]
	Fill << <1, mg_space >> > (z, 0, 0);
	Fill << <1, mg_space - r_offset >> > (r, r_offset, 0);	

	// downsample
	for (int l = 0; l < mg_level - 1; ++l)
	{
		for (int i = 0; i < init_smooth_steps << l; ++i)
		{
			Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 0, max_pos);
			Smooth << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, 1, max_pos);
		}
		Restrict << <dim3(max_pos.y, max_pos.z), max_pos.x >> > (r, z, offset, max_pos);

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