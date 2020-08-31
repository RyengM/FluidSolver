#pragma once

#ifdef SOLVER_EXPORTS
#define SOLVER_API _declspec(dllexport)
#else
#define SOLVER_API _declspec(dllimport)
#endif

class SOLVER_API Solver
{
public:
	Solver(int nx, int ny, int nz, float rho, int max_iter, float dt, float vel_x, float vel_y, float vel_z) :
		nx(nx), ny(ny), nz(nz), rho(rho), max_iter(max_iter), dt(dt), vel_x(vel_x), vel_y(vel_y), vel_z(vel_z) {};
	
	~Solver();

	void Initialize();

	void Update();

	float* GetDensityField();

private:
	void InitCuda();

	void UpdateCuda();

	void FreeCuda();

private:
	// grid size
	int nx = 0;
	int ny = 0;
	int nz = 0;
	// initial density
	float rho = 0;
	int max_iter = 0;
	float dt = 0;
	// initial velocity
	float vel_x = 0;
	float vel_y = 0;
	float vel_z = 0;

private:
	// Device
	// velocity field
	float* f_ux;
	float* f_uy;
	float* f_uz;
	float* f_new_ux;
	float* f_new_uy;
	float* f_new_uz;
	// density field
	float* f_rho;
	float* f_new_rho;
	// pressure field
	float* f_p;
	float* f_new_p;
	// divergence field
	float* f_div;
	// vorticity field
	float* f_vortx;
	float* f_vorty;
	float* f_vortz;

private:
	// Host
	float* f_density;
};