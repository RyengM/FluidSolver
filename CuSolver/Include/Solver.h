#pragma once

#ifdef SOLVER_EXPORTS
#define SOLVER_API _declspec(dllexport)
#else
#define SOLVER_API _declspec(dllimport)
#endif

class SOLVER_API Solver
{
public:
	Solver(int nx, int ny, int nz, float rho, float tempeture, float tempeture_env, int max_iter, float dt, float curl_strength, float vel_x, float vel_y, float vel_z) :
		nx(nx), ny(ny), nz(nz), rho(rho), tempeture(tempeture), tempeture_env(tempeture_env), max_iter(max_iter), dt(dt), curl_strength(curl_strength), vel_x(vel_x), vel_y(vel_y), vel_z(vel_z) {};
	
	~Solver();

	void Initialize();

	void Update();

	void Reduce();

	void Restrict(int offset, int max_pos_x, int max_pos_y, int max_pos_z);

	void Conjugate();

	// multi grid preconditioner, to make M^-1Ax = M^-1b which has a smaller condition number in order to accelerate rate of convergence
	void MG_Preconditioner();

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
	// initial tempture
	float tempeture = 0;
	float tempeture_env = 0;
	// gravity
	float gravity = 9.8f;
	int max_iter = 0;
	// time step
	float dt = 0;
	// vorticity refinement coefficient
	float curl_strength = 0;
	// initial velocity
	float vel_x = 0;
	float vel_y = 0;
	float vel_z = 0;
	// frame
	int current_frame = 0;

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
	// tempeture field
	float* f_tempeture;
	float* f_new_tempeture;
	// pressure field
	float* f_pressure;
	float* f_new_pressure;
	// divergence field
	float* f_div;
	// average velocity field
	float* f_avgux;
	float* f_avguy;
	float* f_avguz;
	// vorticity field
	float* f_vortx;
	float* f_vorty;
	float* f_vortz;
	// conjugae variables
	float* r;						// residual
	float* z;						// M^-1 r
	float* p;						// conjugate gradient
	float* Ap;						// matrix-vector product
	float* x;						// solution
	float* temp;					// temp array for accelerating aTb

	// temp variable, used for data transfer from device to host, the length of the array is one
	float* d_temp_res;

private:
	// Host
	// density field
	float* f_density;

	int mg_level = 4;				// multi grid level
	int mg_space = 0;				// space needed to be allocated for multi grid
	int init_smooth_steps = 2;		// smooth steps for finest grid
	int bottom_smooth_steps = 50;	// smooth steps for coarsest grid

	// stride along search direction
	float alpha = 0;
	// stride for construct conjugate search direction
	float beta = 0;

	float init_rTr = 0;
	float rTr = 0;
	float last_rTr = 0;
	float old_zTr = 0;
	float new_zTr = 0;
	float pAp = 0;
};