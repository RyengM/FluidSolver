#pragma once

#ifdef SOLVER_EXPORTS
#define SOLVER_API _declspec(dllexport)
#else
#define SOLVER_API _declspec(dllimport)
#endif

class SOLVER_API Solver
{
public:
	Solver(int nx, int ny, int nz, float rho, float temperature, float temperature_env, int max_iter, float dt, float curl_strength, float vel_x, float vel_y, float vel_z) :
		nx(nx), ny(ny), nz(nz), rho(rho), temperature(temperature), temperature_env(temperature_env), max_iter(max_iter), dt(dt), curl_strength(curl_strength), vel_x(vel_x), vel_y(vel_y), vel_z(vel_z) {};
	
	~Solver();

	void Initialize();

	void Update();

	void Advect();

	void Project();
	
	void Scatter();

	void ResetGridValue();

	// accelerate aTb operation
	void Reduce();

	void Restrict(int offset, int max_pos_x, int max_pos_y, int max_pos_z);

	void Conjugate();

	// multi grid preconditioner, to make M^-1Ax = M^-1b which has a smaller condition number in order to accelerate rate of convergence
	void MG_Preconditioner();

	float* GetDensityField();

private:
	// allocate memory
	void InitCuda();
	// init the param which should be initialized once
	void InitParam();

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
	float temperature = 0;
	float temperature_env = 0;
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
	// grid stride, used for converting particle from world space to grid index space
	float grid_stride = 0.1f;
	// source param
	float source_pos_x = 3.2f;
	float source_pos_y = 0.f;
	float source_pos_z = 3.2f;
	float source_radius = 1.0f;
	// the probability of particle awake
	float threshold = 0.f;

private:
	// Device
	// velocity field
	float* f_ux;
	float* f_uy;
	float* f_uz;
	// f_new_u in eulerian is used for alternate calculation
	// and in lagrangian is used for accounting the weight of grid
	float* f_new_ux;
	float* f_new_uy;
	float* f_new_uz;
	// density field
	float* f_rho;
	float* f_new_rho;
	// temperature field
	float* f_temperature;
	float* f_new_temperature;
	// pressure field
	float* f_pressure;
	float* f_new_pressure;
	// divergence field
	float* f_div;
	// conjugae variables
	float* r;						// residual
	float* z;						// M^-1 r
	float* p;						// conjugate gradient
	float* Ap;						// matrix-vector product
	float* x;						// solution
	float* temp;					// temp array for accelerating aTb

	// temp variable, used for data transfer from device to host, the length of the array is one
	float* d_temp_res;

	// particle information
	// note that the base world space coordinate in solver is (0, 0, 0)
	// so that particle position can be converted to voxel space immediately
	// mass
	float* p_mass;
	// position
	float* p_px;
	float* p_py;
	float* p_pz;
	// velocity
	float* p_ux;
	float* p_uy;
	float* p_uz;
	// when age equals 0, the particle is sleep
	// otherwise the particle is generated and flow with smoke
	float* p_age;
	
private:
	// Host
	// density field
	float* f_density;

	// variants for conjugate gradient
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