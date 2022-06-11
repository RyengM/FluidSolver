#pragma once

#ifdef SOLVER_EXPORTS
#define SOLVER_API _declspec(dllexport)
#else
#define SOLVER_API _declspec(dllimport)
#endif

#define Nx 64
#define Ny 128
#define Nz 64

class SOLVER_API Solver
{
public:
	Solver() {};
	
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

	void Conjugate(float* res, float* field);

	// multi grid preconditioner, to make M^-1Ax = M^-1b which has a smaller condition number in order to accelerate rate of convergence
	void MG_Preconditioner();

	void PsiVCycle();

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
	int nx = Nx;
	int ny = Ny;
	int nz = Nz;
	// initial density
	float rho = 0.1f;
	// initial tempture
	float temperature = 2.f;
	float temperature_env = 0.f;
	// gravity
	float gravity = 9.8f;
	int max_iter = 30;
	float iter_precision = 1e-3;
	// time step
	float dt = 0.2f;
	// vorticity refinement coefficient
	float curl_strength = 0.1f;
	float ivock_scale = 0.1f;
	// frame
	int current_frame = 0;
	// rising smoke source param
	float source_pos_x = Nx / 2.f;
	float source_pos_y = 10.f;
	float source_pos_z = Nz / 2.f;
	float source_radius = 7.0f;
	// burst source param
	int particle_num = 128;
	float init_normal_radius = 3.f;     // normally distributed particle init position radius
	float init_normal_velocity = 2.f;   // normally distributed particle init velocity radius
	float burst_pos_x = Nx / 2.f;
	float burst_pos_y = Ny - 50.f;
	float burst_pos_z = Nz / 2.f;
	float particle_radius = 2.0f;
	float vel_decay = 0.99f;
	float density_decay = 0.99f;
	float temperature_decay = 0.8f;

private:
	// Device
	// burst particle
	float* p_posx;
	float* p_posy;
	float* p_posz;
	float* p_velx;
	float* p_vely;
	float* p_velz;
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
	// vorticity, vortx means the plane whose normal points to x direction, same as vorty and vortz
	// for conjugate convenience, we regard the dimension of f_vortx as nx*ny*nz, it is not right for the whole space, but we can wrap a layer around vort space
	float* f_vortx;
	float* f_vorty;
	float* f_vortz;
	float* f_new_vortx;
	float* f_new_vorty;
	float* f_new_vortz;
	float* f_psix;
	float* f_psiy;
	float* f_psiz;
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