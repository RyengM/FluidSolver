#include "FluidSolver.h"
#include "Camera.h"

int main()
{
	FluidSolver scene;

	Camera camera(glm::vec3(-5.f, 0.f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f));
	scene.camera = std::move(camera);

	FluidObject fluid(glm::vec3(0, 0, 0), glm::vec3(Nx / 200.f, Ny / 200.f, Nz / 200.f));
	scene.fluidObject = std::move(fluid);

	Ball ball(glm::vec3(0, 0.16, 0), 0.04);
	scene.ball = std::move(ball);

	scene.Render();
	
	return 0;
}