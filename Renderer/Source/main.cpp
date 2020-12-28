#include "FluidSolver.h"
#include "Camera.h"

int main()
{
	FluidSolver scene;

	Camera camera(glm::vec3(-15.f, 0.f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f));
	scene.camera = std::move(camera);

	FluidObject fluid(glm::vec3(0, 0, 0), glm::vec3(Nx * GRIDSTRIDE / 2, Ny * GRIDSTRIDE / 2, Nz * GRIDSTRIDE / 2));
	scene.fluidObject = std::move(fluid);

	//Ball ball(glm::vec3(0, 0.16, 0), 0.04f);
	//scene.ball = std::move(ball);

	RoundSource source(glm::vec3(0.f, 0.f, -Ny * GRIDSTRIDE / 2), 0.24f);
	scene.source = std::move(source);

	scene.Render();
	
	return 0;
}