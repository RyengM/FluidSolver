#include "FluidSolver.h"
#include "Camera.h"

int main()
{
	FluidSolver scene;

	Camera camera(glm::vec3(-5.f, 0.f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f));
	scene.camera = std::move(camera);

	FluidObject fluid(glm::vec3(0, 0, 0), glm::vec3(0.32, 0.32, 1.28));
	scene.AddObject(std::move(fluid));

	scene.Render();
	
	return 0;
}