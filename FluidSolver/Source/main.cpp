#include "FluidSolver.h"
#include "Camera.h"

int main()
{
	FluidSolver scene;

	scene.camera = Camera(glm::vec3(-2.f, 0.f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f));

	FluidObject fluid(glm::vec3(0, 0, 0), glm::vec3(0.32, 0.32, 1.28));
	scene.AddObject(fluid);

	scene.Render();
	
	return 0;
}