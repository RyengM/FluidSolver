#pragma once
#include <GLFW/glfw3.h>
#include <vector>

#include "Camera.h"
#include "Objects.h"

#define Nx 64
#define Ny 256
#define Nz 64

// actually a scene to manage the objects in it, especially for fluids
class FluidSolver
{
public:
	FluidSolver() {};

	void Render();

	// init glfw
	void GLPrepare();

	void GLFinish();

public:
	void ProcessInput(GLFWwindow *window);

public:
	// we assume there is always one camera now, so it is only a decorate now, which will be used later, maybe
	// so the settings about camera is global now, don't care
	Camera camera;
	FluidObject fluidObject;
	Ball ball;
	
private:
	GLFWwindow* window = nullptr;

	const unsigned int screenWidth = 1200;
	const unsigned int screenHeight = 1200;
};