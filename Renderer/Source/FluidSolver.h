#pragma once
#include <GLFW/glfw3.h>
#include <vector>

#include "Camera.h"
#include "Objects.h"

#define Nx 64
#define Ny 512
#define Nz 64
#define GRIDSTRIDE 0.02

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
	Camera camera;
	FluidObject fluidObject;
	Ball ball;
	RoundSource source;
	
private:
	GLFWwindow* window = nullptr;

	const unsigned int screenWidth = 1200;
	const unsigned int screenHeight = 1200;
};