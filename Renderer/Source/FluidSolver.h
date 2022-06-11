#pragma once
#include <GLFW/glfw3.h>
#include <vector>

#include <Solver.h>
#include "Camera.h"
#include "Objects.h"

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
	bool bPause = false;

	const unsigned int screenWidth = 1920;
	const unsigned int screenHeight = 1080;
};