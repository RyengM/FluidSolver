#pragma once
#include <GLFW/glfw3.h>
#include <vector>

#include "Camera.h"
#include "FluidObject.h"

// actually a scene to manage the objects in it, especially for fluids
class FluidSolver
{
public:
	FluidSolver() {};

	void Render();

	// init glfw
	void GLPrepare();

	void GLFinish();

	inline void AddObject(FluidObject obj)
	{
		fluidObjects.emplace_back(obj);
	}

public:
	// we assume there is always one camera now, so it is only a decorate now, which will be used later, maybe
	// so the settings about camera is global now, don't care
	std::vector<Camera> cameraList;
	std::vector<FluidObject> fluidObjects;

private:
	GLFWwindow* window = nullptr;

	const unsigned int screenWidth = 800;
	const unsigned int screenHeight = 800;
};