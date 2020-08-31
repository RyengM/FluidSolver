#include <glad/glad.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "FluidSolver.h"
#include "Shader.h"
#include <Solver.h>
#include <memory>

#define Nx 128
#define Ny 128
#define Nz 256

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void ProcessInput(GLFWwindow *window);

// camera
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw = -90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitch = 0.0f;
float lastX = 800.0f / 2.0;
float lastY = 600.0 / 2.0;
float fov = 45.0f;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

void FluidSolver::Render()
{
	GLPrepare();

	Shader shader;

	// declare ray creation program
	unsigned int createRay = shader.CreateProgram();

	unsigned int createRayComp = shader.CreateCS("Shader/CreateRay.comp");
	shader.Attach(createRay, createRayComp);
	shader.Link(createRay);

	// declare texture which saves our ray direction
	unsigned int rayDir;
	glGenTextures(1, &rayDir);
	glBindTexture(GL_TEXTURE_2D, rayDir);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, screenWidth, screenHeight);
	glBindTexture(GL_TEXTURE_2D, 0);

	// write ray direction to texture
	shader.Use(createRay);
	glBindImageTexture(0, rayDir, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(screenWidth / 8, screenHeight / 8, 1);
	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	shader.DeleteProgram(createRay);

	// declare raymarching program
	unsigned int raymarching = shader.CreateProgram();

	unsigned int raymarchingComp = shader.CreateCS("Shader/Raymarching.comp");
	shader.Attach(raymarching, raymarchingComp);
	shader.Link(raymarching);

	// declare texture which saves the raymarching result
	unsigned int raymarchingRes;
	glGenTextures(1, &raymarchingRes);
	glBindTexture(GL_TEXTURE_2D, raymarchingRes);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, screenWidth, screenHeight);
	glBindTexture(GL_TEXTURE_2D, 0);

	// declare vs and ps
	unsigned int render = shader.CreateProgram();

	unsigned int vert = shader.CreateVS("Shader/Test.vert");
	unsigned int frag = shader.CreatePS("Shader/Test.frag");
	shader.Attach(render, vert);
	shader.Attach(render, frag);
	shader.Link(render);

	unsigned int render_vao, render_vbo;
	glGenVertexArrays(1, &render_vao);
	glBindVertexArray(render_vao);
	glEnableVertexAttribArray(0);
	glGenBuffers(1, &render_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, render_vbo);
	shader.RecordVAO(render_vao);
	shader.RecordVBO(render_vbo);
	static const float verts[] =
	{
		-1.0f, -1.0f, 0.5f, 1.0f,
		 1.0f, -1.0f, 0.5f, 1.0f,
		 1.0f,  1.0f, 0.5f, 1.0f,
		-1.0f,  1.0f, 0.5f, 1.0f,
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// fetch data from cu solver
	Solver solver = Solver(Nx, Ny, Nz, 0.05f, 30, 0.02f, 0.f, 0.f, 10.f);
	solver.Initialize();
	solver.Update();
	// result to stack corruption, but data is right, ignore it now
	float* rho = solver.GetDensityField();

	// declare 3D texture	
	unsigned int densityTexture;
	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &densityTexture);
	glBindTexture(GL_TEXTURE_3D, densityTexture);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, Nx, Ny, Nz,0, GL_RED, GL_FLOAT, &rho[0]);
	glBindTexture(GL_TEXTURE_3D, 0);

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		// per-frame time logic
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		//std::cout << "fps:" << 1 / deltaTime << std::endl;

		ProcessInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// update density field
		solver.Update();
		rho = solver.GetDensityField();
		glBindTexture(GL_TEXTURE_3D, densityTexture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, Nx, Ny, Nz, 0, GL_RED, GL_FLOAT, &rho[0]);

		// calc raymarching result
		shader.Use(raymarching);
		shader.SetVec3(raymarching, "objectPos", glm::vec3(0, 0, 0));
		shader.SetVec3(raymarching, "objectOffset", glm::vec3(0.6, 0.6, 1.28));
		shader.SetVec3(raymarching, "cameraPos", cameraPos);
		shader.SetVec3(raymarching, "camForward", cameraFront);
		glBindImageTexture(0, rayDir, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		glBindImageTexture(1, raymarchingRes, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
		glEnable(GL_TEXTURE_3D);
		glActiveTexture(densityTexture);
		glBindTexture(GL_TEXTURE_3D, densityTexture);
		glDispatchCompute(screenWidth / 8, screenHeight / 8, 1);
		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// draw result to canvas
		shader.Use(render);
		glActiveTexture(raymarchingRes);
		glBindTexture(GL_TEXTURE_2D, raymarchingRes);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	shader.Finalize();
	GLFinish();
}

void FluidSolver::GLPrepare()
{
	// glfw: initialize and configure
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// glfw window creation
	window = glfwCreateWindow(screenWidth, screenHeight, "MySolver", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return ;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback); 

	// tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return ;
	}
}

void FluidSolver::GLFinish()
{
	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void ProcessInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = 2.5 * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// make sure that when pitch is out of bounds, screen doesn't get flipped
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	fov -= (float)yoffset;
	if (fov < 1.0f)
		fov = 1.0f;
	if (fov > 45.0f)
		fov = 45.0f;
}