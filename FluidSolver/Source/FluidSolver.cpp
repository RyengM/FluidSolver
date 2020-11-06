#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "Shader.h"
#include "FluidSolver.h"

#include <Solver.h>
#include <memory>

Camera staticCamera;
// mouse position at last frame
float lastX;
float lastY;
bool firstMouse = true;
// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

void FluidSolver::Render()
{
	GLPrepare();

	Shader shader;

	// declare frame buffer for first pass
	unsigned int fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	// declare render target and scene depth
	unsigned int renderTarget;
	glGenTextures(1, &renderTarget);
	glBindTexture(GL_TEXTURE_2D, renderTarget);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, screenWidth, screenHeight);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	unsigned int sceneDepth;
	glGenTextures(1, &sceneDepth);
	glBindTexture(GL_TEXTURE_2D, sceneDepth);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, screenWidth, screenHeight);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTarget, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, sceneDepth, 0);

	// declare vs and ps for smoke boundingbox wireframe
	unsigned int boundingbox = shader.CreateProgram();

	unsigned int bbVert = shader.CreateVS("Shader/Boundingbox.vert");
	unsigned int bbFrag = shader.CreatePS("Shader/Boundingbox.frag");
	shader.Attach(boundingbox, bbVert);
	shader.Attach(boundingbox, bbFrag);
	shader.Link(boundingbox);

	unsigned int bbVao, bbVbo;
	glGenVertexArrays(1, &bbVao);
	glBindVertexArray(bbVao);
	glEnableVertexAttribArray(0);
	glGenBuffers(1, &bbVbo);
	glBindBuffer(GL_ARRAY_BUFFER, bbVbo);
	shader.RecordVAO(bbVao);
	shader.RecordVBO(bbVbo);
	glm::vec3 bbPos = fluidObject.pos;
	glm::vec3 bbOffset = fluidObject.offset;
	static const float bbVerts[] =
	{
		bbPos.x - bbOffset.x, bbPos.y - bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y - bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y + bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x - bbOffset.x, bbPos.y + bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x - bbOffset.x, bbPos.y + bbOffset.y, bbPos.z + bbOffset.z,
		bbPos.x - bbOffset.x, bbPos.y - bbOffset.y, bbPos.z + bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y - bbOffset.y, bbPos.z + bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y + bbOffset.y, bbPos.z + bbOffset.z,
		bbPos.x - bbOffset.x, bbPos.y + bbOffset.y, bbPos.z + bbOffset.z,
		bbPos.x - bbOffset.x, bbPos.y - bbOffset.y, bbPos.z + bbOffset.z,

		bbPos.x - bbOffset.x, bbPos.y - bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x - bbOffset.x, bbPos.y + bbOffset.y, bbPos.z - bbOffset.z,

		bbPos.x - bbOffset.x, bbPos.y + bbOffset.y, bbPos.z + bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y + bbOffset.y, bbPos.z + bbOffset.z,

		bbPos.x + bbOffset.x, bbPos.y + bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y + bbOffset.y, bbPos.z + bbOffset.z,

		bbPos.x + bbOffset.x, bbPos.y - bbOffset.y, bbPos.z - bbOffset.z,
		bbPos.x + bbOffset.x, bbPos.y - bbOffset.y, bbPos.z + bbOffset.z
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(bbVerts), bbVerts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

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

	// declare vs and ps for final draw
	unsigned int render = shader.CreateProgram();
	unsigned int vert = shader.CreateVS("Shader/Test.vert");
	unsigned int frag = shader.CreatePS("Shader/Test.frag");
	shader.Attach(render, vert);
	shader.Attach(render, frag);
	shader.Link(render);

	unsigned int renderVao, renderVbo;
	glGenVertexArrays(1, &renderVao);
	glBindVertexArray(renderVao);
	glEnableVertexAttribArray(0);
	glGenBuffers(1, &renderVbo);
	glBindBuffer(GL_ARRAY_BUFFER, renderVbo);
	shader.RecordVAO(renderVao);
	shader.RecordVBO(renderVbo);
	static const float verts[] =
	{
		-1.0f, -1.0f, 0.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f,
		 1.0f,  1.0f, 0.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 1.0f,
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// fetch data from cu solver
	Solver solver = Solver(Nx, Ny, Nz, 0.08f, 20.f, 20.f, 30, 0.04f, 0.f, 0.f, 40.f, 0.f);
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

		std::cout << "fps:" << 1.f / deltaTime << std::endl;

		ProcessInput(window);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.f, 0.f, 0.f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw boundingbox
		shader.Use(boundingbox);
		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = staticCamera.GetViewMatrix();
		glm::mat4 projection = glm::perspective(glm::radians(staticCamera.fov), (float)screenWidth / (float)screenHeight, staticCamera.nearPlane, staticCamera.farPlane);
		shader.SetMat4(boundingbox, "model", model);
		shader.SetMat4(boundingbox, "view", view);
		shader.SetMat4(boundingbox, "projection", projection);
		glBindVertexArray(bbVao);
		glDrawArrays(GL_LINE_LOOP, 0, 10);
		glDrawArrays(GL_LINES, 10, 8);
		glBindVertexArray(0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);
		glClearColor(0.f, 0.f, 0.f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// update density field
		solver.Update();
		rho = solver.GetDensityField();
		glBindTexture(GL_TEXTURE_3D, densityTexture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, Nx, Ny, Nz, 0, GL_RED, GL_FLOAT, &rho[0]);

		// calc raymarching result
		shader.Use(raymarching);
		shader.SetVec3(raymarching, "objectPos", bbPos);
		shader.SetVec3(raymarching, "objectOffset", bbOffset);
		shader.SetVec3(raymarching, "cameraPos", staticCamera.position);
		shader.SetVec3(raymarching, "camForward", staticCamera.front);
		shader.SetVec3(raymarching, "camUp", staticCamera.up);
		shader.SetVec3(raymarching, "camRight", staticCamera.right);
		shader.SetVec2(raymarching, "resolution", glm::vec2(screenWidth, screenHeight));
		shader.SetFloat(raymarching, "fov", staticCamera.fov);
		shader.SetFloat(raymarching, "nearPlane", staticCamera.nearPlane);
		shader.SetFloat(raymarching, "farPlane", staticCamera.farPlane);
		glBindImageTexture(0, raymarchingRes, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
		glBindImageTexture(1, renderTarget, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		glBindImageTexture(2, sceneDepth, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
		glEnable(GL_TEXTURE_3D);
		glBindTexture(GL_TEXTURE_3D, densityTexture);
		glDispatchCompute(screenWidth / 8, screenHeight / 8, 1);
		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// draw result to canvas
		shader.Use(render);
		glBindTexture(GL_TEXTURE_2D, raymarchingRes);
		glBindVertexArray(renderVao);
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
	typedef void* (*FUNC)(GLFWwindow*, int, int);
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

	// init static camera for callback
	staticCamera = camera;
	lastX = screenWidth / 2.f;
	lastY = screenHeight / 2.f;
}

void FluidSolver::GLFinish()
{
	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void FluidSolver::ProcessInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		staticCamera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		staticCamera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		staticCamera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		staticCamera.ProcessKeyboard(RIGHT, deltaTime);

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
		std::cout << "fuck" << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	// reversed since y-coordinates go from bottom to top
	float yoffset = lastY - ypos; 

	lastX = xpos;
	lastY = ypos;

	staticCamera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	staticCamera.ProcessMouseScroll(yoffset);
}