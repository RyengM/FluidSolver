#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "Shader.h"
#include "FluidSolver.h"

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

	unsigned int sceneDepth;
	glGenTextures(1, &sceneDepth);
	glBindTexture(GL_TEXTURE_2D, sceneDepth);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, screenWidth, screenHeight);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

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
		-0.5, -0.5, -0.5,
		 0.5, -0.5, -0.5,
		 0.5,  0.5, -0.5,
		-0.5,  0.5, -0.5,
		-0.5,  0.5,  0.5,
		-0.5, -0.5,  0.5,
		 0.5, -0.5,  0.5,
		 0.5,  0.5,  0.5,
		-0.5,  0.5,  0.5,
		-0.5, -0.5,  0.5,

		-0.5, -0.5, -0.5,
		-0.5,  0.5, -0.5,

		-0.5,  0.5,  0.5,
		 0.5,  0.5,  0.5,

		 0.5,  0.5, -0.5,
		 0.5,  0.5,  0.5,

		 0.5, -0.5, -0.5,
		 0.5, -0.5,  0.5
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(bbVerts), bbVerts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	// declare raymarching program
	unsigned int raymarching = shader.CreateProgram();
	unsigned int raymarchingVert = shader.CreateVS("Shader/Raymarching.vert");
	unsigned int raymarchingFrag = shader.CreatePS("Shader/Raymarching.frag");
	shader.Attach(raymarching, raymarchingVert);
	shader.Attach(raymarching, raymarchingFrag);
	shader.Link(raymarching);

	// build box
	const float boxVertices[] = {
		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,

		-0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,

		-0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,

		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,

		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,
		-0.5f, -0.5f, -0.5f,

		-0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f
	};
	unsigned int boxVbo, boxVao;
	glGenVertexArrays(1, &boxVao);
	glGenBuffers(1, &boxVbo);
	glBindVertexArray(boxVao);
	glBindBuffer(GL_ARRAY_BUFFER, boxVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(boxVertices), boxVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	// fetch data from cu solver
	Solver solver = Solver();
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

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glClearColor(0.f, 0.f, 0.f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, screenWidth, screenHeight);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::scale(model, glm::vec3(Nx / 32, Ny / 32, Nz / 32));
		glm::mat4 view = staticCamera.GetViewMatrix();
		glm::mat4 projection = glm::perspective(glm::radians(staticCamera.fov), (float)screenWidth / (float)screenHeight, staticCamera.nearPlane, staticCamera.farPlane);
		// draw boundingbox
		shader.Use(boundingbox);
		shader.SetMat4(boundingbox, "model", model);
		shader.SetMat4(boundingbox, "view", view);
		shader.SetMat4(boundingbox, "projection", projection);
		glBindVertexArray(bbVao);
		glDrawArrays(GL_LINE_LOOP, 0, 10);
		glDrawArrays(GL_LINES, 10, 8);
		glBindVertexArray(0);

		glDisable(GL_DEPTH_TEST);
		if (!bPause)
		{
			// update density field
			solver.Update();
			rho = solver.GetDensityField();
			glBindTexture(GL_TEXTURE_3D, densityTexture);
			glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, Nx, Ny, Nz, 0, GL_RED, GL_FLOAT, &rho[0]);
		}

		// raymarching
		shader.Use(raymarching);
		shader.SetMat4(raymarching, "model", model);
		shader.SetMat4(raymarching, "view", view);
		shader.SetMat4(raymarching, "proj", projection);
		shader.SetVec3(raymarching, "bbMin", glm::vec3(-0.5, -0.5, -0.5));
		shader.SetVec3(raymarching, "bbMax", glm::vec3(0.5, 0.5, 0.5));
		shader.SetVec3(raymarching, "light.pos", glm::vec3(2.0, 1.0, 1.0));
		shader.SetVec3(raymarching, "light.strength", glm::vec3(0.584, 0.514, 0.451) * glm::vec3(5.0));
		shader.SetVec3(raymarching, "light.dir", glm::normalize(glm::vec3(0.0) - glm::vec3(2.0, 1.0, 1.0)));
		shader.SetVec3(raymarching, "eyePos", staticCamera.position);
		shader.SetVec3(raymarching, "lookDir", staticCamera.front);
		shader.SetFloat(raymarching, "nearPlane", staticCamera.nearPlane);
		shader.SetFloat(raymarching, "farPlane", staticCamera.farPlane);
		shader.SetFloat(raymarching, "xResolution", screenWidth);
		shader.SetFloat(raymarching, "yResolution", screenHeight);
		shader.SetInt(raymarching, "densityTexture", 0);
		shader.SetInt(raymarching, "sceneDepthTexture", 1);
		glBindImageTexture(1, sceneDepth, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
		glEnable(GL_TEXTURE_3D);
		glBindTexture(GL_TEXTURE_3D, densityTexture);
		glBindVertexArray(boxVao);
		glDrawArrays(GL_TRIANGLES, 0, sizeof(boxVertices));
		glBindVertexArray(0);

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

	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
		bPause = true;
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
		bPause = false;
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