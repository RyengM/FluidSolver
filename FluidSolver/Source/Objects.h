#pragma once
#include <glm/glm.hpp>


// we assume fluid object is a box
class FluidObject
{
public:
	FluidObject() {};
	FluidObject(glm::vec3 inPos, glm::vec3 inOffset) : pos(inPos), offset(inOffset) {};

public:
	// center of object
	glm::vec3 pos = glm::vec3(0, 0, 0);

	// half length of object
	glm::vec3 offset = glm::vec3(0, 0, 0);
};

class Ball
{
public:
	Ball() {};
	Ball(glm::vec3 inPos, float inRadius) : pos(inPos), radius(inRadius) {};

private:
	glm::vec3 pos = glm::vec3(0, 0, 0);
	float radius = 0.f;
};