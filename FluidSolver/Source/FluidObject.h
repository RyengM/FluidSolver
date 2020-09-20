#pragma once
#include <glm/glm.hpp>

// we assume fluid object is a box
class FluidObject
{
public:
	FluidObject();
	FluidObject(glm::vec3 inPos, glm::vec3 inOffset) : pos(inPos), offset(inOffset) {};
	

public:
	// center of object
	glm::vec3 pos = glm::vec3(0, 0, 0);

	// half length of object
	glm::vec3 offset = glm::vec3(0, 0, 0);
};