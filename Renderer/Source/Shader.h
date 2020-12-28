#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

// shader resource handler, note that all the resource should be deleted after it is useless inorder to avoid memory leak
class Shader
{
public:
	Shader() {};

	unsigned int CreateVS(const char* vertexPath);

	unsigned int CreatePS(const char* fragmentPath);

	unsigned int CreateCS(const char* computePath);

	// attach shader
	void Attach(unsigned int programID, unsigned int shaderID);

	// link shader to program
	void Link(unsigned int programID);

	// use program
	void Use(unsigned int programID);

	unsigned int CreateProgram();

	void DeleteProgram(unsigned int programID);

	void RecordVAO(unsigned int vaoID);

	void RecordVBO(unsigned int vboID);

	void DeleteVAO(unsigned int vaoID);

	void DeleteVBO(unsigned int vboID);

	// maybe useless, free all the resource about shader
	void Finalize();

	// utility uniform functions
	inline void SetBool(unsigned int programID, const std::string& name, bool value) const
	{
		glUniform1i(glGetUniformLocation(programID, name.c_str()), (int)value);
	}

	inline void SetInt(unsigned int programID, const std::string& name, int value) const
	{
		glUniform1i(glGetUniformLocation(programID, name.c_str()), value);
	}

	inline void SetFloat(unsigned int programID, const std::string& name, float value) const
	{
		glUniform1f(glGetUniformLocation(programID, name.c_str()), value);
	}

	inline void SetVec2(unsigned int programID, const std::string& name, const glm::vec2 &value) const
	{
		glUniform2fv(glGetUniformLocation(programID, name.c_str()), 1, &value[0]);
	}

	inline void SetVec2(unsigned int programID, const std::string& name, float x, float y) const
	{
		glUniform2f(glGetUniformLocation(programID, name.c_str()), x, y);
	}

	inline void SetVec3(unsigned int programID, const std::string& name, const glm::vec3 &value) const
	{
		glUniform3fv(glGetUniformLocation(programID, name.c_str()), 1, &value[0]);
	}

	inline void SetVec3(unsigned int programID, const std::string& name, float x, float y, float z) const
	{
		glUniform3f(glGetUniformLocation(programID, name.c_str()), x, y, z);
	}

	inline void SetVec4(unsigned int programID, const std::string& name, const glm::vec4 &value) const
	{
		glUniform4fv(glGetUniformLocation(programID, name.c_str()), 1, &value[0]);
	}

	inline void SetVec4(unsigned int programID, const std::string& name, float x, float y, float z, float w)
	{
		glUniform4f(glGetUniformLocation(programID, name.c_str()), x, y, z, w);
	}

	inline void SetMat2(unsigned int programID, const std::string& name, const glm::mat2 &mat) const
	{
		glUniformMatrix2fv(glGetUniformLocation(programID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}

	inline void SetMat3(unsigned int programID, const std::string& name, const glm::mat3 &mat) const
	{
		glUniformMatrix3fv(glGetUniformLocation(programID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}

	inline void SetMat4(unsigned int programID, const std::string& name, const glm::mat4 &mat) const
	{
		glUniformMatrix4fv(glGetUniformLocation(programID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}

private:
	// utility function for checking shader compilation/linking errors.
	void CheckCompileErrors(GLuint shader, std::string type);

private:
	// program ID
	std::vector<unsigned int> programList;
	std::vector<unsigned int> shaderList;
	std::vector<unsigned int> vaoList;
	std::vector<unsigned int> vboList;
};