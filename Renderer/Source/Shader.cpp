#include "Shader.h"

unsigned int Shader::CreateVS(const char* vertexPath)
{
	std::string vertexCode;
	std::ifstream vShaderFile;
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		vShaderFile.open(vertexPath);
		std::stringstream vShaderStream;
		vShaderStream << vShaderFile.rdbuf();
		vShaderFile.close();
		vertexCode = vShaderStream.str();
	}
	catch (std::ifstream::failure& e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}

	const char* vShaderCode = vertexCode.c_str();
	unsigned int vertex;
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	CheckCompileErrors(vertex, "VERTEX");

	return vertex;
}

unsigned int Shader::CreatePS(const char* fragmentPath)
{
	std::string fragmentCode;
	std::ifstream fShaderFile;
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		fShaderFile.open(fragmentPath);
		std::stringstream fShaderStream;
		fShaderStream << fShaderFile.rdbuf();
		fShaderFile.close();
		fragmentCode = fShaderStream.str();
	}
	catch (std::ifstream::failure& e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}

	const char* fShaderCode = fragmentCode.c_str();
	unsigned int fragment;
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	CheckCompileErrors(fragment, "FRAGMENT");

	return fragment;
}

unsigned int Shader::CreateCS(const char* computePath)
{
	std::string computeCode;
	std::ifstream cShaderFile;
	cShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		cShaderFile.open(computePath);
		std::stringstream cShaderStream;
		cShaderStream << cShaderFile.rdbuf();
		cShaderFile.close();
		computeCode = cShaderStream.str();
	}
	catch (std::ifstream::failure& e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}

	const char* cShaderCode = computeCode.c_str();
	unsigned int compute;
	compute = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(compute, 1, &cShaderCode, NULL);
	glCompileShader(compute);
	CheckCompileErrors(compute, "COMPUTE");

	return compute;
}

void Shader::Attach(unsigned int programID, unsigned int shaderID)
{
	glAttachShader(programID, shaderID);
	shaderList.emplace_back(shaderID);
}

void Shader::Link(unsigned int programID)
{
	glLinkProgram(programID);
	CheckCompileErrors(programID, "PROGRAM");
	for (unsigned int shader : shaderList)
		glDeleteShader(shader);
}

void Shader::Use(unsigned int programID)
{
	glUseProgram(programID);
}

unsigned int Shader::CreateProgram()
{
	unsigned int ID = glCreateProgram();
	programList.emplace_back(ID);
	return ID;
}

void Shader::DeleteProgram(unsigned int programID)
{
	for (std::vector<unsigned int>::iterator itr = programList.begin(); itr != programList.end(); ++itr)
	{
		if (*itr == programID)
		{
			programList.erase(itr);
			break;
		}
	}
	glDeleteProgram(programID);
}

void Shader::RecordVAO(unsigned int vaoID)
{
	vaoList.emplace_back(vaoID);
}

void Shader::RecordVBO(unsigned int vboID)
{
	vboList.emplace_back(vboID);
}

void Shader::DeleteVAO(unsigned int vaoID)
{
	for (std::vector<unsigned int>::iterator itr = vaoList.begin(); itr != vaoList.end(); ++itr)
	{
		if (*itr == vaoID)
		{
			vaoList.erase(itr);
			break;
		}
	}
	glDeleteVertexArrays(1, &vaoID);
}

void Shader::DeleteVBO(unsigned int vboID)
{
	for (std::vector<unsigned int>::iterator itr = vboList.begin(); itr != vboList.end(); ++itr)
	{
		if (*itr == vboID)
		{
			vboList.erase(itr);
			break;
		}
	}
	glDeleteVertexArrays(1, &vboID);
}

void Shader::Finalize()
{
	for (auto program : programList)
		glDeleteProgram(program);
	for (auto vao : vaoList)
		glDeleteVertexArrays(1, &vao);
	for (auto vbo : vboList)
		glDeleteBuffers(1, &vbo);
}

void Shader::CheckCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}