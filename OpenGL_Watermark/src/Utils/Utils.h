// Utils.h
#pragma once
#include <glad/glad.h>

#include <iostream>
#include <string>   
#include <fstream>  
#include <sstream>  
#include <filesystem>

// --- 셰이더 파일(.comp)을 읽어 C++ string으로 반환하는 유틸리티 ---
inline std::string loadShaderSourceFromFile(const char* shaderFileName)
{
    namespace fs = std::filesystem;

    // 우리가 찾고 싶은 상대 경로: 프로젝트 루트 기준 "src/Shaders"
    const fs::path relativeShaderDir = fs::path("src") / "Shaders";

    // 1) 현재 작업 디렉토리부터 시작해서
    fs::path current = fs::current_path();
    fs::path shaderDir;

    // 2) 부모 디렉토리로 한 칸씩 올라가면서 src/Shaders 존재 여부 확인
    for (fs::path p = current; !p.empty(); p = p.parent_path())
    {
        fs::path candidate = p / relativeShaderDir;
        if (fs::exists(candidate) && fs::is_directory(candidate))
        {
            shaderDir = candidate;
            break;
        }
    }

    if (shaderDir.empty())
    {
        std::cerr << "[ShaderLoader] Could not find shader directory: "
            << relativeShaderDir << "\n"
            << "Current path: " << fs::current_path() << std::endl;
        return {};
    }

    // 최종 셰이더 파일 경로: (...)/src/Shaders/ + 파일 이름
    fs::path shaderPath = shaderDir / shaderFileName;

    if (!fs::exists(shaderPath))
    {
        std::cerr << "[ShaderLoader] Shader file not found: "
            << shaderPath << std::endl;
        return {};
    }

    std::string shaderCode;
    std::ifstream shaderFile;
    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        shaderFile.open(shaderPath);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        shaderCode = shaderStream.str();
    }
    catch (std::ifstream::failure& e)
    {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << shaderPath << std::endl;
        return "";
    }
    return shaderCode;
}

// --- 컴퓨트 셰이더를 컴파일/링크하는 유틸리티 ---
inline GLuint loadComputeShader(const char* computePath)
{
    std::string computeCodeString = loadShaderSourceFromFile(computePath);
    if (computeCodeString.empty()) return 0;
    const char* computeCode = computeCodeString.c_str();

    const GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeCode, nullptr);
    glCompileShader(computeShader);

    GLint success;
    GLchar infoLog[1024];
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(computeShader, 1024, nullptr, infoLog);
        std::cerr << "ERROR::COMPUTE_SHADER::COMPILATION_FAILED\n" << computePath << "\n" << infoLog << std::endl;
        glDeleteShader(computeShader);
        return 0;
    }

    const GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, computeShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 1024, nullptr, infoLog);
        std::cerr << "ERROR::COMPUTE_PROGRAM::LINKING_FAILED\n" << computePath << "\n" << infoLog << std::endl;
        glDeleteShader(computeShader);
        glDeleteProgram(shaderProgram);
        return 0;
    }
    glDeleteShader(computeShader);
    std::cout << "Compute Shader '" << computePath << "' loaded successfully." << '\n';
    return shaderProgram;
}