// Utils.h
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <stb_image_write.h>

#include <glad/glad.h>

#include <iostream>
#include <string>   
#include <fstream>  
#include <sstream>  
#include <filesystem>

struct ImageMetrics {
    double psnr;
    double ssim; // (간소화: 0.0으로 처리)
};

// PSNR 계산 함수
inline ImageMetrics CalculateMetrics(const std::vector<unsigned char>& original,
    const std::vector<unsigned char>& target,
    int width, int height)
{
    double mse = 0.0;
    size_t totalPixels = static_cast<size_t>(width) * height * 4; // RGBA

    // OpenMP가 가능하다면 #pragma omp parallel for reduction(+:mse)
    for (size_t i = 0; i < totalPixels; ++i) {
        double diff = static_cast<double>(original[i]) - static_cast<double>(target[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(totalPixels); // 채널 단위 평균

    // 8bit 이미지 기준 최대값 255
    double psnr = (mse < 1e-10) ? 100.0 : (10.0 * log10((255.0 * 255.0) / mse));
    return { psnr, 0.0 };
}


// 결과 저장 구조체
struct BenchmarkResult {
    std::string algoName;
    std::string resolution;
    float gpuTimeMs;
    float cpuTimeMs;
    double psnr;
};

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

// 텍스처 저장 유틸리티
inline void SaveTextureToPNG(GLuint textureID, int width, int height, std::string filename) {
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4);
}