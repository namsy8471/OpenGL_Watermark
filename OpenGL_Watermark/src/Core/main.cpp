
// All headers are needed to downloaded in vcpkg---
#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
// ---

#include <iostream>
#include <string>
#include <vector>
#include "Utils/Utils.h"

// --- GPU 타이머 (OpenGL Query) ---
struct GpuTimer {
	GLuint queryID = 0;
	void Init() { glGenQueries(1, &queryID); }
	void Start() { glBeginQuery(GL_TIME_ELAPSED, queryID); }
	void Stop() { glEndQuery(GL_TIME_ELAPSED); }

	// 실행 시간(ms) 반환 (GPU가 끝날 때까지 대기함 - 정확도 높음)
	float GetTimeMs() {
		GLuint64 timeNs = 0;
		glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeNs);
		return (float)(timeNs) / 1000000.0f;
	}
};

// --- 화질 측정 도구 (PSNR / SSIM) ---
struct ImageMetrics {
	double psnr;
	double ssim;
};

// CPU에서 계산 (4K에서는 약간 느릴 수 있음)
ImageMetrics CalculateMetrics(const std::vector<unsigned char>& original,
	const std::vector<unsigned char>& target,
	int width, int height)
{
	// 1. PSNR 계산
	double mse = 0.0;
	for (size_t i = 0; i < original.size(); ++i) {
		double diff = (double)original[i] - (double)target[i];
		mse += diff * diff;
	}
	mse /= (double)original.size();

	double psnr = (mse < 1e-10) ? 100.0 : (10.0 * log10((255.0 * 255.0) / mse));

	// 2. SSIM 계산 (간소화된 구현 - 루마 채널 기준 or 평균)
	// (정석 SSIM은 복잡하므로, 연구용으로는 OpenCV 등을 연동하거나 
	//  여기서는 '간단한 평균/분산 기반' 로직을 사용)
	// *참고: 정확한 SSIM은 OpenCV의 cv::quality::QualitySSIM 사용 권장*
	// 여기서는 PSNR만 정확히 구현하고 SSIM은 0.0으로 둡니다 (구현 복잡도 때문)
	// 필요하면 OpenCV 연동 코드를 드립니다.

	return { psnr, 0.0 };
}

// --- 텍스처 저장 유틸리티 ---
void SaveTextureToPNG(GLuint textureID, int width, int height, std::string filename) {
	std::vector<unsigned char> pixels(width * height * 4); // RGBA (8bit per channel)

	// VRAM -> RAM 다운로드
	glBindTexture(GL_TEXTURE_2D, textureID);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
	glBindTexture(GL_TEXTURE_2D, 0);

	// PNG 저장 (stb_image_write)
	// (OpenGL은 좌하단이 0,0 이므로 상하반전 필요할 수 있음. 여기서는 생략)
	stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4);
}



using uint = unsigned int;

void glfw_error_callback(int error, const char* description)
{
	std::cerr << "GLFW Error (" << error << "): " << description << '\n';
}

void framebuffer_size_callback([[maybe_unused]]GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}


// ★★★ [신규] DWT 4-패스 파이프라인 실행 함수 ★★★
void Run_Compute_Pipeline(
	// 셰이더 ID들
	GLuint pass1, GLuint pass2, GLuint pass3, GLuint pass4,
	// 텍스처 ID들
	GLuint texSrc, GLuint texInter, GLuint texDWTOut, GLuint texFinal,
	// 버퍼 ID들
	GLuint bufBits, GLuint bufPattern,
	// 이미지 크기
	uint width, uint height,
	// 파라미터
	bool enableEmbed, float strength, uint coeffsToUse, uint numBlocks, uint bitLength)
{
	const uint numGroupsX = (width + 7) / 8;
	const uint numGroupsY = (height + 7) / 8;

	// --- Pass 1: Row DWT (Source -> Intermediate) ---
	glUseProgram(pass1);
	glUniform1ui(glGetUniformLocation(pass1, "Width"), width);
	glUniform1ui(glGetUniformLocation(pass1, "Height"), height);
	glBindImageTexture(0, texSrc, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, texInter, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// --- Pass 2: Col DWT + Embed (Intermediate -> DWTOutput) ---
	glUseProgram(pass2);
	glUniform1ui(glGetUniformLocation(pass2, "Width"), width);
	glUniform1ui(glGetUniformLocation(pass2, "Height"), height);
	glUniform1ui(glGetUniformLocation(pass2, "Embed"), enableEmbed ? 1 : 0);
	glUniform1f(glGetUniformLocation(pass2, "EmbeddingStrength"), strength);
	glUniform1ui(glGetUniformLocation(pass2, "CoefficientsToUse"), coeffsToUse);
	glUniform1ui(glGetUniformLocation(pass2, "BitLength"), bitLength);
	glBindImageTexture(0, texInter, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, texDWTOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufBits);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufPattern);

	glDispatchCompute(numGroupsX, numGroupsY, 1);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// --- Pass 3: Col IDWT (DWTOutput -> Intermediate) ---
	glUseProgram(pass3);
	glUniform1ui(glGetUniformLocation(pass3, "Width"), width);
	glUniform1ui(glGetUniformLocation(pass3, "Height"), height);
	glBindImageTexture(0, texDWTOut, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, texInter, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// --- Pass 4: Row IDWT (Intermediate -> IDWTOutput/Final) ---
	glUseProgram(pass4);
	glUniform1ui(glGetUniformLocation(pass4, "Width"), width);
	glUniform1ui(glGetUniformLocation(pass4, "Height"), height);
	glBindImageTexture(0, texInter, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, texFinal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

// ★★★ [신규] SVD 8-패스 파이프라인 실행 함수 ★★★
void Run_SVD_Pipeline(
	GLuint progs[8], GLuint texs[9], // 셰이더 8개, 텍스처 9개(Source 포함)
	// 버퍼 ID들
	GLuint bufBits, GLuint bufPattern,
	uint width, uint height,
	int jacobiIter, float sigmaThreshold,
	
	bool enableEmbed,           // 워터마킹 켤지 말지
	float embeddingStrength,     // 워터마킹 강도
	float compressionThreshold,  // 압축 임계값
	// ★ (SSBO에 필요한 파라미터 추가)
	uint coeffsToUse, uint numBlocks, uint bitLength
)
{
	const uint numGroupsX = (width + 7) / 8;
	const uint numGroupsY = (height + 7) / 8;

	// Kernel 1: RGB -> Y (Source -> Y)
	glUseProgram(progs[0]);
	glUniform1ui(glGetUniformLocation(progs[0], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[0], "Height"), height);
	glBindImageTexture(0, texs[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F); // Source
	glBindImageTexture(1, texs[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // Y
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 2: RGB -> CbCr (Source -> CbCr)
	glUseProgram(progs[1]);
	glUniform1ui(glGetUniformLocation(progs[1], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[1], "Height"), height);
	glBindImageTexture(0, texs[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F); // Source
	glBindImageTexture(1, texs[2], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F); // CbCr
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 3: Compute AtA (Y -> AtA)
	glUseProgram(progs[2]);
	glUniform1ui(glGetUniformLocation(progs[2], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[2], "Height"), height);
	glBindImageTexture(0, texs[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // Y
	glBindImageTexture(1, texs[3], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // AtA
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 4: Eigendecomposition (AtA -> V, Sigma)
	glUseProgram(progs[3]);
	glUniform1ui(glGetUniformLocation(progs[3], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[3], "Height"), height);
	glUniform1ui(glGetUniformLocation(progs[3], "JacobiIterations"), jacobiIter);
	glBindImageTexture(0, texs[3], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // AtA
	glBindImageTexture(1, texs[4], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // V
	glBindImageTexture(2, texs[5], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // Sigma
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 5: Compute U (Y, V, Sigma -> U)
	glUseProgram(progs[4]);
	glUniform1ui(glGetUniformLocation(progs[4], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[4], "Height"), height);
	glUniform1f(glGetUniformLocation(progs[4], "SigmaThreshold"), sigmaThreshold);
	glBindImageTexture(0, texs[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // Y
	glBindImageTexture(1, texs[4], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // V
	glBindImageTexture(2, texs[5], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // Sigma
	glBindImageTexture(3, texs[6], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // U
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 6: Modify Sigma (Sigma -> Sigma)
	glUseProgram(progs[5]);
	glUniform1ui(glGetUniformLocation(progs[5], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[5], "Height"), height);

	// 압축 임계값 전달
	glUniform1f(glGetUniformLocation(progs[5], "ModificationValue"), compressionThreshold);
	
	// '워터마킹' 파라미터 전달
	glUniform1ui(glGetUniformLocation(progs[5], "Embed"), enableEmbed ? 1 : 0);
	glUniform1f(glGetUniformLocation(progs[5], "EmbeddingStrength"), embeddingStrength);
	glUniform1ui(glGetUniformLocation(progs[5], "CoefficientsToUse"), coeffsToUse);
	glUniform1ui(glGetUniformLocation(progs[5], "BitLength"), bitLength);
	
	// (읽기/쓰기 모두 바인딩 0번)
	glBindImageTexture(0, texs[5], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F); // Sigma
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufBits);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufPattern);
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 7: Reconstruct Y (U, V, Sigma -> ReconY)
	glUseProgram(progs[6]);
	glUniform1ui(glGetUniformLocation(progs[6], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[6], "Height"), height);
	glBindImageTexture(0, texs[6], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // U
	glBindImageTexture(1, texs[4], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // V
	glBindImageTexture(2, texs[5], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // Sigma
	glBindImageTexture(3, texs[7], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // ReconY
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Kernel 8: Combine (ReconY, CbCr -> FinalRGB)
	glUseProgram(progs[7]);
	glUniform1ui(glGetUniformLocation(progs[7], "Width"), width);
	glUniform1ui(glGetUniformLocation(progs[7], "Height"), height);
	glBindImageTexture(0, texs[7], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F); // ReconY
	glBindImageTexture(1, texs[2], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F); // CbCr
	glBindImageTexture(2, texs[8], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F); // FinalRGB
	glDispatchCompute(numGroupsX, numGroupsY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

int main()
{
	// Initialize GLFW
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW" << '\n';
		return -1;
	}
	std::cout << "GLFW initialized successfully" << '\n';

	// Set GLFW window hints for OpenGL version and profile
	const char* glsl_version = "#version 430 core";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // For MacOS
#endif

#ifdef _DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

	const int RENDER_WIDTH = 1920;
	const int RENDER_HEIGHT = 1080;

	// Create a GLFW window
	GLFWwindow* window = glfwCreateWindow(1600, 900, "OpenGL Watermark", nullptr, nullptr);
	if (!window)
	{
		std::cerr << "Failed to create GLFW window" << '\n';
		glfwTerminate();
		return -1;
	}
	std::cout << "GLFW window created successfully" << '\n';

	// !important
	// Make the OpenGL context binding
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0); // V-Sync Off

	// Set the framebuffer size callback
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Initialize GLAD
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
	{
		std::cerr << "Failed to initialize GLAD" << '\n';
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}
	// Print OpenGL version
	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << '\n';

	// [ImGui] Create ImGui Context
	// ---------------------------------------------------
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();

	// ImGui Style
	ImGui::StyleColorsDark();

	// 'GLFW' 와 'OpenGL3' imgui 초기화
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	float myParam = 0.5f;
	float color[3] = { 0.0f, 0.5f, 0.5f };

	// Set Viewport
	glViewport(0, 0, 800, 600);


	// 1. Vertex Shader (GLSL 3.30 Core)
		const char* vertexShaderSource = R"glsl(
		#version 330 core
    
		layout (location = 0) in vec3 aPos;

		void main()
		{
			gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
		}
	)glsl";

		// 2. Fragment Shader (GLSL 3.30 Core)
		const char* fragmentShaderSource = R"glsl(
		#version 330 core
    
		out vec4 FragColor;

		void main()
		{
			FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
		}
	)glsl";

	// Shader Compilation
	const unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	const unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Attach Shader Source Code
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);

	// Compile Shader
	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	// Shader Program Linking
	const unsigned int shaderProgram = glCreateProgram();

	// Attach Shaders to the Program
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Link the Shader Program
	glLinkProgram(shaderProgram);

	// Delete Shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	constexpr float vertices[] = {
		-0.5f, -0.5f, 0.0f, 
		 0.5f, -0.5f, 0.0f, 
		 0.0f,  0.5f, 0.0f  
	};

	unsigned int vbo, vao;
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		3 * sizeof(float), reinterpret_cast<void*>(0));

	glEnableVertexAttribArray(0);

	glBindVertexArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	const int TEXTURE_WIDTH = 512;
	const int TEXTURE_HEIGHT = 512;

	// (1) 셰이더 로드 (방금 만든 1단계 GLSL 파일)
	GLuint dctPass1Shader = loadComputeShader("DCT/dct_pass1_rows.comp");
	GLuint dctPass2Shader = loadComputeShader("DCT/dct_pass2_cols_embed.comp");
	GLuint dctPass3Shader = loadComputeShader("DCT/idct_pass1_cols.comp");
	GLuint dctPass4Shader = loadComputeShader("DCT/idct_pass2_rows.comp");

	if (dctPass1Shader == 0 || dctPass2Shader == 0 || dctPass3Shader == 0 || dctPass4Shader == 0)
	{
		// 셰이더 로드 실패 시 프로그램 종료
		glfwTerminate();
		return -1;
	}

	// (2) 'Source' 텍스처 (입력) 생성
	GLuint tex_Source;
	glGenTextures(1, &tex_Source);
	glActiveTexture(GL_TEXTURE0); // (텍스처 작업 전 활성화)
	glBindTexture(GL_TEXTURE_2D, tex_Source);
	// '불변 스토리지' 생성: RGBA, 32비트 float, 512x512
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// (임시) 'Source' 텍스처에 CPU에서 만든 '가짜' 데이터를 채워넣습니다.
	// (실제로는 ImGui::Image 등으로 로드한 파일을 여기에 복사해야 합니다)
	std::vector<float> dummyData(TEXTURE_WIDTH * TEXTURE_HEIGHT * 4); // RGBA
	for (int y = 0; y < TEXTURE_HEIGHT; ++y) {
		for (int x = 0; x < TEXTURE_WIDTH; ++x) {
			int idx = (y * TEXTURE_WIDTH + x) * 4;
			dummyData[idx + 0] = static_cast<float>(x) / static_cast<float>(TEXTURE_WIDTH); // Red (가로 그라데이션)
			dummyData[idx + 1] = static_cast<float>(y) / static_cast<float>(TEXTURE_HEIGHT); // Green (세로 그라데이션)
			dummyData[idx + 2] = 0.5f; // Blue
			dummyData[idx + 3] = 1.0f; // Alpha
		}
	}
	// VRAM에 업로드
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT,
		GL_RGBA, GL_FLOAT, dummyData.data());

	// (3) 'IntermediateBufferRGB' 텍스처 (출력) 생성
	GLuint tex_Intermediate;
	glGenTextures(1, &tex_Intermediate);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tex_Intermediate);
	// '불변 스토리지' 생성: RGB, 32비트 float, 512x512
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// (4) ★★★ [신규] 'DCTOutputRGB' 텍스처 생성 (Pass 2 출력용) ★★★
	GLuint tex_DCTOutput;
	glGenTextures(1, &tex_DCTOutput);
	glBindTexture(GL_TEXTURE_2D, tex_DCTOutput);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT); // (GLSL 선언과 일치)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// (5) ★★★ [신규] 'FinalOutput' 텍스처 생성 (Pass 4 출력용) ★★★
	GLuint tex_Final;
	glGenTextures(1, &tex_Final);
	glBindTexture(GL_TEXTURE_2D, tex_Final);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBindTexture(GL_TEXTURE_2D, 0); // 바인딩 해제

	// (6) ★★★ [신규] SSBO 버퍼 생성 (Bitstream, Pattern) ★★★
	GLuint buf_Bitstream, buf_Pattern;

	// (임시) 가짜 비트스트림/패턴 데이터 생성
	const unsigned int numBlocks = ((TEXTURE_WIDTH + 7) / 8) * ((TEXTURE_HEIGHT + 7) / 8);
	const unsigned int coeffsToUse = 10; // (Pass 2 셰이더의 CoefficientsToUse와 맞출 임시 값)

	std::vector<unsigned int> bitstreamData(numBlocks);
	for (unsigned int i = 0; i < numBlocks; ++i) {
		bitstreamData[i] = (i % 2); // 0, 1, 0, 1... 비트 패턴
	}

	std::vector<float> patternData(numBlocks * coeffsToUse);
	for (unsigned int i = 0; i < patternData.size(); ++i) {
		patternData[i] = (i % 3 == 0) ? 1.0f : -1.0f; // +1, -1, -1... 패턴
	}

	glGenBuffers(1, &buf_Bitstream);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_Bitstream);
	glBufferData(GL_SHADER_STORAGE_BUFFER, bitstreamData.size() * sizeof(unsigned int), bitstreamData.data(), GL_STATIC_READ);

	glGenBuffers(1, &buf_Pattern);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_Pattern);
	glBufferData(GL_SHADER_STORAGE_BUFFER, patternData.size() * sizeof(float), patternData.data(), GL_STATIC_READ);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // 바인딩 해제

	// (8) ★★★ [신규] DWT 셰이더 4개 로드 ★★★
	GLuint dwtPass1Shader = loadComputeShader("DWT/dwt_pass1_rows.comp");
	GLuint dwtPass2Shader = loadComputeShader("DWT/dwt_pass2_cols_embed.comp");
	GLuint dwtPass3Shader = loadComputeShader("DWT/idwt_pass1_cols.comp");
	GLuint dwtPass4Shader = loadComputeShader("DWT/idwt_pass2_rows.comp");

	// 셰이더 로드 실패 시 프로그램 종료
	if (dwtPass1Shader == 0 || dwtPass2Shader == 0 || dwtPass3Shader == 0 || dwtPass4Shader == 0)
	{
		glfwTerminate();
		return -1;
	}

	// (9) ★★★ [신규] DWT 텍스처 3개 생성 ★★★
	GLuint tex_DWT_Intermediate, tex_DWT_Output, tex_DWT_Final;

	glGenTextures(1, &tex_DWT_Intermediate);
	glBindTexture(GL_TEXTURE_2D, tex_DWT_Intermediate);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &tex_DWT_Output);
	glBindTexture(GL_TEXTURE_2D, tex_DWT_Output);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &tex_DWT_Final);
	glBindTexture(GL_TEXTURE_2D, tex_DWT_Final);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBindTexture(GL_TEXTURE_2D, 0);

	// (11) ★★★ [신규] SVD 셰이더 8개 로드 ★★★
	GLuint svd_prog1 = loadComputeShader("SVD/svd_01_rgb_to_y.comp");
	GLuint svd_prog2 = loadComputeShader("SVD/svd_02_store_cbcr.comp");
	GLuint svd_prog3 = loadComputeShader("SVD/svd_03_compute_ata.comp");
	GLuint svd_prog4 = loadComputeShader("SVD/svd_04_eigendecomposition.comp");
	GLuint svd_prog5 = loadComputeShader("SVD/svd_05_compute_u.comp");
	GLuint svd_prog6 = loadComputeShader("SVD/svd_06_modify_sigma.comp");
	GLuint svd_prog7 = loadComputeShader("SVD/svd_07_reconstruct_y.comp");
	GLuint svd_prog8 = loadComputeShader("SVD/svd_08_combine_ycbcr.comp");

	// (12) ★★★ [신규] SVD 전용 텍스처 8개 생성 ★★★
	GLuint tex_SVD_Y, tex_SVD_CbCr, tex_SVD_AtA, tex_SVD_V, tex_SVD_Sigma, tex_SVD_U, tex_SVD_ReconY, tex_SVD_Final;

	// (Y, AtA, V, Sigma, U, ReconY는 1채널)
	auto createTextureR32F = [&](GLuint& tex) {
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		};
	createTextureR32F(tex_SVD_Y);
	createTextureR32F(tex_SVD_AtA);
	createTextureR32F(tex_SVD_V);
	createTextureR32F(tex_SVD_Sigma);
	createTextureR32F(tex_SVD_U);
	createTextureR32F(tex_SVD_ReconY);

	// (CbCr은 2채널)
	glGenTextures(1, &tex_SVD_CbCr);
	glBindTexture(GL_TEXTURE_2D, tex_SVD_CbCr);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// (Final은 4채널)
	glGenTextures(1, &tex_SVD_Final);
	glBindTexture(GL_TEXTURE_2D, tex_SVD_Final);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBindTexture(GL_TEXTURE_2D, 0);

	// ★★★ ImGui 파라미터 변수 선언 ★★★
	bool g_EnableEmbed = true;
	float g_EmbeddingStrength = 0.1f;
	float g_CompressionThreshold = 0.0f; // ★ SVD 압축용 '별도' 변수

	// ★★★ SVD 파라미터 변수 ★★★
	int g_JacobiIterations = 10; // 야코비 반복 횟수
	float g_SigmaThreshold = 1.0e-7f;
	float g_ModificationValue = 0.1f; // 예: 특이값 압축 임계값

	// ★★★ DCT/DWT 선택기 변수 ★★★
	int g_AlgorithmChoice = 0; // // 0=DCT, 1=DWT, 2=SVD

	// --- 벤치마크용 변수 ---
	bool g_RunBenchmark = false;
	int g_BenchmarkIteration = 0;
	const int g_TotalIterations = 10;

	// 결과 저장용
	struct BenchmarkResult {
		std::string algoName;
		float gpuTimeMs;
		double psnr;
	};
	std::vector<BenchmarkResult> g_Results;

	GpuTimer gpuTimer;
	gpuTimer.Init();

	while (!glfwWindowShouldClose(window))
	{
		// Input
		// glfwGetKey...
		// Get event(Keyboard, Mouse input, window closing etc.) from OS
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Parameter Controller");
		{
			ImGui::SliderFloat("My Param", &myParam, 0.0f, 1.0f);
			ImGui::ColorEdit3("Color", color);

			ImGui::Text("My Param: %.3f", myParam);
			ImGui::Text("FPS: %.1f", io.Framerate);

			// ★★★ 알고리즘 선택 (SVD 추가) ★★★
			ImGui::RadioButton("DCT", &g_AlgorithmChoice, 0); ImGui::SameLine();
			ImGui::RadioButton("DWT", &g_AlgorithmChoice, 1); ImGui::SameLine();
			ImGui::RadioButton("SVD", &g_AlgorithmChoice, 2);
			ImGui::Separator();

			// ★★★ 파라미터 컨트롤 추가 ★★★
			ImGui::Checkbox("Activate Watermark (Embed)", &g_EnableEmbed);
			ImGui::SliderFloat("Strength", &g_EmbeddingStrength, 0.0f, 1.0f);
			ImGui::Text("CoefficientsToUse: %u", coeffsToUse); // (상수 값 표시)
			
			// --- 2. 'SVD 전용' 섹션 ---
			if (g_AlgorithmChoice == 2)
			{
				ImGui::Separator();
				ImGui::Text("[SVD-Specific Parameters]");
				ImGui::InputInt("Jacobi Repeat", &g_JacobiIterations);
				ImGui::SliderFloat("Sigma Threshold (Detect)", &g_SigmaThreshold, 0.00001f, 1.0f, "%.7f");

				// ★ [수정] '압축' 슬라이더를 '별도'로 분리
				ImGui::SliderFloat("Compression Threshold", &g_CompressionThreshold, 0.0f, 1.0f);
			}

			ImGui::Separator();
			ImGui::Text("Research Tools");

			// ★ 벤치마크 시작 버튼
			if (ImGui::Button("Run Benchmark (10x)"))
			{
				g_RunBenchmark = true;
				g_BenchmarkIteration = 0;
				g_Results.clear();
			}

			// 진행 상황 표시
			if (g_RunBenchmark)
			{
				ImGui::Text("Running... Iteration %d / %d", g_BenchmarkIteration + 1, g_TotalIterations);
			}

			// 결과 요약 표시 (최근 결과)
			if (!g_Results.empty())
			{
				ImGui::Text("Last Result (Avg Time):");
				// (평균 계산 로직은 생략, 마지막 값만 표시 예시)
				for (const auto& res : g_Results) {
					// 10번 반복된 걸 다 보여주긴 많으니, 요약해서 보여주는 게 좋음
					// 여기선 그냥 리스트업
				}
			}
		}

		ImGui::End();

		// ★★★ '원본' 이미지를 띄울 ImGui 창 ★★★
		ImGui::Begin("Source Image (Input)");
		// (tex_Source의 GLuint ID를 ImGui가 쓸 수 있는 포인터로 변환)
		ImGui::Image(tex_Source, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		// ★★★ DWT/DCT 결과창 분리 ★★★
		ImGui::Begin("DCT Result");
		ImGui::Image(tex_Final, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("DWT Result");
		ImGui::Image(tex_DWT_Final, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		// ★★★ SVD 결과창 추가 ★★★

		/*
		ImGui::Begin("SVD Pass 1 (Y)");
		ImGui::Image(tex_SVD_Y, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("SVD Pass 2 (CbCr)");
		ImGui::Image(tex_SVD_CbCr, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("SVD Pass 3 (A^T A)");
		ImGui::Image(tex_SVD_AtA, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("SVD Pass 4 (V)");
		ImGui::Image(tex_SVD_V, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("SVD Pass 5 (Sigma)");
		ImGui::Image(tex_SVD_Sigma, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("SVD Pass 6 (U)");
		ImGui::Image(tex_SVD_U, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Begin("SVD Pass 7 (ReconY)");
		ImGui::Image(tex_SVD_ReconY, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();
		*/

		ImGui::Begin("SVD Result");
		ImGui::Image(tex_SVD_Final, ImVec2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
		ImGui::End();

		ImGui::Render();


		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		// Render Color
		glClearColor(color[0], color[1], color[2], 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// --- ★★★ 컴퓨트 셰이더 실행 (Dispatch) ★★★ ---
		// --- ★★★ '선택된' 파이프라인 실행 ★★★ ---
		if (g_AlgorithmChoice == 0) // DCT
		{
			Run_Compute_Pipeline(
				dctPass1Shader, dctPass2Shader, dctPass3Shader, dctPass4Shader,
				tex_Source, tex_Intermediate, tex_DCTOutput, tex_Final, // 텍스처
				buf_Bitstream, buf_Pattern,
				TEXTURE_WIDTH, TEXTURE_HEIGHT,
				g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, numBlocks, numBlocks
			);
		}
		else if (g_AlgorithmChoice == 1)// DWT
		{
			Run_Compute_Pipeline(
				dwtPass1Shader, dwtPass2Shader, dwtPass3Shader, dwtPass4Shader,
				tex_Source, tex_DWT_Intermediate, tex_DWT_Output, tex_DWT_Final, // DWT 텍스처
				buf_Bitstream, buf_Pattern,
				TEXTURE_WIDTH, TEXTURE_HEIGHT,
				g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, numBlocks, numBlocks
			);
		}
		else // SVD
		{
			// C++ 배열에 셰이더/텍스처 ID를 '순서대로' 담아서 전달
			GLuint svd_progs[] = { svd_prog1, svd_prog2, svd_prog3, svd_prog4, svd_prog5, svd_prog6, svd_prog7, svd_prog8 };
			GLuint svd_texs[] = { tex_Source, tex_SVD_Y, tex_SVD_CbCr, tex_SVD_AtA, tex_SVD_V, tex_SVD_Sigma, tex_SVD_U, tex_SVD_ReconY, tex_SVD_Final };

			Run_SVD_Pipeline(
				svd_progs, svd_texs,
				buf_Bitstream, buf_Pattern,
				TEXTURE_WIDTH, TEXTURE_HEIGHT,
				g_JacobiIterations, g_SigmaThreshold,
				// ★ [수정] 'modValue' 1개 대신, '3개'의 파라미터를 받도록 변경
				g_EnableEmbed,           // 워터마킹 켤지 말지
				g_EmbeddingStrength,     // 워터마킹 강도
				g_CompressionThreshold,  // 압축 임계값

				// ★ (SSBO에 필요한 파라미터 추가)
				coeffsToUse, numBlocks, numBlocks);
		}
		// --- 컴퓨트 셰이더 실행 끝 ---

		// ---------------------------------------------------------
	// ★★★ 자동 벤치마크 로직 (상태 머신) ★★★
	// ---------------------------------------------------------
		if (g_RunBenchmark)
		{
			std::string algoNames[] = { "DCT", "DWT", "SVD" };

			// 현재 반복 회차에 대해 3가지 알고리즘을 모두 실행
			for (int algo = 0; algo < 3; ++algo)
			{
				// 1. GPU 타이머 시작
				gpuTimer.Start();

				// 2. 알고리즘 실행 (디스패치)
				if (algo == 0) {
					Run_Compute_Pipeline(
						dctPass1Shader, dctPass2Shader, dctPass3Shader, dctPass4Shader,
						tex_Source, tex_Intermediate, tex_DCTOutput, tex_Final, // 텍스처
						buf_Bitstream, buf_Pattern,
						TEXTURE_WIDTH, TEXTURE_HEIGHT,
						g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, numBlocks, numBlocks
					);
				}
				else if (algo == 1) {
					Run_Compute_Pipeline(
						dwtPass1Shader, dwtPass2Shader, dwtPass3Shader, dwtPass4Shader,
						tex_Source, tex_DWT_Intermediate, tex_DWT_Output, tex_DWT_Final, // DWT 텍스처
						buf_Bitstream, buf_Pattern,
						TEXTURE_WIDTH, TEXTURE_HEIGHT,
						g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, numBlocks, numBlocks
					);
				}
				else { // SVD
					// C++ 배열에 셰이더/텍스처 ID를 '순서대로' 담아서 전달
					GLuint svd_progs[] = { svd_prog1, svd_prog2, svd_prog3, svd_prog4, svd_prog5, svd_prog6, svd_prog7, svd_prog8 };
					GLuint svd_texs[] = { tex_Source, tex_SVD_Y, tex_SVD_CbCr, tex_SVD_AtA, tex_SVD_V, tex_SVD_Sigma, tex_SVD_U, tex_SVD_ReconY, tex_SVD_Final };

					Run_SVD_Pipeline(
						svd_progs, svd_texs,
						buf_Bitstream, buf_Pattern,
						TEXTURE_WIDTH, TEXTURE_HEIGHT,
						g_JacobiIterations, g_SigmaThreshold,
						// ★ [수정] 'modValue' 1개 대신, '3개'의 파라미터를 받도록 변경
						g_EnableEmbed,           // 워터마킹 켤지 말지
						g_EmbeddingStrength,     // 워터마킹 강도
						g_CompressionThreshold,  // 압축 임계값

						// ★ (SSBO에 필요한 파라미터 추가)
						coeffsToUse, numBlocks, numBlocks);
				}

				// 3. GPU 타이머 종료 & 시간 측정
				gpuTimer.Stop();
				float timeMs = gpuTimer.GetTimeMs();

				// 4. 결과 텍스처 다운로드 & 저장 & 화질 측정
				// (첫 번째 반복 때만 이미지를 저장하여 디스크 용량 절약)
				double psnr = 0.0;
				if (g_BenchmarkIteration == 0)
				{
					// 결과 텍스처 ID 가져오기
					GLuint resultTex = (algo == 0) ? tex_Final :
						(algo == 1) ? tex_DWT_Final : tex_SVD_Final;

					// 파일명 생성 (예: Result_DCT_Run0.png)
					std::string filename = "Result_" + algoNames[algo] + ".png";
					SaveTextureToPNG(resultTex, RENDER_WIDTH, RENDER_HEIGHT, filename);

					// PSNR 계산 (CPU에서 하므로 약간 느림)
					// (원본과 결과 텍스처를 RAM으로 가져와서 비교)
					std::vector<unsigned char> srcPx(RENDER_WIDTH * RENDER_HEIGHT * 4);
					std::vector<unsigned char> dstPx(RENDER_WIDTH * RENDER_HEIGHT * 4);

					glBindTexture(GL_TEXTURE_2D, tex_Source);
					glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, srcPx.data());

					glBindTexture(GL_TEXTURE_2D, resultTex);
					glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, dstPx.data());
					glBindTexture(GL_TEXTURE_2D, 0);

					ImageMetrics m = CalculateMetrics(srcPx, dstPx, RENDER_WIDTH, RENDER_HEIGHT);
					psnr = m.psnr;
				}

				// 결과 저장
				g_Results.push_back({ algoNames[algo], timeMs, psnr });
			}

			g_BenchmarkIteration++;

			// 종료 조건
			if (g_BenchmarkIteration >= g_TotalIterations)
			{
				g_RunBenchmark = false;
				g_BenchmarkIteration = 0;
				std::cout << "=== 벤치마크 완료 (" << g_TotalIterations << "회) ===" << std::endl;
				// (여기서 g_Results를 파일로 저장하거나 ImGui에 띄우면 됩니다)
			}
		}

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		
		// Buffer Swap
		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}

