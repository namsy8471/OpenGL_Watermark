// =================================================================================
// GPU-Accelerated Watermarking Benchmark System (DCT / DWT / SVD / DFT + SSIM)
// =================================================================================

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <iomanip>

using uint = unsigned int;

// -----------------------------------------------------------------------------
// 1. 유틸리티 및 수학 헬퍼
// -----------------------------------------------------------------------------

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start;
    void Start() { start = std::chrono::high_resolution_clock::now(); }
    float GetTimeMs() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        return duration.count();
    }
};

struct GpuTimer {
    GLuint queryID = 0;
    void Init() { glGenQueries(1, &queryID); }
    void Start() { glBeginQuery(GL_TIME_ELAPSED, queryID); }
    void Stop() { glEndQuery(GL_TIME_ELAPSED); }
    float GetTimeMs() {
        GLuint64 timeNs = 0;
        glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeNs);
        return static_cast<float>(timeNs) / 1000000.0f;
    }
};

struct ImageMetrics {
    double psnr;
    double ssim;
};

// [New] 공격 유형 정의
enum class AttackType {
    NONE = 0,
    CROP_CENTER,    // 중앙 크롭 (비율)
    RESIZE_SCALE,   // 리사이즈 (비율)
    NOISE_SNP,      // 소금-후추 노이즈 (강도)
    QUANTIZATION,   // 색상 양자화 (단계)
    GAMMA_CORRECT,  // 감마 보정 (값)
    BRIGHTNESS,      // 밝기 조절 (값)
	ROTATION	   // 회전 (각도)
};

// [New] 공격 설정 구조체
struct AttackConfig {
    AttackType type;
    float param;      // 공격 강도 (예: 0.5)
    std::string name; // CSV 출력용 이름 (예: "Crop_50%")
};

// [New] 공격 결과 저장 구조체
struct AttackResult {
    std::string algoName;
    std::string attackName;
    float param;
    float ber;
    float recoveredBER;
    double psnr; // 공격받은 이미지의 화질
};

// 결과 저장 구조체 (SSIM 필드 추가)
struct BenchmarkResult {
    std::string algoName;
    std::string resolution;
    double alpha;
    float gpuTimeMs;
    float cpuTimeMs;
    double psnr;
    double ssim; // Added
    float ber;
};

// [수정된 함수] 유효 비트 수(count)만큼만 비교
float CalculateBER(const std::vector<uint>& original, const std::vector<uint>& extracted, size_t count) {
    if (original.empty() || extracted.empty()) return 1.0f;

    size_t errorCount = 0;
    // 범위 초과 방지
    size_t validCount = std::min({ count, original.size(), extracted.size() });

    for (size_t i = 0; i < validCount; ++i) {
        uint orgBit = (original[i] > 0) ? 1 : 0;
        uint extBit = (extracted[i] > 0) ? 1 : 0;

        if (orgBit != extBit) {
            errorCount++;
        }
    }
    return static_cast<float>(errorCount) / static_cast<float>(validCount);
}

// [Senior Guide] SSIM 계산 함수
// 메모리 접근 패턴(Cache Locality)을 고려해야 하지만, 이미지 전체를 스캔해야 하므로
// CPU 캐시 미스는 필연적입니다. 정확한 계산을 위해 가우시안 윈도우 대신 
// 8x8 윈도우 평균을 사용하는 간소화된 버전을 구현합니다.
double CalculateSSIM_Core(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, int width, int height)
{
    const double C1 = 6.5025;  // (0.01 * 255)^2
    const double C2 = 58.5225; // (0.03 * 255)^2

    double ssim_sum = 0.0;
    int win_size = 8; // Block size for localized statistics
    int num_windows = 0;

    // Loop with step for performance (approximate) or step=1 for accuracy
    // 4K 해상도에서 모든 픽셀을 슬라이딩 윈도우로 계산하면 너무 느리므로, 
    // 블록 단위(Non-overlapping)로 처리하여 근사치를 구합니다.
    for (int y = 0; y < height; y += win_size) {
        for (int x = 0; x < width; x += win_size) {
            double mu1 = 0.0, mu2 = 0.0;
            double sigma1_sq = 0.0, sigma2_sq = 0.0, sigma12 = 0.0;

            int count = 0;
            // 윈도우 내부 순회
            for (int dy = 0; dy < win_size; ++dy) {
                for (int dx = 0; dx < win_size; ++dx) {
                    if (y + dy >= height || x + dx >= width) continue;

                    size_t idx = ((y + dy) * width + (x + dx)) * 4; // RGBA
                    // Luminance 변환: 0.299R + 0.587G + 0.114B (간소화를 위해 G채널만 사용하거나 평균 사용)
                    // 여기서는 RGB 평균을 사용합니다.
                    double val1 = (img1[idx] + img1[idx + 1] + img1[idx + 2]) / 3.0;
                    double val2 = (img2[idx] + img2[idx + 1] + img2[idx + 2]) / 3.0;

                    mu1 += val1;
                    mu2 += val2;

                    // 분산을 나중에 구하기 위해 값 저장 필요없음 (One-pass 알고리즘은 아니지만 단순화)
                    // 정확한 분산을 위해 Two-pass가 필요하지만, 여기선 제곱의 합을 누적
                    sigma1_sq += val1 * val1;
                    sigma2_sq += val2 * val2;
                    sigma12 += val1 * val2;

                    count++;
                }
            }

            if (count == 0) continue;

            mu1 /= count;
            mu2 /= count;

            // E[X^2] - (E[X])^2 공식을 이용한 분산 계산
            sigma1_sq = (sigma1_sq / count) - (mu1 * mu1);
            sigma2_sq = (sigma2_sq / count) - (mu2 * mu2);
            sigma12 = (sigma12 / count) - (mu1 * mu2);

            double ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) /
                ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2));

            ssim_sum += ssim_map;
            num_windows++;
        }
    }

    return (num_windows > 0) ? (ssim_sum / num_windows) : 0.0;
}

ImageMetrics CalculateMetrics(const std::vector<unsigned char>& original,
    const std::vector<unsigned char>& target,
    int width, int height)
{
    // 1. PSNR Calculation
    double mse = 0.0;
    size_t totalPixels = static_cast<size_t>(width) * height;

    // [Optimization] OpenMP를 쓰면 좋겠지만, 표준 C++로 진행합니다.
    // 메모리 접근의 지역성을 위해 채널별로 루프를 돌지 않고 픽셀 단위로 돕니다.
    for (size_t i = 0; i < totalPixels * 4; ++i) {
        double diff = static_cast<double>(original[i]) - static_cast<double>(target[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(totalPixels * 4);
    double psnr = (mse < 1e-10) ? 100.0 : (10.0 * log10((255.0 * 255.0) / mse));

    // 2. SSIM Calculation (추가됨)
    double ssim = CalculateSSIM_Core(original, target, width, height);

    return { psnr, ssim };
}

void SaveTextureToPNG(GLuint textureID, int width, int height, std::string filename) {
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4);
}

// 셰이더 로더 (기존 유지)
std::string loadShaderSourceFromFile(const char* shaderFilePath) {
    std::ifstream shaderFile;

    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        shaderFile.open("src/Shaders/" + std::string(shaderFilePath));
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        return shaderStream.str();
    }
    catch (...) {
        // 파일이 없으면 에러 로그만 찍고 빈 문자열 반환
        std::cerr << "Warning: Could not load shader " << shaderFilePath << std::endl; 
        return "";
    }
}

GLuint loadComputeShader(const char* computePath) {
    std::string codeStr = loadShaderSourceFromFile(computePath);
    if (codeStr.empty()) return 0;
    const char* code = codeStr.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &code, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "COMPILE ERROR: " << computePath << "\n" << infoLog << std::endl;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(program, 1024, NULL, infoLog);
        std::cerr << "LINK ERROR: " << computePath << "\n" << infoLog << std::endl;
        return 0;
    }
    glDeleteShader(shader);
    return program;
}


void SaveResultsToCSV(const std::vector<BenchmarkResult>& results, const char* filename)
{
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(3);
    file << "Iteration,Algorithm,Resolution,Alpha,GPU Time (ms),CPU Time (ms),PSNR (dB),SSIM,BER\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        // 알고리즘 4개 (DCT, DWT, SVD, DFT)
        int iter = static_cast<int>(i) + 1;
        file << iter << "," 
    		<< res.algoName << ","
    		<< res.resolution << ","
			<< res.alpha << ","
            << res.gpuTimeMs << "," 
    		<< res.cpuTimeMs << ","
    		<< res.psnr << ","
    		<< res.ssim << ","
    		<< res.ber << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

// [New] 다수결 투표를 통한 서명 복구 및 BER 계산
// extracted: 전체 이미지에서 추출된 비트 스트림 (수만 개)
// originalSignature: 원본 서명 (64개)
// msgLen: 서명 길이 (64)
float CalculateRecoveredBER(const std::vector<uint>& extracted, const std::vector<uint>& originalSignature, int msgLen) {
    if (extracted.empty() || originalSignature.empty()) return 1.0f;

    // 1. 투표함 초기화 (0에 투표한 수, 1에 투표한 수)
    std::vector<int> vote0(msgLen, 0);
    std::vector<int> vote1(msgLen, 0);

    // 2. 전체 블록을 순회하며 투표
    for (size_t i = 0; i < extracted.size(); ++i) {
        int signatureIdx = i % msgLen; // 서명의 몇 번째 비트인지 확인

        if (extracted[i] == 0) vote0[signatureIdx]++;
        else                   vote1[signatureIdx]++;
    }

    // 3. 개표 (다수결로 복구)
    std::vector<uint> recoveredSignature(msgLen);
    for (int i = 0; i < msgLen; ++i) {
        // 1이 더 많으면 1, 아니면 0 (동점이면 0 처리)
        if (vote1[i] > vote0[i]) recoveredSignature[i] = 1;
        else                     recoveredSignature[i] = 0;
    }

    // 4. 복구된 서명과 원본 서명 비교 (진짜 BER)
    int errorCount = 0;
    for (int i = 0; i < msgLen; ++i) {
        if (recoveredSignature[i] != originalSignature[i]) {
            errorCount++;
        }
    }

    return (float)errorCount / (float)msgLen;
}

// C++ 구현부

// n이 2의 제곱수인지 확인하는 유틸
bool IsPowerOfTwo(uint n) { return (n & (n - 1)) == 0; }

// ------------------------------------------------------------------
// [Helper] 2의 제곱수 계산 (Bit Twiddling)
// ------------------------------------------------------------------
uint NextPowerOfTwo(uint v) {
    v--;
    v |= v >> 1; 
	v |= v >> 2; 
	v |= v >> 4; 
	v |= v >> 8; 
	v |= v >> 16;
    v++;
    return v;
}

// ------------------------------------------------------------------
// [Core] 단일 방향 FFT 패스 (로그 N 단계 수행)
// ------------------------------------------------------------------
void Run_FFT_Pass(GLuint progReorder, GLuint progFFT, GLuint texIn, GLuint texOut, uint size, int direction, int inverse)
{
    // 1. Shift 값 계산 (비트 리버설을 위한 핵심 상수)
    // 2048(2^11) -> 32비트 전체 뒤집기 -> (32-11)=21만큼 우측 시프트해야 0~2047 범위에 들어옴
    int stages = 0;
    int tempSize = size;
    while (tempSize > 1) { tempSize >>= 1; stages++; }
    int shift = 32 - stages;

    // 2. [줄 세우기] Reorder (이게 빠지면 격자무늬 나옴!)
    glUseProgram(progReorder);
    glUniform1i(glGetUniformLocation(progReorder, "u_Size"), size);
    glUniform1i(glGetUniformLocation(progReorder, "u_Mode"), direction);
    glUniform1i(glGetUniformLocation(progReorder, "u_Shift"), shift); // ★ 중요

    // 입력(texIn)을 정렬해서 -> 출력(texOut)에 씀
    glBindImageTexture(0, texIn, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, texOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);

    glDispatchCompute((size + 31) / 32, (size + 31) / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // 핑퐁용 임시 ID (초기 상태)
    // 입력(texIn) -> Ping -> Pong -> Ping ... -> 출력(texOut)
    // 주의: 첫 단계는 texIn에서 읽어야 함.
    // 간단한 구현을 위해, texIn 내용을 먼저 texPing(texOut)에 복사해두고 시작하거나,
    // 쉐이더 내부에서 Ping/Pong 바인딩을 교체해야 함.

    // 여기서는 "Input Texture"와 "Output Texture"를 핑퐁 버퍼로 간주하고 교차 바인딩합니다.
    // 첫 스테이지: Read(texIn), Write(texOut) -> 이렇게 하면 texOut이 Ping 역할
    // 하지만 이러면 texIn이 손상될 수 없으므로, texIn과 texOut 외에 별도의 Temp 텍스처가 필요할 수 있음.
    // ★ 가장 쉬운 방법: texIn을 texOut(Ping)에 복사 후, texOut과 texPong을 핑퐁.

    // (간소화된 로직: 외부에서 이미 texIn이 Padding되어 Ping에 들어있다고 가정)
    // 즉, 인자로 들어오는 texIn과 texOut을 핑퐁 버퍼 두 개라고 생각합니다.

    GLuint bufRead = texOut;
    GLuint bufWrite = texIn;

    glUseProgram(progFFT);
    glUniform1i(glGetUniformLocation(progFFT, "u_PassID"), direction); // 0:Horiz, 1:Vert
    glUniform1i(glGetUniformLocation(progFFT, "u_Inverse"), inverse);
    glUniform1i(glGetUniformLocation(progFFT, "u_Size"), size);
    

    for (int i = 0; i < stages; ++i) {
        glUniform1i(glGetUniformLocation(progFFT, "u_Stage"), i);

        // 바인딩
        glBindImageTexture(0, bufRead, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
        glBindImageTexture(1, bufWrite, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);

        // Dispatch (32x32 WorkGroup)
        glDispatchCompute((size + 31) / 32, (size + 31) / 32, 1);

        // 배리어 (다음 스테이지가 앞 스테이지 결과를 읽어야 함)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Swap
        std::swap(bufRead, bufWrite);
    }

    // 만약 루프가 끝났는데 결과가 texIn 쪽에 가 있다면?
    // stages가 짝수면 bufRead가 texIn으로 돌아옴. 
    // 이 경우 최종 결과를 texOut으로 한 번 더 복사해줘야 하지만, 
    // 보통 2048(11단계, 홀수)이면 Swap으로 인해 자동으로 bufWrite가 texOut이 됩니다.
    // (11번 Swap -> Read/Write가 뒤집힘 -> 최종 결과는 bufWrite에 있음)

    // ※ 안전장치: 최종 결과가 담긴 텍스처 ID를 반환하거나 복사해야 하지만,
    // 여기서는 호출자가 PING/PONG을 관리하도록 설계합니다.

    // 4. 결과 위치 보정
    // (2048은 11단계(홀수)라서 FFT 끝나면 bufRead(결과)가 texIn에 들어있음)
    // 우리는 texOut에 결과가 있기를 기대하므로 복사해줌.
    if (bufRead != texOut) {
        glCopyImageSubData(bufRead, GL_TEXTURE_2D, 0, 0, 0, 0,
            texOut, GL_TEXTURE_2D, 0, 0, 0, 0,
            size, size, 1);
    }

}

// ------------------------------------------------------------------
// [Master] 전체 파이프라인
// ------------------------------------------------------------------
void Run_Full_FFT_Pipeline(
	GLuint progPad, GLuint progReorder, GLuint progFFT, GLuint progEmbed, GLuint progCrop, GLuint debugProg,
    GLuint texSrc, GLuint texFinal,
    GLuint texPing, GLuint texPong, // 2048x2048 RG32F
    GLuint bufBits,
    uint width, uint height,
    bool enableEmbed, float strength)
{
    // 1. 패딩 사이즈 계산
    uint paddedSize = std::max(NextPowerOfTwo(width), NextPowerOfTwo(height));

    // -------------------------------------------------------
    // Step 1: Padding (Src -> Ping)
    // -------------------------------------------------------
    glUseProgram(progPad);
    glUniform1ui(glGetUniformLocation(progPad, "SrcWidth"), width);
    glUniform1ui(glGetUniformLocation(progPad, "SrcHeight"), height);
    glUniform1ui(glGetUniformLocation(progPad, "PaddedSize"), paddedSize);

    glBindImageTexture(0, texSrc, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texPing, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F); // Ping에 초기화

    glDispatchCompute((paddedSize + 31) / 32, (paddedSize + 31) / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // -------------------------------------------------------
    // Step 2: FFT Horizontal (Ping <-> Pong)
    // 최종 결과가 Pong에 남도록 설계 (11단계면 Ping->Pong 끝)
    // -------------------------------------------------------
    // 주의: Run_FFT_Pass 내부에서 핑퐁하므로, 입력으로 Ping, Pong을 주면 
    // 서로 주고받다가 마지막에 적절한 곳에 멈춤.
    // 2048(11단계-홀수) 기준: Ping(In) -> ... -> Pong(Out)
    Run_FFT_Pass(progReorder, progFFT, texPing, texPong, paddedSize, 0, 0);

    // 현재 데이터 위치: Pong (Horizontal 완료됨)

    // -------------------------------------------------------
    // Step 3: FFT Vertical (Pong <-> Ping)
    // -------------------------------------------------------
    Run_FFT_Pass(progReorder, progFFT, texPong, texPing, paddedSize, 1, 0);

    // 현재 데이터 위치: Ping (Full Frequency Domain)

    // -------------------------------------------------------
    // Step 4: Embedding (Ping Modify)
    // -------------------------------------------------------
    glUseProgram(progEmbed);
    glUniform1ui(glGetUniformLocation(progEmbed, "PaddedSize"), paddedSize);
    glUniform1ui(glGetUniformLocation(progEmbed, "Embed"), enableEmbed ? 1 : 0);
    glUniform1f(glGetUniformLocation(progEmbed, "Strength"), strength);

    glBindImageTexture(0, texPing, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F); // In-Place
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufBits);

    glDispatchCompute((paddedSize + 31) / 32, (paddedSize + 31) / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // -------------------------------------------------------
    // Step 5: IFFT Vertical (Ping <-> Pong)
    // -------------------------------------------------------
    Run_FFT_Pass(progReorder, progFFT, texPing, texPong, paddedSize, 1, 1); // Inverse Flag = 1

    // 현재 데이터 위치: Pong

    // -------------------------------------------------------
    // Step 6: IFFT Horizontal (Pong <-> Ping)
    // -------------------------------------------------------
    Run_FFT_Pass(progReorder, progFFT, texPong, texPing, paddedSize, 0, 1);

    // 현재 데이터 위치: Ping (최종 복원되었으나 Padded & Scaled 상태)

    // -------------------------------------------------------
    // Step 7: Crop & Normalize (Ping -> Final)
    // -------------------------------------------------------
    glUseProgram(progCrop);
    glUniform1ui(glGetUniformLocation(progCrop, "SrcWidth"), width);
    glUniform1ui(glGetUniformLocation(progCrop, "SrcHeight"), height);
    glUniform1ui(glGetUniformLocation(progCrop, "PaddedSize"), paddedSize);

    glBindImageTexture(0, texPing, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, texFinal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(2, texSrc, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

    glDispatchCompute((width + 31) / 32, (height + 31) / 32, 1);

    // 최종 배리어 (UI 렌더링 위해)
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// -----------------------------------------------------------------------------
// 2. 파이프라인 실행 함수들
// -----------------------------------------------------------------------------

void Run_Compute_Pipeline(
    GLuint pass1, GLuint pass2, GLuint pass3, GLuint pass4,
    GLuint texSrc, GLuint texInter, GLuint texOut, GLuint texFinal,
    GLuint bufBits, GLuint bufPattern,
    uint width, uint height,
    bool enableEmbed, float strength, uint coeffsToUse, uint numBlocks, uint bitLength)
{
    // (기존 구현 유지)
    const uint numGroupsX = (width + 31) / 32; // Workgroup size optimization (assuming 32x32 local size)
    const uint numGroupsY = (height + 31) / 32;
    // ... (생략된 코드는 질문의 코드와 동일하다고 가정하고 실행)

    // Note: 실제 구현시 Dispatch 파라미터는 LocalGroupSize에 맞춰야 합니다.
    // 여기서는 편의상 8x8로 가정합니다.
    const uint nx = (width + 7) / 8;
    const uint ny = (height + 7) / 8;

    glUseProgram(pass1);
    glUniform1ui(glGetUniformLocation(pass1, "Width"), width);
    glUniform1ui(glGetUniformLocation(pass1, "Height"), height);
    glBindImageTexture(0, texSrc, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texInter, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(nx, ny, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(pass2);
    glUniform1ui(glGetUniformLocation(pass2, "Width"), width);
    glUniform1ui(glGetUniformLocation(pass2, "Height"), height);
    glUniform1ui(glGetUniformLocation(pass2, "Embed"), enableEmbed ? 1 : 0);
    glUniform1f(glGetUniformLocation(pass2, "EmbeddingStrength"), strength);
    glUniform1ui(glGetUniformLocation(pass2, "CoefficientsToUse"), coeffsToUse);
    glUniform1ui(glGetUniformLocation(pass2, "BitLength"), bitLength);
    glBindImageTexture(0, texInter, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufBits);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufPattern);
    glDispatchCompute(nx, ny, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(pass3);
    glUniform1ui(glGetUniformLocation(pass3, "Width"), width);
    glUniform1ui(glGetUniformLocation(pass3, "Height"), height);
    glBindImageTexture(0, texOut, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texInter, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(nx, ny, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(pass4);
    glUniform1ui(glGetUniformLocation(pass4, "Width"), width);
    glUniform1ui(glGetUniformLocation(pass4, "Height"), height);
    glBindImageTexture(0, texInter, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texFinal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(nx, ny, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

void Run_SVD_Pipeline(
    GLuint progs[8], GLuint texs[9],
    GLuint bufBits, GLuint bufPattern,
    uint width, uint height,
    int jacobiIter, float sigmaThreshold,
    bool enableEmbed, float embeddingStrength, float compressionThreshold,
    uint coeffsToUse, uint numBlocks, uint bitLength)
{
    // (SVD 코드는 기존 질문의 코드 사용)
    const uint numGroupsX = (width + 7) / 8;
    const uint numGroupsY = (height + 7) / 8;

    // 1: RGB->Y

    glUseProgram(progs[0]);
    glUniform1ui(glGetUniformLocation(progs[0], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[0], "Height"), height);
    glBindImageTexture(0, texs[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texs[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);



    // 2: RGB->CbCr
    glUseProgram(progs[1]);
    glUniform1ui(glGetUniformLocation(progs[1], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[1], "Height"), height);
    glBindImageTexture(0, texs[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texs[2], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);



    // 3: AtA
    glUseProgram(progs[2]);
    glUniform1ui(glGetUniformLocation(progs[2], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[2], "Height"), height);
    glBindImageTexture(0, texs[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, texs[3], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);



    // 4: Eigen
    glUseProgram(progs[3]);
    glUniform1ui(glGetUniformLocation(progs[3], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[3], "Height"), height);
    glUniform1ui(glGetUniformLocation(progs[3], "JacobiIterations"), jacobiIter);
    glBindImageTexture(0, texs[3], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, texs[4], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // V
    glBindImageTexture(2, texs[5], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // Sigma
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


    // 5: Compute U
    glUseProgram(progs[4]);
    glUniform1ui(glGetUniformLocation(progs[4], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[4], "Height"), height);
    glUniform1f(glGetUniformLocation(progs[4], "SigmaThreshold"), sigmaThreshold);
    glBindImageTexture(0, texs[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, texs[4], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, texs[5], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, texs[6], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // U
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);



    // 6: Modify Sigma (Embed/Compress)
    glUseProgram(progs[5]);
    glUniform1ui(glGetUniformLocation(progs[5], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[5], "Height"), height);
    glUniform1f(glGetUniformLocation(progs[5], "ModificationValue"), compressionThreshold);
    glUniform1ui(glGetUniformLocation(progs[5], "Embed"), enableEmbed ? 1 : 0);
    glUniform1f(glGetUniformLocation(progs[5], "EmbeddingStrength"), embeddingStrength);
    glUniform1ui(glGetUniformLocation(progs[5], "CoefficientsToUse"), coeffsToUse);
    glUniform1ui(glGetUniformLocation(progs[5], "BitLength"), bitLength);
    glBindImageTexture(0, texs[5], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufBits);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufPattern);
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);



    // 7: Reconstruct Y
    glUseProgram(progs[6]);
    glUniform1ui(glGetUniformLocation(progs[6], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[6], "Height"), height);
    glBindImageTexture(0, texs[6], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, texs[4], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, texs[5], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, texs[7], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F); // ReconY
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);



    // 8: Combine
    glUseProgram(progs[7]);
    glUniform1ui(glGetUniformLocation(progs[7], "Width"), width);
    glUniform1ui(glGetUniformLocation(progs[7], "Height"), height);
    glBindImageTexture(0, texs[7], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, texs[2], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(2, texs[8], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F); // Final
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}


// [NEW] Optimized One-Shot Pipeline (DCT, DWT, SVD shared structure)
// 최적화된 셰이더는 4x4 블록 단위로 처리하며, 8x8 스레드 그룹(Local Size)을 사용합니다.
// 즉 1개의 스레드가 4x4 블록 1개를 담당합니다.
// 1 WorkGroup (8x8 threads) = 64 blocks = 32x32 pixels coverage.
void Run_Optimized_OneShot_Pipeline(
    GLuint prog,
    GLuint texSrc, GLuint texFinal,
    GLuint bufBits,
    uint width, uint height,
    bool enableEmbed, float strength)
{
    // Dispatch Size Calculation
    // WorkGroup covers 32x32 pixels (since 1 thread = 4x4 pixels, local_size = 8x8)
    const uint numGroupsX = (width + 31) / 32;
    const uint numGroupsY = (height + 31) / 32;

    glUseProgram(prog);
    glUniform1ui(glGetUniformLocation(prog, "Width"), width);
    glUniform1ui(glGetUniformLocation(prog, "Height"), height);
    glUniform1ui(glGetUniformLocation(prog, "Embed"), enableEmbed ? 1 : 0);
    glUniform1f(glGetUniformLocation(prog, "Strength"), strength);

    // Calculate total blocks for bitstream indexing
    uint totalBlocks = ((width + 3) / 4) * ((height + 3) / 4);
    glUniform1ui(glGetUniformLocation(prog, "BitSize"), totalBlocks);

    glBindImageTexture(0, texSrc, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texFinal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufBits);

    // One-Shot Dispatch
    glDispatchCompute(numGroupsX, numGroupsY, 1);

    // Barrier for ImGui Read
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

// -----------------------------------------------------------------------------
// ★ Extraction Pipeline (추출용)
// -----------------------------------------------------------------------------
// 공격받은 이미지(Marked)와 원본(Original)을 넣고 비트를 추출함
void Run_Extraction_Pipeline(GLuint progExtract, GLuint texOriginal, GLuint texMarked, GLuint bufExtracted, uint width, uint height, uint bitSize)
{
    // Dispatch Size: 4x4 Block Based -> 8x8 Thread Group -> 32x32 Pixels per Group
    const uint numGroupsX = (width + 31) / 32;
    const uint numGroupsY = (height + 31) / 32;

    glUseProgram(progExtract);
    glUniform1ui(glGetUniformLocation(progExtract, "Width"), width);
    glUniform1ui(glGetUniformLocation(progExtract, "Height"), height);
    glUniform1ui(glGetUniformLocation(progExtract, "BitSize"), bitSize);

    // Binding 0: Original, 1: Marked
    glBindImageTexture(0, texOriginal, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texMarked, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

    // Binding 2: Extracted Bits Buffer
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufExtracted);

    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT); // CPU Read를 위한 배리어
}

// Legacy (8x8 Block) 전용 추출 파이프라인
void Run_Legacy_Extraction(GLuint progExtract, GLuint texOriginal, GLuint texMarked, GLuint bufExtracted, uint width, uint height, uint bitSize)
{
    // Legacy는 8x8 픽셀이 1개 그룹
    const uint numGroupsX = (width + 7) / 8;
    const uint numGroupsY = (height + 7) / 8;

    glUseProgram(progExtract);
    glUniform1ui(glGetUniformLocation(progExtract, "Width"), width);
    glUniform1ui(glGetUniformLocation(progExtract, "Height"), height);
    glUniform1ui(glGetUniformLocation(progExtract, "BitSize"), bitSize);

    // Legacy 추출은 Spread Spectrum이라 'CoefficientsToUse'가 필요함 (기본 10 설정)
    glUniform1ui(glGetUniformLocation(progExtract, "CoefficientsToUse"), 10);

    glBindImageTexture(0, texOriginal, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texMarked, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufExtracted);
    // (패턴 버퍼 바인딩은 호출부에서 미리 해줌)

    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

// [Legacy DFT 전용 추출 파이프라인]
void Run_Full_FFT_Extraction(
    GLuint progPad, GLuint progReorder, GLuint progFFT, GLuint progExtractGlobal,
    GLuint texOrg, GLuint texMark, // 입력: 원본, 마킹 이미지 (Spatial)
    GLuint texPing, GLuint texPong, // 작업용: FFT 변환에 사용 (2048x2048 RG32F)
    GLuint texBackupFreq,           // 작업용: 원본의 주파수 데이터를 잠시 보관할 곳 (tex_DFT_Complex 사용)
    GLuint bufExtracted, uint width, uint height, uint bitSize)
{
    uint paddedSize = std::max(NextPowerOfTwo(width), NextPowerOfTwo(height));

    // ---------------------------------------------------------
    // 1. 원본(Original) 이미지 -> 주파수 변환
    // ---------------------------------------------------------
    // 1-1. Padding (Src -> Ping)
    glUseProgram(progPad);
    glUniform1ui(glGetUniformLocation(progPad, "SrcWidth"), width);
    glUniform1ui(glGetUniformLocation(progPad, "SrcHeight"), height);
    glUniform1ui(glGetUniformLocation(progPad, "PaddedSize"), paddedSize);
    glBindImageTexture(0, texOrg, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texPing, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute((paddedSize + 31) / 32, (paddedSize + 31) / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // 1-2. FFT (Ping -> Pong -> Ping)
    // Horizontal
    Run_FFT_Pass(progReorder, progFFT, texPing, texPong, paddedSize, 0, 0);
    // Vertical (결과는 Ping에 저장됨)
    Run_FFT_Pass(progReorder, progFFT, texPong, texPing, paddedSize, 1, 0);

    // 1-3. 원본 주파수 데이터 백업 (Ping -> Backup)
    // 마킹된 이미지 변환할 때 Ping을 또 써야 하므로, 원본 결과를 대피시킴
    glCopyImageSubData(texPing, GL_TEXTURE_2D, 0, 0, 0, 0,
        texBackupFreq, GL_TEXTURE_2D, 0, 0, 0, 0,
        paddedSize, paddedSize, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS); // 복사 완료 대기

    // ---------------------------------------------------------
    // 2. 마킹(Marked) 이미지 -> 주파수 변환
    // ---------------------------------------------------------
    // 2-1. Padding (Mark -> Ping)
    glUseProgram(progPad);
    glBindImageTexture(0, texMark, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texPing, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute((paddedSize + 31) / 32, (paddedSize + 31) / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // 2-2. FFT
    Run_FFT_Pass(progReorder, progFFT, texPing, texPong, paddedSize, 0, 0);
    Run_FFT_Pass(progReorder, progFFT, texPong, texPing, paddedSize, 1, 0);
    // 결과는 Ping에 있음 (Marked Frequency)

    // ---------------------------------------------------------
    // 3. 비교 및 추출 (Global Extract Shader)
    // ---------------------------------------------------------
    glUseProgram(progExtractGlobal);
    glUniform1ui(glGetUniformLocation(progExtractGlobal, "PaddedSize"), paddedSize);
    glUniform1ui(glGetUniformLocation(progExtractGlobal, "BitSize"), bitSize);

    // Binding 0: 원본 주파수 (백업해둔 것)
    glBindImageTexture(0, texBackupFreq, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    // Binding 1: 마킹 주파수 (방금 변환한 것)
    glBindImageTexture(1, texPing, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    // Binding 2: 결과 버퍼
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufExtracted);

    glDispatchCompute((paddedSize + 31) / 32, (paddedSize + 31) / 32, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

// [New] 공격 실행 헬퍼 함수
void Run_Attack_Shader(GLuint prog, GLuint texIn, GLuint texOut, int w, int h, float param, AttackType type) {
    glUseProgram(prog);

    // 공격 유형별 유니폼 설정 (셰이더 변수명에 맞춤)
    if (type == AttackType::CROP_CENTER) glUniform1f(glGetUniformLocation(prog, "Ratio"), param);
    else if (type == AttackType::RESIZE_SCALE) glUniform1f(glGetUniformLocation(prog, "Scale"), param);
    else if (type == AttackType::NOISE_SNP) glUniform1f(glGetUniformLocation(prog, "Strength"), param);
    else if (type == AttackType::QUANTIZATION) glUniform1f(glGetUniformLocation(prog, "Quality"), param);
    else if (type == AttackType::GAMMA_CORRECT) glUniform1f(glGetUniformLocation(prog, "Gamma"), param);
    else if (type == AttackType::BRIGHTNESS) glUniform1f(glGetUniformLocation(prog, "Gain"), param);
    else if (type == AttackType::ROTATION) glUniform1f(glGetUniformLocation(prog, "Angle"), param);

    glBindImageTexture(0, texIn, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, texOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glDispatchCompute((w + 31) / 32, (h + 31) / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

// [New] 공격 결과 CSV 저장
void SaveAttackResultsToCSV(const std::vector<AttackResult>& results, const char* filename) {
    std::ofstream file(filename);
    file << "Algorithm,Attack Type,Parameter,BER,Recovered BER,PSNR (Attacked)\n";
    for (const auto& res : results) {
        file << res.algoName << "," << res.attackName << "," << res.param << ","
            << res.ber << "," << res.recoveredBER << "," << res.psnr << "\n";
    }
    file.close();
    std::cout << "[System] Attack results saved to " << filename << std::endl;
}

// -----------------------------------------------------------------------------
// 3. 전역 리소스 관리자
// -----------------------------------------------------------------------------
struct ResourceManager {
    // DCT/DWT/SVD 텍스처 (기존 유지)
    GLuint tex_Source = 0;
    GLuint tex_Intermediate = 0, tex_DCTOutput = 0, tex_Final = 0;
    GLuint tex_DWT_Intermediate = 0, tex_DWT_Output = 0, tex_DWT_Final = 0;
    GLuint tex_SVD_Y = 0, tex_SVD_CbCr = 0, tex_SVD_AtA = 0;
    GLuint tex_SVD_V = 0, tex_SVD_Sigma = 0, tex_SVD_U = 0, tex_SVD_ReconY = 0, tex_SVD_Final = 0;
    GLuint tex_Opt_Final = 0;

    // [DFT용 텍스처 추가]
    // 복소수 저장을 위해 RG32F 포맷 사용 (R: Real, G: Imaginary)
    GLuint tex_DFT_Complex = 0;
    GLuint tex_DFT_Final = 0;
    GLuint tex_FFT_Ping = 0; // 핑 (작업용 1)
    GLuint tex_FFT_Pong = 0; // 퐁 (작업용 2)

    GLuint tex_Attacked = 0; // ★ [추가] 공격받은 이미지 저장용


    GLuint buf_Bitstream = 0, buf_Pattern = 0;
    GLuint buf_ExtractedBits = 0;
    uint numBlocks = 0;
    uint numBlocks4x4 = 0; // for optimized/extraction


    void Release() {
        // (기존 삭제 코드 유지 + DFT 추가)
        GLuint textures[] = { tex_Source, tex_Intermediate, tex_DCTOutput, tex_Final,
            tex_DWT_Intermediate, tex_DWT_Output, tex_DWT_Final,
            tex_SVD_Y, tex_SVD_CbCr, tex_SVD_AtA, tex_SVD_V, tex_SVD_Sigma, tex_SVD_U, tex_SVD_ReconY, tex_SVD_Final,
            tex_DFT_Complex, tex_DFT_Final, tex_Opt_Final, tex_FFT_Ping, tex_FFT_Pong }; // Added
        glDeleteTextures(20, textures);
        glDeleteBuffers(1, &buf_Bitstream);
        glDeleteBuffers(1, &buf_Pattern);
        glDeleteTextures(1, &tex_Attacked);
    }

    void Resize(int width, int height, uint coeffsToUse) {
        Release();

        auto createTex = [&](GLuint& id, GLenum fmt) {
            glGenTextures(1, &id);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexStorage2D(GL_TEXTURE_2D, 1, fmt, width, height);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Edge handling important for DFT
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            };

        createTex(tex_Source, GL_RGBA32F);

        // DCT & DWT
        createTex(tex_Intermediate, GL_RGBA32F);
        createTex(tex_DCTOutput, GL_RGBA32F);
        createTex(tex_Final, GL_RGBA32F);
        createTex(tex_DWT_Intermediate, GL_RGBA32F);
        createTex(tex_DWT_Output, GL_RGBA32F);
        createTex(tex_DWT_Final, GL_RGBA32F);

        // SVD
        createTex(tex_SVD_Y, GL_R32F);
        createTex(tex_SVD_CbCr, GL_RG32F);
        createTex(tex_SVD_AtA, GL_R32F);
        createTex(tex_SVD_V, GL_R32F);
        createTex(tex_SVD_Sigma, GL_R32F);
        createTex(tex_SVD_U, GL_R32F);
        createTex(tex_SVD_ReconY, GL_R32F);
        createTex(tex_SVD_Final, GL_RGBA32F);

		createTex(tex_DFT_Final, GL_RGBA32F); // DFT 최종 출력 텍스처
		createTex(tex_Opt_Final, GL_RGBA32F); // Optimized One-Shot 최종 출력 텍스처
        createTex(tex_Attacked, GL_RGBA32F);

        // [FFT 전용 텍스처 생성]
		// 가로/세로 중 큰 쪽을 기준으로 2의 제곱수를 구함 (예: 1920 -> 2048)
        uint paddedSize = NextPowerOfTwo(std::max(width, height));

        auto createTexForFFT = [&](GLuint& id, GLenum fmt, int w, int h) {
            glGenTextures(1, &id);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexStorage2D(GL_TEXTURE_2D, 1, fmt, w, h);
            // FFT는 인접 픽셀을 보간하면 안 되므로 NEAREST 필수
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            };

        // [DFT]
        // RG32F 포맷 사용 (R:실수, G:허수)
        createTexForFFT(tex_FFT_Ping, GL_RG32F, paddedSize, paddedSize);
        createTexForFFT(tex_FFT_Pong, GL_RG32F, paddedSize, paddedSize);
        createTexForFFT(tex_DFT_Complex, GL_RG32F, paddedSize, paddedSize); // DFT 복소수 텍스처

        glBindTexture(GL_TEXTURE_2D, 0);

        // [Test Pattern Generation]
        // 그라데이션 + 격자 패턴으로 주파수 성분이 잘 보이도록 생성
        std::vector<float> dummyData(static_cast<size_t>(width) * height * 4);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t idx = (static_cast<size_t>(y) * width + x) * 4;
                float u = (float)x / width;
                float v = (float)y / height;

                // 좀 더 복잡한 패턴 생성 (주파수 테스트용)
                float val = 0.5f + 0.5f * sin(u * 50.0f) * cos(v * 50.0f);

                dummyData[idx + 0] = u;
                dummyData[idx + 1] = v;
                dummyData[idx + 2] = val;
                dummyData[idx + 3] = 1.0f;
            }
        }
        glBindTexture(GL_TEXTURE_2D, tex_Source);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, dummyData.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        // Buffers
        numBlocks = ((width + 7) / 8) * ((height + 7) / 8);
        numBlocks4x4 = ((width + 3) / 4) * ((height + 3) / 4); // For One-Shot & Extraction

        std::vector<uint> bitstreamData(numBlocks4x4);
        //for (size_t i = 0; i < numBlocks4x4; ++i) bitstreamData[i] = (rand() % 2); // Random bits
        // 1. 짧은 서명(Signature) 정의 (예: 64비트)
        // 실제로는 "COPYRIGHT" 같은 의미 있는 데이터나 로고 비트맵이 됩니다.
        int messageLength = 64;
        std::vector<uint> signature(messageLength);

        // 서명 생성 (여기서는 랜덤으로 만들지만, 고정된 패턴이어도 됨)
        // 매번 실행할 때마다 달라지게 하려면 rand(), 고정하려면 하드코딩
        for (int k = 0; k < messageLength; ++k) signature[k] = (rand() % 2);

        // 2. 전체 버퍼에 서명을 반복해서 채워 넣기 (Tiling)
        // 블록이 100만 개여도, 64비트 패턴이 계속 반복됩니다.
        for (size_t i = 0; i < numBlocks4x4; ++i) {
            bitstreamData[i] = signature[i % messageLength];
        }


        std::vector<float> patternData(numBlocks * coeffsToUse);
        for (size_t i = 0; i < patternData.size(); ++i) patternData[i] = ((rand() % 2) == 0) ? 1.0f : -1.0f;

        glGenBuffers(1, &buf_Bitstream);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_Bitstream);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bitstreamData.size() * sizeof(uint), bitstreamData.data(), GL_STATIC_READ);

        glGenBuffers(1, &buf_Pattern);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_Pattern);
        glBufferData(GL_SHADER_STORAGE_BUFFER, patternData.size() * sizeof(float), patternData.data(), GL_STATIC_READ);

        // ★ Extracted Bits Buffer (Readback용)
        glGenBuffers(1, &buf_ExtractedBits);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_ExtractedBits);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bitstreamData.size() * sizeof(uint), nullptr, GL_DYNAMIC_READ); // Empty init
    
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
};

// -----------------------------------------------------------------------------
// 4. MAIN
// -----------------------------------------------------------------------------

int main()
{
    if (!glfwInit()) return -1;
    const char* glsl_version = "#version 430 core";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1600, 900, "Watermark Benchmark v2.0 (Master Edition)", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // V-Sync Off for benchmarking
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) return -1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // [Load Shaders] - 유저가 제공한 기존 셰이더 경로 유지
    GLuint dctPass1 = loadComputeShader("DCT/dct_pass1_rows.comp");
    GLuint dctPass2 = loadComputeShader("DCT/dct_pass2_cols_embed.comp");
    GLuint dctPass3 = loadComputeShader("DCT/idct_pass1_cols.comp");
    GLuint dctPass4 = loadComputeShader("DCT/idct_pass2_rows.comp");

    GLuint dwtPass1 = loadComputeShader("DWT/dwt_pass1_rows.comp");
    GLuint dwtPass2 = loadComputeShader("DWT/dwt_pass2_cols_embed.comp");
    GLuint dwtPass3 = loadComputeShader("DWT/idwt_pass1_cols.comp");
    GLuint dwtPass4 = loadComputeShader("DWT/idwt_pass2_rows.comp");

    // ★ [Load NEW Optimized Shaders] ★
    // 이전에 만들어드린 단일 파일 셰이더들을 해당 경로에 저장해야 합니다.
    GLuint dctOptProg = loadComputeShader("DCT/dct_opt_4x4.comp");
    GLuint dwtOptProg = loadComputeShader("DWT/dwt_opt_4x4.comp");

    // DFT Shaders (New)
    GLuint dftPadProg = loadComputeShader("DFT/dft_preprocess_pad.comp");
    GLuint dftCoreProg = loadComputeShader("FFT/fft_radix2.comp");
    GLuint dftEmbedProg = loadComputeShader("DFT/dft_process_embed.comp");
    GLuint dftCropProg = loadComputeShader("DFT/dft_postprocess_crop.comp");
	GLuint dftReorderProg = loadComputeShader("DFT/dft_reorder.comp");
	GLuint debugProbeProg = loadComputeShader("DFT/debug_probe.comp");

    GLuint dftOptProg = loadComputeShader("DFT/dft_opt_4x4.comp");

	// SVD Shaders
    GLuint svd_progs[8];
    svd_progs[0] = loadComputeShader("SVD/svd_01_rgb_to_y.comp");
    svd_progs[1] = loadComputeShader("SVD/svd_02_store_cbcr.comp");
    svd_progs[2] = loadComputeShader("SVD/svd_03_compute_ata.comp");
    svd_progs[3] = loadComputeShader("SVD/svd_04_eigendecomposition.comp");
    svd_progs[4] = loadComputeShader("SVD/svd_05_compute_u.comp");
    svd_progs[5] = loadComputeShader("SVD/svd_06_modify_sigma.comp");
    svd_progs[6] = loadComputeShader("SVD/svd_07_reconstruct_y.comp");
    svd_progs[7] = loadComputeShader("SVD/svd_08_combine_ycbcr.comp");

	GLuint svd4x4Prog = loadComputeShader("SVD/svd_block_4x4.comp");
	GLuint svdImplictProg = loadComputeShader("SVD/svd_implict_4x4.comp");

    // ★ [Load Extraction Shaders]
    GLuint extractDctProg = loadComputeShader("Extraction/extract_dct.comp");
    GLuint extractDwtProg = loadComputeShader("Extraction/extract_dwt.comp");
    GLuint extractSvdProg = loadComputeShader("Extraction/extract_svd.comp");
    GLuint extractDftProg = loadComputeShader("Extraction/extract_dft.comp");

    GLuint extractDctSSProg = loadComputeShader("Extraction/extract_dct_ss.comp");
    GLuint extractDwtSSProg = loadComputeShader("Extraction/extract_dwt_ss.comp");
    GLuint extractDftGlobalProg = loadComputeShader("Extraction/extract_dft_global.comp");

    // Resource Manager
    ResourceManager resMgr;
    int currentWidth = 1920;
    int currentHeight = 1080;
    uint coeffsToUse = 10;
    resMgr.Resize(currentWidth, currentHeight, coeffsToUse);

    // Benchmark Params
    bool g_EnableEmbed = true;
    float g_EmbeddingStrength = 25.0f; // DFT/DCT needs higher strength usually
    float g_CompressionThreshold = 0.0f;
    uint g_JacobiIterations = 4;
    float g_SigmaThreshold = 1.0e-7f;
    int g_SVDMode = 0;
    int g_AlgorithmChoice = 0; // 0=DCT, 1=DWT, 2=SVD, 3=DFT

    bool g_RunBenchmark = false;
    int g_BenchmarkIteration = 0;
    const int g_IterationsPerRes = 10; // 100회는 너무 기니 테스트용 10회
    int g_CurrentResIndex = 0;
    struct Resolution { std::string name; int w; int h; };
    Resolution resolutions[] = { {"FHD", 1920, 1080}, {"4K", 3840, 2160} };

    // ★ Alpha Sweep 전용 변수
    bool g_RunAlphaSweep = false;
    double g_SweepCurrentAlpha = 0.0f;
    int g_SweepAlgoIndex = 0; // 0~6 (모든 알고리즘 순회)
    int g_SweepResIndex = 1; // 0: FHD, 1: 4K
    bool g_RunAttackTest = false;
    std::vector<AttackResult> g_AttackResults;

    bool g_RunAlphaAttackSweep = false;

    // 공격 시나리오 정의
    std::vector<AttackConfig> g_AttackScenarios = {
        { AttackType::NONE, 0.0f, "No Attack" },
        { AttackType::CROP_CENTER, 0.5f, "Crop 50%" },
        { AttackType::CROP_CENTER, 0.7f, "Crop 70%" }, // 30% 잘림
        { AttackType::RESIZE_SCALE, 0.5f, "Resize 50%" },
        { AttackType::NOISE_SNP, 0.01f, "Salt&Pepper 1%" },
        { AttackType::NOISE_SNP, 0.05f, "Salt&Pepper 5%" },
        { AttackType::QUANTIZATION, 32.0f, "Quantize (32 levels)" },
        { AttackType::GAMMA_CORRECT, 0.8f, "Gamma 0.8" },
        { AttackType::BRIGHTNESS, 0.2f, "Brightness +0.2" },
        { AttackType::ROTATION, 1.0f, "Rotation 1 deg" },   // 미세 회전 (격자 파괴)
        { AttackType::ROTATION, 5.0f, "Rotation 5 deg" },   // 소규모 회전
        { AttackType::ROTATION, 45.0f, "Rotation 45 deg" }  // 대규모 회전
    };

    std::vector<BenchmarkResult> g_Results;
    GpuTimer gpuTimer;
	gpuTimer.Init();
    CpuTimer cpuTimer;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI
        ImGui::Begin("Control Panel");
        ImGui::Text("Res: %dx%d | FPS: %.1f", currentWidth, currentHeight, ImGui::GetIO().Framerate);

        ImGui::Text("Algorithm Selection:");
        ImGui::RadioButton("DCT", &g_AlgorithmChoice, 0);
    	ImGui::SameLine();
        ImGui::RadioButton("DCT (Optimized 1-Pass)", &g_AlgorithmChoice, 5); // New

    	ImGui::RadioButton("DWT", &g_AlgorithmChoice, 1);
    	ImGui::SameLine();
        ImGui::RadioButton("DWT (Optimized 1-Pass)", &g_AlgorithmChoice, 6); // New

        ImGui::RadioButton("SVD (Block)", &g_AlgorithmChoice, 2);
        ImGui::SameLine();
        ImGui::RadioButton("SVD (Implicit)", &g_AlgorithmChoice, 3);
    	
        ImGui::RadioButton("DFT", &g_AlgorithmChoice, 4); // Added
    	ImGui::SameLine();
        ImGui::RadioButton("DFT (Optimized 1-Pass)", &g_AlgorithmChoice, 7); // New

        ImGui::Checkbox("Embed", &g_EnableEmbed);
        ImGui::SliderFloat("Strength", &g_EmbeddingStrength, 0.0f, 50.0f);

        if (ImGui::Button("Run Full Benchmark")) {
            g_RunBenchmark = true;
            g_BenchmarkIteration = 0;
            g_CurrentResIndex = 0;
            g_Results.clear();
        }

        // ... 기존 UI 버튼들 아래에 추가 ...
        ImGui::Separator();
        ImGui::Text("Research Mode:");
        if (ImGui::Button("Run Alpha Sweep (0.0 -> 1.0)")) {
            g_RunAlphaSweep = true;
            g_SweepCurrentAlpha = 0.0f;
            g_SweepAlgoIndex = 0;
            g_Results.clear();
            std::cout << "[System] Starting Alpha Sweep..." << std::endl;
        }
        ImGui::Text("Finds optimal Alpha for PSNR ~40dB");

        // [Main Loop 내부 UI 부분]
        if (ImGui::Button("Run Attack Robustness Test")) {
            g_RunAttackTest = true;
            g_AttackResults.clear();
            std::cout << "[System] Starting Attack Simulation..." << std::endl;
        }

        if (ImGui::Button("Run Alpha-Attack Sweep (Ultimate Test)")) {
            g_RunAlphaAttackSweep = true;
            g_AttackResults.clear(); // 결과 초기화
            std::cout << "[System] Starting Alpha-Attack Sweep..." << std::endl;
        }

        ImGui::End();

        // Preview
        GLuint previewTex = resMgr.tex_Source; // Default
        switch (g_AlgorithmChoice) {
        case 0: previewTex = resMgr.tex_Final; break;
        case 1: previewTex = resMgr.tex_DWT_Final; break;
        case 2: case 3: previewTex = resMgr.tex_SVD_Final; break;
        case 4: previewTex = resMgr.tex_DFT_Final; break;
        case 5: case 6: case 7: previewTex = resMgr.tex_Opt_Final; break; // Optimized Output
        }

        ImGui::Begin("Preview");
    	if (previewTex) ImGui::Image(previewTex, ImVec2(1280, 720));
        ImGui::End();

        // Logic
        if (g_RunBenchmark) {

            // Benchmark Loop
            Resolution res = resolutions[g_CurrentResIndex];
            if (g_BenchmarkIteration == 0 && g_CurrentResIndex < 2) {
                resMgr.Resize(res.w, res.h, coeffsToUse);
                glFinish();
                currentWidth = res.w; currentHeight = res.h;
            }

            // [Benchmark Algorithm List]
            // 0:DCT_Legacy, 1:DWT_Legacy, 2:SVD_Block, 3:SVD_Impl, 4:DFT, 5:DCT_Opt, 6:DWT_Opt
            std::string algoNames[] = {
                "DCT (Legacy)", "DWT (Legacy)", "SVD (Block)", "SVD (Implicit)",
                "DFT", "DCT (Optimized)", "DWT (Optimized)", "DFT (Optimized)"
            };

            // i loops through all 8 algorithms
            for (int i = 0; i < 8; ++i) {
                cpuTimer.Start();
                gpuTimer.Start();

                if (i == 0) Run_Compute_Pipeline(dctPass1, dctPass2, dctPass3, dctPass4,
                    resMgr.tex_Source, resMgr.tex_Intermediate, resMgr.tex_DCTOutput, resMgr.tex_Final,
                    resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight,
                    g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                else if (i == 1) Run_Compute_Pipeline(dwtPass1, dwtPass2, dwtPass3, dwtPass4,
                    resMgr.tex_Source, resMgr.tex_DWT_Intermediate, resMgr.tex_DWT_Output, resMgr.tex_DWT_Final,
                    resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight,
                    g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                else if (i == 2) Run_Optimized_OneShot_Pipeline(svd4x4Prog, resMgr.tex_Source, resMgr.tex_SVD_Final, resMgr.buf_Bitstream, res.w, res.h, g_EnableEmbed, g_EmbeddingStrength);
                else if (i == 3) Run_Optimized_OneShot_Pipeline(svdImplictProg, resMgr.tex_Source, resMgr.tex_SVD_Final, resMgr.buf_Bitstream, res.w, res.h, g_EnableEmbed, g_EmbeddingStrength);
                else if (i == 4) Run_Full_FFT_Pipeline(
                    dftPadProg,    // Pad
                    dftReorderProg,
                    dftCoreProg,   // Core FFT
                    dftEmbedProg,  // Embed
                    dftCropProg,   // Crop
                    debugProbeProg,  // Debug Probe (optional)
                    resMgr.tex_Source,    // 원본
                    resMgr.tex_DFT_Final, // 최종 결과 (Resize 함수에 변수명 맞춰주세요. 예: tex_Final)
                    resMgr.tex_FFT_Ping,  // 작업용 1 (2048 size)
                    resMgr.tex_FFT_Pong,  // 작업용 2 (2048 size)
                    resMgr.buf_Bitstream, // 워터마크 비트
                    currentWidth, currentHeight,
                    g_EnableEmbed, g_EmbeddingStrength
                );
                else if (i == 5) Run_Optimized_OneShot_Pipeline(dctOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, res.w, res.h, g_EnableEmbed, g_EmbeddingStrength);
                else if (i == 6) Run_Optimized_OneShot_Pipeline(dwtOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, res.w, res.h, g_EnableEmbed, g_EmbeddingStrength);


                gpuTimer.Stop();
                float cpuTime = cpuTimer.GetTimeMs();
                float gpuTime = gpuTimer.GetTimeMs();

                // Metrics (Calculate only on first iteration to save time)
                double psnr = 0, ssim = 0;
                if (g_BenchmarkIteration == 0) {
                    GLuint outTex = (i == 0) ? resMgr.tex_Final : (i == 1) ? resMgr.tex_DWT_Final : (i == 2 || i == 3) ? resMgr.tex_SVD_Final : (i == 4) ? resMgr.tex_DFT_Final : resMgr.tex_Opt_Final;
                    // Readback & Calc Logic...
                }
                g_Results.push_back({ algoNames[i], res.name, g_SweepCurrentAlpha, gpuTime, cpuTime, psnr, ssim });
            }

            g_BenchmarkIteration++;
            if (g_BenchmarkIteration >= g_IterationsPerRes) {
                g_BenchmarkIteration = 0;
                g_CurrentResIndex++;
                if (g_CurrentResIndex >= 2) {
                    g_RunBenchmark = false;
                    SaveResultsToCSV(g_Results, "Benchmark_Ultimate_Results.csv");
                    // Reset
                    g_CurrentResIndex = 0;
                    resMgr.Resize(1920, 1080, coeffsToUse);
                    currentWidth = 1920; currentHeight = 1080;
                }
            }

            
        }

        // ... 기존 g_RunBenchmark 블록 끝난 뒤 ...

        else if (g_RunAlphaSweep) {

            // 1. 현재 해상도 설정 및 리소스 초기화 (해상도가 바뀌거나 첫 시작일 때)
            Resolution currentRes = resolutions[g_SweepResIndex];

            // 첫 프레임(Alpha=0)에서만 리사이즈 수행 (매번 하면 느려짐)
            if (g_SweepCurrentAlpha == 0.0f && g_SweepAlgoIndex == 0) {
                if (currentWidth != currentRes.w || currentHeight != currentRes.h) {
                    std::cout << "[System] Resizing to " << currentRes.name << " (" << currentRes.w << "x" << currentRes.h << ")..." << std::endl;
                    resMgr.Resize(currentRes.w, currentRes.h, coeffsToUse);
                    currentWidth = currentRes.w;
                    currentHeight = currentRes.h;
                    glFinish(); // 메모리 할당 대기
                }
            }
            

            // 2. 셰이더에 보낼 Strength 계산
            // 셰이더: alpha = Strength * 0.01 
            // 목표: alpha (0~1)
            // 따라서: Strength = alpha * 100.0
            float currentStrength = g_SweepCurrentAlpha * 100.0;

            std::string algoNames[] = {
                "DCT (Legacy)", "DWT (Legacy)", "SVD (Block)", "SVD (Implicit)",
				"DFT", "DCT (Optimized)", "DWT (Optimized)", "DFT (Optimized)"
            };

            cpuTimer.Start();
            gpuTimer.Start();

            // 2. Embedding
            // ★ [수정] 평균을 내기 위한 반복 횟수 설정 (30~50회 추천)
            // 횟수가 많을수록 그래프가 매끄러워지지만, 전체 측정 시간이 길어집니다.
            const int BENCHMARK_SAMPLES = 50;
            int i = g_SweepAlgoIndex;

            // ★ [핵심] 반복 루프 추가
            for (int k = 0; k < BENCHMARK_SAMPLES; ++k)
            {
                
                if (i == 0) Run_Compute_Pipeline(dctPass1, dctPass2, dctPass3, dctPass4, resMgr.tex_Source, resMgr.tex_Intermediate, resMgr.tex_DCTOutput, resMgr.tex_Final, resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight, true, currentStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                else if (i == 1) Run_Compute_Pipeline(dwtPass1, dwtPass2, dwtPass3, dwtPass4, resMgr.tex_Source, resMgr.tex_DWT_Intermediate, resMgr.tex_DWT_Output, resMgr.tex_DWT_Final, resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight, true, currentStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                else if (i == 2) Run_SVD_Pipeline(
                    svd_progs,              // 셰이더 프로그램 배열 (main 상단에 로드됨)
                    svd_progs,               // 텍스처 배열
                    resMgr.buf_Bitstream,   // 워터마크 비트 버퍼
                    resMgr.buf_Pattern,     // 패턴 버퍼
                    currentWidth,
                    currentHeight,
                    g_JacobiIterations,     // 자코비 반복 횟수 (UI 연동)
                    g_SigmaThreshold,       // 임계값 (UI 연동)
                    g_EnableEmbed,          // 워터마크 삽입 여부
                    currentStrength,    // 강도
                    g_CompressionThreshold, // 압축 임계값
                    coeffsToUse,            // 사용 계수 (보통 10)
                    resMgr.numBlocks,       // Legacy는 8x8 블록 개수 사용
                    resMgr.numBlocks        // 비트 길이도 블록 개수와 동일하게 설정
                );
                else if (i == 3) Run_Optimized_OneShot_Pipeline(svdImplictProg, resMgr.tex_Source, resMgr.tex_SVD_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                else if (i == 4) Run_Full_FFT_Pipeline(dftPadProg, dftReorderProg, dftCoreProg, dftEmbedProg, dftCropProg, debugProbeProg, resMgr.tex_Source, resMgr.tex_DFT_Final, resMgr.tex_FFT_Ping, resMgr.tex_FFT_Pong, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                else if (i == 5) Run_Optimized_OneShot_Pipeline(dctOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                else if (i == 6) Run_Optimized_OneShot_Pipeline(dwtOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                else if (i == 7) Run_Optimized_OneShot_Pipeline(dftOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
            }

            gpuTimer.Stop(); // 타이머 종료 (이제 50회분의 시간이 기록됨)
            // 3. Extraction (추출)
            // Legacy(0,1)는 Spread Spectrum이라 정확한 Extraction을 위해선 전용 셰이더가 필요하나, 
            // 여기서는 최적화 셰이더와 동일한 논리로 동작하는 기본 추출기를 사용하여 BER 측정 시도.
            GLuint finalTex = 0;
            // 텍스처 ID 매핑 (반복문 밖으로 뺌)
            if (i == 0) finalTex = resMgr.tex_Final;
        	else if (i == 1) finalTex = resMgr.tex_DWT_Final;
        	else if (i == 2 || i == 3) finalTex = resMgr.tex_SVD_Final;
        	else if (i == 4) finalTex = resMgr.tex_DFT_Final;
        	else finalTex = resMgr.tex_Opt_Final;

			if (i == 0) // DCT Legacy
            {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resMgr.buf_Pattern); // 패턴 버퍼 필수
                Run_Legacy_Extraction(extractDctSSProg, resMgr.tex_Source, finalTex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks);
            }
            else if (i == 1) // DWT Legacy
            {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resMgr.buf_Pattern); // 패턴 버퍼 필수
                Run_Legacy_Extraction(extractDwtSSProg, resMgr.tex_Source, finalTex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks);
            }

            else if (i == 2 || i == 3) { // SVD
                Run_Extraction_Pipeline(extractSvdProg, resMgr.tex_Source, finalTex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
            }

            else if (i == 4) { // DFT Legacy
                Run_Full_FFT_Extraction(
                    dftPadProg, dftReorderProg, dftCoreProg, extractDftGlobalProg, // 셰이더들
                    resMgr.tex_Source, finalTex,       // 입력 이미지 (원본, 마킹됨)
                    resMgr.tex_FFT_Ping, resMgr.tex_FFT_Pong, // 작업용
                    resMgr.tex_DFT_Complex,            // 백업용 (ResourceManager에 추가되어 있음)
                    resMgr.buf_ExtractedBits,
                    currentWidth, currentHeight,
                    32768 // ★ 중요: Embed 셰이더의 상수(32768)와 일치시켜야 함.
                          // (Legacy Embed에서 bitIdx = linearIdx % 32768 했으므로 여기서도 동일하게)
                );  
            }

            else if(i == 5) { // DCT Optimized
                Run_Extraction_Pipeline(extractDctProg, resMgr.tex_Source, finalTex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
            }
            else if (i == 6) { // DWT Optimized
                Run_Extraction_Pipeline(extractDwtProg, resMgr.tex_Source, finalTex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
            }
            
			else if (i == 7) // DFT Optimized
            {
                Run_Extraction_Pipeline(extractDftProg, resMgr.tex_Source, finalTex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
            }

           
            glFinish(); // 확실한 동기화

            // 4. Readback & Metrics
            std::vector<unsigned char> srcPx(static_cast<size_t>(currentWidth)* currentHeight * 4);
            std::vector<unsigned char> dstPx(static_cast<size_t>(currentWidth)* currentHeight * 4);
            glBindTexture(GL_TEXTURE_2D, resMgr.tex_Source);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, srcPx.data());
            glBindTexture(GL_TEXTURE_2D, finalTex);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, dstPx.data());

            // ★ Read Extracted Bits
            std::vector<uint> orgBits(resMgr.numBlocks4x4);
            std::vector<uint> extBits(resMgr.numBlocks4x4);

            // Read Original Bits (CPU Copy is faster than VRAM readback if static, but let's read buffer for correctness)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resMgr.buf_Bitstream);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, orgBits.size() * sizeof(uint), orgBits.data());

            // Read Extracted Bits
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resMgr.buf_ExtractedBits);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, extBits.size() * sizeof(uint), extBits.data());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            // Calc
            ImageMetrics m = CalculateMetrics(srcPx, dstPx, currentWidth, currentHeight);
            // ★ [핵심] 알고리즘별 유효 비트 수 설정
            size_t validBits = resMgr.numBlocks4x4; // 기본 (최적화 버전)
            if (i == 0 || i == 1) validBits = resMgr.numBlocks; // Legacy는 8x8 개수만큼만 유효

            // 수정된 함수 호출
            float ber = CalculateBER(orgBits, extBits, validBits);
            float avgGpuTime = gpuTimer.GetTimeMs() / static_cast<float>(BENCHMARK_SAMPLES);
            float avgCpuTime = cpuTimer.GetTimeMs() / static_cast<float>(BENCHMARK_SAMPLES);

            // 5. 결과 저장
            g_Results.push_back({
                algoNames[i],
                currentRes.name,
                g_SweepCurrentAlpha, // 0.0 ~ 1.0
                avgGpuTime,
                avgCpuTime,
                m.psnr,
                m.ssim,
                ber
                });

            // 진행 상황 로그 출력 (너무 많으면 느려지니 0.1단위만 출력해도 됨)
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Algo: " << i << " | Alpha: " << g_SweepCurrentAlpha << " | PSNR: " << m.psnr << '\n';

            // 6. 다음 단계로 이동
            g_SweepCurrentAlpha += 0.1f; // 0.001 단위 증가

            if (g_SweepCurrentAlpha > 2.0001f) { // 부동소수점 오차 고려
                g_SweepCurrentAlpha = 0.0f;
                g_SweepAlgoIndex++; // 다음 알고리즘으로

                if (g_SweepAlgoIndex >= 8) {
                    g_SweepAlgoIndex = 0;
                    g_SweepResIndex++; // ★ 다음 해상도 (FHD -> 4K)

                    // 모든 해상도 완료?
                    if (g_SweepResIndex >= 2) {
                        g_RunAlphaSweep = false;
                        g_SweepResIndex = 0; // 초기화
                        SaveResultsToCSV(g_Results, "Alpha_Sweep_Results_Ultimate.csv");
                        std::cout << "[System] All Benchmarks Completed!" << std::endl;

                        // FHD로 복귀 (UI용)
                        resMgr.Resize(1920, 1080, coeffsToUse);
                        currentWidth = 1920; currentHeight = 1080;
                    }
                }
            }
        }

        // [New] 공격 벤치마크 실행 로직
        else if (g_RunAttackTest) {

            // 1. 공격용 셰이더 로드 (최초 1회)
            static GLuint atkCrop = loadComputeShader("Attack/attack_crop.comp");
            static GLuint atkResize = loadComputeShader("Attack/attack_resize.comp");
            static GLuint atkNoise = loadComputeShader("Attack/attack_noise.comp");
            static GLuint atkQuant = loadComputeShader("Attack/attack_quantize.comp");
            static GLuint atkGamma = loadComputeShader("Attack/attack_gamma.comp");
            static GLuint atkBright = loadComputeShader("Attack/attack_brightness.comp");
            static GLuint atkRot = loadComputeShader("Attack/attack_rotation.comp");

            // 2. 테스트할 알고리즘 목록 (원하는 것만 선택 가능)
            int targetAlgos[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
            // 0:DCT_Leg, 1:DWT_Leg, 2:SVD_Std, 3:SVD_Imp, 4:DFT_Leg, 5:DCT_Opt, 6:DWT_Opt, 7:DFT_Opt
            std::string algoNames[] = { "DCT(L)", "DWT(L)", "SVD(S)", "SVD(I)", "DFT(L)", "DCT(O)", "DWT(O)", "DFT(O)" };

            // 3. 고정 강도 설정 (잘 보이는 강도로 설정, 예: Alpha 0.2 -> Strength 20.0)
            float fixedAlpha = 0.2f;
            float fixedStrength = fixedAlpha * 100.0f;

            // === 2중 루프: 알고리즘 -> 공격 ===
            for (int algoIdx : targetAlgos) {
                for (const auto& scenario : g_AttackScenarios) {

                    // (A) Embedding (기존 파이프라인 재사용)
                    // 반복 횟수(Samples)는 1회면 충분함 (BER 측정용이므로)
                    if (algoIdx == 0) Run_Compute_Pipeline(dctPass1, dctPass2, dctPass3, dctPass4, resMgr.tex_Source, resMgr.tex_Intermediate, resMgr.tex_DCTOutput, resMgr.tex_Final, resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight, true, fixedStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                    else if (algoIdx == 1) Run_Compute_Pipeline(dwtPass1, dwtPass2, dwtPass3, dwtPass4, resMgr.tex_Source, resMgr.tex_DWT_Intermediate, resMgr.tex_DWT_Output, resMgr.tex_DWT_Final, resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight, true, fixedStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                    else if (algoIdx == 2) Run_SVD_Pipeline(
                        svd_progs,              // 셰이더 프로그램 배열 (main 상단에 로드됨)
                        svd_progs,               // 텍스처 배열
                        resMgr.buf_Bitstream,   // 워터마크 비트 버퍼
                        resMgr.buf_Pattern,     // 패턴 버퍼
                        currentWidth,
                        currentHeight,
                        g_JacobiIterations,     // 자코비 반복 횟수 (UI 연동)
                        g_SigmaThreshold,       // 임계값 (UI 연동)
                        g_EnableEmbed,          // 워터마크 삽입 여부
                        fixedStrength,    // 강도
                        g_CompressionThreshold, // 압축 임계값
                        coeffsToUse,            // 사용 계수 (보통 10)
                        resMgr.numBlocks,       // Legacy는 8x8 블록 개수 사용
                        resMgr.numBlocks        // 비트 길이도 블록 개수와 동일하게 설정
                    );
                    else if (algoIdx == 3) Run_Optimized_OneShot_Pipeline(svdImplictProg, resMgr.tex_Source, resMgr.tex_SVD_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, fixedStrength);
                    else if (algoIdx == 4) Run_Full_FFT_Pipeline(dftPadProg, dftReorderProg, dftCoreProg, dftEmbedProg, dftCropProg, debugProbeProg, resMgr.tex_Source, resMgr.tex_DFT_Final, resMgr.tex_FFT_Ping, resMgr.tex_FFT_Pong, resMgr.buf_Bitstream, currentWidth, currentHeight, true, fixedStrength);
                    else if (algoIdx == 5) Run_Optimized_OneShot_Pipeline(dctOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, fixedStrength);
                    else if (algoIdx == 6) Run_Optimized_OneShot_Pipeline(dwtOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, fixedStrength);
                    else if (algoIdx == 7) Run_Optimized_OneShot_Pipeline(dftOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, fixedStrength);

                    glMemoryBarrier(GL_ALL_BARRIER_BITS);

                    // (B) Attack Execution
                    GLuint texMarked = 0; // Embed 결과 텍스처
                    if (algoIdx == 0) texMarked = resMgr.tex_Final;
                    else if (algoIdx == 1) texMarked = resMgr.tex_DWT_Final;
                    else if (algoIdx == 2 || algoIdx == 3) texMarked = resMgr.tex_SVD_Final;
                    else if (algoIdx == 4) texMarked = resMgr.tex_DFT_Final;
                    else texMarked = resMgr.tex_Opt_Final;

                    // 공격 수행 -> 결과는 tex_Attacked에 저장
                    if (scenario.type == AttackType::NONE) {
                        // 공격 없으면 그냥 복사
                        glCopyImageSubData(texMarked, GL_TEXTURE_2D, 0, 0, 0, 0,
                            resMgr.tex_Attacked, GL_TEXTURE_2D, 0, 0, 0, 0,
                            currentWidth, currentHeight, 1);
                    }
                    else {
                        GLuint prog = 0;
                        if (scenario.type == AttackType::CROP_CENTER) prog = atkCrop;
                        else if (scenario.type == AttackType::RESIZE_SCALE) prog = atkResize;
                        else if (scenario.type == AttackType::NOISE_SNP) prog = atkNoise;
                        else if (scenario.type == AttackType::QUANTIZATION) prog = atkQuant;
                        else if (scenario.type == AttackType::GAMMA_CORRECT) prog = atkGamma;
                        else if (scenario.type == AttackType::BRIGHTNESS) prog = atkBright;
                        else if (scenario.type == AttackType::ROTATION) prog = atkRot;

                        Run_Attack_Shader(prog, texMarked, resMgr.tex_Attacked, currentWidth, currentHeight, scenario.param, scenario.type);
                    }

                    glMemoryBarrier(GL_ALL_BARRIER_BITS);

                    // (C) Extraction (입력을 tex_Attacked로 변경!)
                    if (algoIdx == 0) {
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resMgr.buf_Pattern);
                        Run_Legacy_Extraction(extractDctSSProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks);
                    }
                    else if (algoIdx == 1) {
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resMgr.buf_Pattern);
                        Run_Legacy_Extraction(extractDwtSSProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks);
                    }
                    else if (algoIdx == 2 || algoIdx == 3) {
                        Run_Extraction_Pipeline(extractSvdProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
                    }
                    else if (algoIdx == 4) {
                        Run_Full_FFT_Extraction(dftPadProg, dftReorderProg, dftCoreProg, extractDftGlobalProg,
                            resMgr.tex_Source, resMgr.tex_Attacked,
                            resMgr.tex_FFT_Ping, resMgr.tex_FFT_Pong, resMgr.tex_DFT_Complex, resMgr.buf_ExtractedBits,
                            currentWidth, currentHeight, 32768);
                    }
                    else if (algoIdx == 5) Run_Extraction_Pipeline(extractDctProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
                    else if (algoIdx == 6) Run_Extraction_Pipeline(extractDwtProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
                    else if (algoIdx == 7) Run_Extraction_Pipeline(extractDftProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);

                    glFinish();

                    // (D) BER & PSNR Calculation
                    // PSNR은 "공격받은 이미지" vs "원본" 비교 (공격 강도 확인용)
                    // 워터마크가 살아있는지는 BER로 확인
                    std::vector<unsigned char> srcPx(currentWidth * currentHeight * 4);
                    std::vector<unsigned char> atkPx(currentWidth * currentHeight * 4);

                    glBindTexture(GL_TEXTURE_2D, resMgr.tex_Source);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, srcPx.data());
                    glBindTexture(GL_TEXTURE_2D, resMgr.tex_Attacked);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, atkPx.data());

                    // Read Bits
                    std::vector<uint> orgBits(resMgr.numBlocks4x4);
                    std::vector<uint> extBits(resMgr.numBlocks4x4);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resMgr.buf_Bitstream);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, orgBits.size() * sizeof(uint), orgBits.data());
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resMgr.buf_ExtractedBits);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, extBits.size() * sizeof(uint), extBits.data());
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

                    // Calc Metrics
                    ImageMetrics m = CalculateMetrics(srcPx, atkPx, currentWidth, currentHeight);
                    size_t validBits = (algoIdx == 0 || algoIdx == 1) ? resMgr.numBlocks : resMgr.numBlocks4x4;
                    // 서명 길이(64)는 위에서 정의한 messageLength와 같아야 합니다.
                    int msgLen = 64;

                    // 원본 서명 추출 (반복된 패턴의 첫 64개만 가져옴)
                    std::vector<uint> originalSignature;
                    if (orgBits.size() >= msgLen) {
                        originalSignature.assign(orgBits.begin(), orgBits.begin() + msgLen);
                    }

                    // 복구 BER 계산
                    float recoveredBER = CalculateRecoveredBER(extBits, originalSignature, msgLen);

                    // 참고용: Raw BER도 같이 보고 싶으면 남겨둠
                    float rawBER = CalculateBER(orgBits, extBits, validBits);

                    // (E) Log & Save
                    // 화면에는 Recovered BER을 출력해서 "성공"했음을 확인
                    std::cout << "Algo: " << algoNames[algoIdx] << " | Attack: " << scenario.name
                        << " | Recov.BER: " << recoveredBER << " (Raw: " << rawBER << ")"
                        << " | PSNR: " << m.psnr << std::endl;

                    // CSV에는 Recovered BER 저장
                    g_AttackResults.push_back({ algoNames[algoIdx], scenario.name, scenario.param, rawBER, recoveredBER, m.psnr });
                }
            }

            // CSV 저장
            SaveAttackResultsToCSV(g_AttackResults, "Attack_Robustness_Results.csv");
            g_RunAttackTest = false;
            std::cout << "[System] Attack Simulation Finished!" << std::endl;
        }

        else if (g_RunAlphaAttackSweep) {

            // 1. 공격 셰이더 로드 (static으로 한 번만 로드)
            static GLuint atkCrop = loadComputeShader("Attack/attack_crop.comp");
            static GLuint atkResize = loadComputeShader("Attack/attack_resize.comp");
            static GLuint atkNoise = loadComputeShader("Attack/attack_noise.comp");
            static GLuint atkQuant = loadComputeShader("Attack/attack_quantize.comp");
            static GLuint atkGamma = loadComputeShader("Attack/attack_gamma.comp");
            static GLuint atkBright = loadComputeShader("Attack/attack_brightness.comp");
            static GLuint atkRot = loadComputeShader("Attack/attack_rotation.comp");

            // 2. 타겟 알고리즘 및 공격 설정
            int targetAlgos[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
            std::string algoNames[] = { "DCT(L)", "DWT(L)", "SVD(S)", "SVD(I)", "DFT(L)", "DCT(O)", "DWT(O)", "DFT(O)" };

            // ★ 해상도 목록 정의 (FHD, 4K)
            struct ResConfig { std::string name; int w; int h; };
            std::vector<ResConfig> targetResolutions = {
                //{ "FHD", 1920, 1080 },
                { "4K", 3840, 2160 }
            };

            // =========================================================
        // Loop 1: Resolution (FHD -> 4K)
        // =========================================================
            for (const auto& res : targetResolutions) {

                std::cout << "\n[System] Switching to Resolution: " << res.name << " (" << res.w << "x" << res.h << ")" << std::endl;

                // ★ 해상도 변경 및 메모리 재할당
                resMgr.Resize(res.w, res.h, coeffsToUse);
                currentWidth = res.w;
                currentHeight = res.h;
                glFinish(); // 메모리 할당 대기

                // 3. ★ Alpha Loop 추가 (0.0 ~ 1.5, 0.1 간격)
                for (float alpha = 0.0f; alpha <= 1.51f; alpha += 0.1f) {

                    float currentStrength = alpha * 100.0f; // 셰이더용 강도 변환
                    std::cout << "\n>>> Processing Alpha: " << alpha << " <<<" << std::endl;

                    // Algorithm Loop
                    for (int algoIdx : targetAlgos) {
                        // Attack Loop
                        for (const auto& scenario : g_AttackScenarios) {

                            // (A) Embedding (현재 Alpha 강도로 삽입)
                            if (algoIdx == 0) Run_Compute_Pipeline(dctPass1, dctPass2, dctPass3, dctPass4, resMgr.tex_Source, resMgr.tex_Intermediate, resMgr.tex_DCTOutput, resMgr.tex_Final, resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight, true, currentStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                            else if (algoIdx == 1) Run_Compute_Pipeline(dwtPass1, dwtPass2, dwtPass3, dwtPass4, resMgr.tex_Source, resMgr.tex_DWT_Intermediate, resMgr.tex_DWT_Output, resMgr.tex_DWT_Final, resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight, true, currentStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
                            else if (algoIdx == 2) Run_SVD_Pipeline(
                                svd_progs,              // 셰이더 프로그램 배열 (main 상단에 로드됨)
                                svd_progs,               // 텍스처 배열
                                resMgr.buf_Bitstream,   // 워터마크 비트 버퍼
                                resMgr.buf_Pattern,     // 패턴 버퍼
                                currentWidth,
                                currentHeight,
                                g_JacobiIterations,     // 자코비 반복 횟수 (UI 연동)
                                g_SigmaThreshold,       // 임계값 (UI 연동)
                                g_EnableEmbed,          // 워터마크 삽입 여부
                                currentStrength,    // 강도
                                g_CompressionThreshold, // 압축 임계값
                                coeffsToUse,            // 사용 계수 (보통 10)
                                resMgr.numBlocks,       // Legacy는 8x8 블록 개수 사용
                                resMgr.numBlocks        // 비트 길이도 블록 개수와 동일하게 설정
                            );
                            else if (algoIdx == 3) Run_Optimized_OneShot_Pipeline(svdImplictProg, resMgr.tex_Source, resMgr.tex_SVD_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                            else if (algoIdx == 4) Run_Full_FFT_Pipeline(dftPadProg, dftReorderProg, dftCoreProg, dftEmbedProg, dftCropProg, debugProbeProg, resMgr.tex_Source, resMgr.tex_DFT_Final, resMgr.tex_FFT_Ping, resMgr.tex_FFT_Pong, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                            else if (algoIdx == 5) Run_Optimized_OneShot_Pipeline(dctOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                            else if (algoIdx == 6) Run_Optimized_OneShot_Pipeline(dwtOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);
                            else if (algoIdx == 7) Run_Optimized_OneShot_Pipeline(dftOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final, resMgr.buf_Bitstream, currentWidth, currentHeight, true, currentStrength);

                            glMemoryBarrier(GL_ALL_BARRIER_BITS);

                            // (B) Attack Execution
                            GLuint texMarked = 0;
                            if (algoIdx == 0) texMarked = resMgr.tex_Final;
                            else if (algoIdx == 1) texMarked = resMgr.tex_DWT_Final;
                            else if (algoIdx == 2 || algoIdx == 3) texMarked = resMgr.tex_SVD_Final;
                            else if (algoIdx == 4) texMarked = resMgr.tex_DFT_Final;
                            else texMarked = resMgr.tex_Opt_Final;

                            if (scenario.type == AttackType::NONE) {
                                glCopyImageSubData(texMarked, GL_TEXTURE_2D, 0, 0, 0, 0, resMgr.tex_Attacked, GL_TEXTURE_2D, 0, 0, 0, 0, currentWidth, currentHeight, 1);
                            }
                            else {
                                GLuint prog = 0;
                                if (scenario.type == AttackType::CROP_CENTER) prog = atkCrop;
                                else if (scenario.type == AttackType::RESIZE_SCALE) prog = atkResize;
                                else if (scenario.type == AttackType::NOISE_SNP) prog = atkNoise;
                                else if (scenario.type == AttackType::QUANTIZATION) prog = atkQuant;
                                else if (scenario.type == AttackType::GAMMA_CORRECT) prog = atkGamma;
                                else if (scenario.type == AttackType::BRIGHTNESS) prog = atkBright;
                                else if (scenario.type == AttackType::ROTATION) prog = atkRot;

                                Run_Attack_Shader(prog, texMarked, resMgr.tex_Attacked, currentWidth, currentHeight, scenario.param, scenario.type);
                            }
                            glMemoryBarrier(GL_ALL_BARRIER_BITS);

                            // (C) Extraction
                            if (algoIdx == 0) { glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resMgr.buf_Pattern); Run_Legacy_Extraction(extractDctSSProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks); }
                            else if (algoIdx == 1) { glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resMgr.buf_Pattern); Run_Legacy_Extraction(extractDwtSSProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks); }
                            else if (algoIdx == 2 || algoIdx == 3) Run_Extraction_Pipeline(extractSvdProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
                            else if (algoIdx == 4) Run_Full_FFT_Extraction(dftPadProg, dftReorderProg, dftCoreProg, extractDftGlobalProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.tex_FFT_Ping, resMgr.tex_FFT_Pong, resMgr.tex_DFT_Complex, resMgr.buf_ExtractedBits, currentWidth, currentHeight, 32768);
                            else if (algoIdx == 5) Run_Extraction_Pipeline(extractDctProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
                            else if (algoIdx == 6) Run_Extraction_Pipeline(extractDwtProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);
                            else if (algoIdx == 7) Run_Extraction_Pipeline(extractDftProg, resMgr.tex_Source, resMgr.tex_Attacked, resMgr.buf_ExtractedBits, currentWidth, currentHeight, resMgr.numBlocks4x4);

                            glFinish();

                            // (D) Metric Calculation (Recovered BER)

                            // PSNR은 "공격받은 이미지" vs "원본" 비교 (공격 강도 확인용)
                            // 워터마크가 살아있는지는 BER로 확인
                            std::vector<unsigned char> srcPx(currentWidth * currentHeight * 4);
                            std::vector<unsigned char> watermarkedPx(currentWidth * currentHeight * 4);

                            glBindTexture(GL_TEXTURE_2D, resMgr.tex_Source);
                            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, srcPx.data());

                            GLuint texWatermarked = 0;
                            if (algoIdx == 0) texWatermarked = resMgr.tex_Final;
                            else if (algoIdx == 1) texWatermarked = resMgr.tex_DWT_Final;
                            else if (algoIdx == 2 || algoIdx == 3) texWatermarked = resMgr.tex_SVD_Final;
                            else if (algoIdx == 4) texWatermarked = resMgr.tex_DFT_Final;
                            else texWatermarked = resMgr.tex_Opt_Final;

                            glBindTexture(GL_TEXTURE_2D, texWatermarked);
                            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, watermarkedPx.data());


                            // 간략화를 위해 버퍼 읽는 부분만 표시
                            std::vector<uint> orgBits(resMgr.numBlocks4x4);
                            std::vector<uint> extBits(resMgr.numBlocks4x4);
                            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resMgr.buf_Bitstream);
                            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, orgBits.size() * sizeof(uint), orgBits.data());
                            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resMgr.buf_ExtractedBits);
                            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, extBits.size() * sizeof(uint), extBits.data());
                            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

                            // 2. Raw BER 계산 (단순 1:1 비교)
                            // Legacy는 8x8 블록 수만큼, Optimized는 4x4 블록 수만큼 유효
                            size_t validBits = (algoIdx == 0 || algoIdx == 1) ? resMgr.numBlocks : resMgr.numBlocks4x4;
                            float rawBER = CalculateBER(orgBits, extBits, validBits);

                            // Recovered BER 사용
                            int msgLen = 64;
                            std::vector<uint> originalSignature;
                            if (orgBits.size() >= msgLen) originalSignature.assign(orgBits.begin(), orgBits.begin() + msgLen);

                            float recoveredBER = CalculateRecoveredBER(extBits, originalSignature, msgLen);

                            // PSNR 계산 (생략 가능하거나 필요시 추가)
                            // 3. PSNR 계산 (원본 vs 최종)
                            ImageMetrics m = CalculateMetrics(srcPx, watermarkedPx, currentWidth, currentHeight);

                            // CSV 저장을 위해 Alpha 값을 구조체에 저장하는 것이 좋음
                            // 기존 AttackResult 구조체에 'alpha' 필드가 없다면 추가하거나, 'param'에 꼼수를 써야 함.
                            // 제일 좋은 건 CSV 형식을 바꾸는 것.

                            // (E) 저장 (CSV 포맷 변경: Alpha 컬럼 추가)
                            std::ofstream file;
                            // 파일이 없거나 첫 루프면 헤더 생성 (덮어쓰기)
                            if (alpha == 0.0f && algoIdx == 0 && scenario.type == AttackType::NONE && !std::filesystem::exists("Alpha_Attack_Sweep_Results.csv")) {
                                file.open("Alpha_Attack_Sweep_Results.csv", std::ios::out);
                                // ★ 헤더에 PSNR 추가
                                file << "Resolution,Algorithm,Alpha,Attack Type,Parameter,RawBER,RecovBER,PSNR,SSIM\n";
                            }
                            else {
                                file.open("Alpha_Attack_Sweep_Results.csv", std::ios::app); // 이어쓰기
                            }

                            // 데이터 쓰기
                            file << res.name << ","
                                << algoNames[algoIdx] << ","
                                << alpha << ","
                                << scenario.name << ","
                                << scenario.param << ","
                                << rawBER << ","
                                << recoveredBER << ","
                                << m.psnr << ","
                                << m.ssim << "\n"; // ★ PSNR 값 저장

                            file.close();

                            // 콘솔 로그 (진행 상황 확인용)
                            // 너무 많이 출력하면 느리니까 No Attack일 때만 자세히, 나머지는 간략히
                            if (scenario.type == AttackType::NONE) {
                                std::cout << "[" << res.name << "]" << "Algo: " << algoNames[algoIdx] << " | Alpha: " << alpha
                                    << " | PSNR: " << m.psnr << " | SSIM: " << m.ssim << " | RAW BER: " << rawBER << " | BER: " << recoveredBER << std::endl;
                            }
                        }
                    }
                }
            }

            // 끝나면 FHD로 복귀 (UI용)
            resMgr.Resize(1920, 1080, coeffsToUse);
            currentWidth = 1920; currentHeight = 1080;

            g_RunAlphaAttackSweep = false;
            std::cout << "[System] Alpha-Attack Sweep Finished!" << std::endl;
        }

        else {

            if (g_AlgorithmChoice == 0) {
                Run_Compute_Pipeline(dctPass1, dctPass2, dctPass3, dctPass4,
                    resMgr.tex_Source, resMgr.tex_Intermediate, resMgr.tex_DCTOutput, resMgr.tex_Final,
                    resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight,
                    g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
            }
            else if (g_AlgorithmChoice == 1) {
                Run_Compute_Pipeline(dwtPass1, dwtPass2, dwtPass3, dwtPass4,
                    resMgr.tex_Source, resMgr.tex_DWT_Intermediate, resMgr.tex_DWT_Output, resMgr.tex_DWT_Final,
                    resMgr.buf_Bitstream, resMgr.buf_Pattern, currentWidth, currentHeight,
                    g_EnableEmbed, g_EmbeddingStrength, coeffsToUse, resMgr.numBlocks, resMgr.numBlocks);
            }
            else if (g_AlgorithmChoice == 2) { // SVD Block
                // 2. Legacy SVD 파이프라인 실행
                Run_SVD_Pipeline(
                    svd_progs,              // 셰이더 프로그램 배열 (main 상단에 로드됨)
                    svd_progs,               // 텍스처 배열
                    resMgr.buf_Bitstream,   // 워터마크 비트 버퍼
                    resMgr.buf_Pattern,     // 패턴 버퍼
                    currentWidth,
                    currentHeight,
                    g_JacobiIterations,     // 자코비 반복 횟수 (UI 연동)
                    g_SigmaThreshold,       // 임계값 (UI 연동)
                    g_EnableEmbed,          // 워터마크 삽입 여부
                    g_EmbeddingStrength,    // 강도
                    g_CompressionThreshold, // 압축 임계값
                    coeffsToUse,            // 사용 계수 (보통 10)
                    resMgr.numBlocks,       // Legacy는 8x8 블록 개수 사용
                    resMgr.numBlocks        // 비트 길이도 블록 개수와 동일하게 설정
                );
            }
            else if (g_AlgorithmChoice == 3) { // SVD Implicit
                Run_Optimized_OneShot_Pipeline(svdImplictProg, resMgr.tex_Source, resMgr.tex_SVD_Final,
                    resMgr.buf_Bitstream, currentWidth, currentHeight, g_EnableEmbed, g_EmbeddingStrength);
            }
            else if (g_AlgorithmChoice == 4) { // DFT

                Run_Full_FFT_Pipeline(
                    dftPadProg,    // Pad
                    dftReorderProg,
                    dftCoreProg,   // Core FFT
                    dftEmbedProg,  // Embed
                    dftCropProg,   // Crop
                    debugProbeProg,  // Debug Probe (optional)
                    resMgr.tex_Source,    // 원본
                    resMgr.tex_DFT_Final, // 최종 결과 (Resize 함수에 변수명 맞춰주세요. 예: tex_Final)
                    resMgr.tex_FFT_Ping,  // 작업용 1 (2048 size)
                    resMgr.tex_FFT_Pong,  // 작업용 2 (2048 size)
                    resMgr.buf_Bitstream, // 워터마크 비트
                    currentWidth, currentHeight,
                    g_EnableEmbed, g_EmbeddingStrength
                );
            }

            else if (g_AlgorithmChoice == 5) { // DCT Optimized
                Run_Optimized_OneShot_Pipeline(dctOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final,
                    resMgr.buf_Bitstream, currentWidth, currentHeight, g_EnableEmbed, g_EmbeddingStrength);
            }
            else if (g_AlgorithmChoice == 6) { // DWT Optimized
                Run_Optimized_OneShot_Pipeline(dwtOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final,
                    resMgr.buf_Bitstream, currentWidth, currentHeight, g_EnableEmbed, g_EmbeddingStrength);
            }
            
			else if (g_AlgorithmChoice == 7) { // DFT Optimized
                Run_Optimized_OneShot_Pipeline(dftOptProg, resMgr.tex_Source, resMgr.tex_Opt_Final,
                    resMgr.buf_Bitstream, currentWidth, currentHeight, g_EnableEmbed, g_EmbeddingStrength);
            }
            
        }



        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    resMgr.Release();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}