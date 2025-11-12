#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

void framebuffer_size_callback([[maybe_unused]]GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

int main()
{
	// Initialize GLFW
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	// Set GLFW window hints for OpenGL version and profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // For MacOS
#endif

	// Create a GLFW window
	GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Watermark", nullptr, nullptr);
	if (!window)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	// !important
	// Make the OpenGL context binding
	glfwMakeContextCurrent(window);

	// Set the framebuffer size callback
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Initialize GLAD
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
	{
		std::cerr << "Failed to initialize GLAD" << std::endl;
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	// Set Viewport
	glViewport(0, 0, 800, 600);


	// C++ 코드 내에 문자열로 셰이더를 정의합니다.
	// 1. Vertex Shader (GLSL 3.30 Core)
		const char* vertexShaderSource = R"glsl(
		#version 330 core
    
		// 이 셰이더는 1개의 '입력(in)'을 받습니다.
		// layout (location = 0)은 '0번 슬롯'이라는 뜻입니다. (VAO와 연결될 지점)
		layout (location = 0) in vec3 aPos; // "aPos"라는 이름의 3차원 벡터(vec3) 입력

		void main()
		{
			// 입력받은 aPos를 '그대로' gl_Position이라는 '출력'으로 내보냅니다.
			// gl_Position은 VS의 최종 정점 위치를 나타내는 '내장 변수'입니다.
			// (x, y, z, w) 4차원이어야 하므로 vec4로 변환합니다. w=1.0은 투영 변환을 위함입니다.
			gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
		}
	)glsl";

		// 2. Fragment Shader (GLSL 3.30 Core)
		const char* fragmentShaderSource = R"glsl(
		#version 330 core
    
		// 이 셰이더는 1개의 '출력(out)'을 가집니다.
		out vec4 FragColor; // "FragColor"라는 이름의 4차원 벡터(RGBA) 출력

		void main()
		{
			// 이 픽셀을 '주황색' (R=1.0, G=0.5, B=0.2)으로 칠합니다.
			FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
		}
	)glsl";

	// [이 코드는 main() 함수 시작 부분, '초기화' 시 1번만 실행됩니다]

	// 1. 셰이더 객체 생성
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// 2. 셰이더 소스 코드(C++ 문자열)를 셰이더 객체에 연결
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);

	// 3. ★ 드라이버가 GLSL을 GPU 기계어로 컴파일 ★
	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);
	// (실제 프로덕션 코드에서는 여기서 컴파일 성공 여부를 glGetShaderiv로 확인해야 합니다)

	// 4. '셰이더 프로그램' 객체 생성 (VS와 FS를 담을 '통')
	unsigned int shaderProgram = glCreateProgram();

	// 5. 컴파일된 셰이더들을 '프로그램'에 부착(Attach)
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// 6. ★ 셰이더들을 '링크(Link)' ★
	// VS의 'out'과 FS의 'in'을 연결하고, 최종 GPU 실행 코드를 생성합니다.
	glLinkProgram(shaderProgram);
	// (마찬가지로 glGetProgramiv로 링크 성공 여부 확인 필요)

	// 7. 원본 셰이더 객체들은 '프로그램'에 묶였으므로 더 이상 필요 없습니다.
	// 드라이버 메모리에서 해제합니다.
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// 1. 원자재 (CPU RAM에 정의)
	// 삼각형 정점 3개 (x, y, z 좌표)
	float vertices[] = {
		-0.5f, -0.5f, 0.0f, // 정점 1 (왼쪽 아래)
		 0.5f, -0.5f, 0.0f, // 정점 2 (오른쪽 아래)
		 0.0f,  0.5f, 0.0f  // 정점 3 (위쪽 중앙)
	};

	// 2. VBO와 VAO를 위한 핸들(ID) 생성
	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// 3. ★★★ '조립 설명서(VAO)' 기록 시작 ★★★
	// "지금부터의 모든 버퍼/속성 설정은 이 VAO에 저장하라"
	glBindVertexArray(VAO);

	// 4. '원자재(VBO)'를 VRAM에 업로드

	// (1) VBO를 '현재 작업 대상'으로 바인딩 (상태 머신)
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// (2) ★ CPU -> GPU로 데이터 전송 (가장 비싼 초기화 단계) ★
	// CPU 메모리(vertices)의 내용을 GPU VRAM(VBO)으로 복사합니다.
	// GL_STATIC_DRAW: 이 데이터는 '거의 변하지 않음'을 드라이버에게 알리는 힌트.
	// 드라이버는 이 힌트를 보고 GPU가 가장 읽기 빠른 VRAM 영역에 데이터를 배치합니다.
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// 5. '조립 설명서'에 내용 기입 (VAO에 저장됨)
	// "셰이더의 'location = 0'번 입력 슬롯에 VBO의 데이터를 이렇게 연결하라"

	// (1) 어떤 데이터를(VBO): 현재 바인딩된 GL_ARRAY_BUFFER (즉, VBO)
	// (2) 어떻게 해석할지:
	//    - 0: 'location = 0' 슬롯에
	//    - 3: 3개씩 (vec3)
	//    - GL_FLOAT: float 타입으로
	//    - GL_FALSE: 정규화하지 않음
	//    - 3 * sizeof(float): 한 정점의 총 크기 (Stride)
	//    - (void*)0: 이 데이터의 시작 오프셋 (Offset)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	// "이 '0번 슬롯'을 활성화하라"
	glEnableVertexAttribArray(0);

	// 6. ★★★ '조립 설명서(VAO)' 기록 완료 ★★★
	// 다른 코드에 의해 VAO 설정이 '오염'되는 것을 막기 위해 바인딩을 해제합니다.
	glBindVertexArray(0);

	// (VBO 바인딩도 해제해도 되지만, VAO가 0번 슬롯의 VBO 바인딩을 기억하고 있습니다)
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	while (!glfwWindowShouldClose(window))
	{
		// Input
		// glfwGetKey...

		// Render
		glClearColor(0.2f, 0.3f, 0.3f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		// --- ★★★ 효율적인 그리기 커맨드 (단 3줄) ★★★ ---

		// (1) 사용할 '설계도(Program)'를 상태 머신에 설정
		// 드라이버는 'shaderProgram' ID에 해당하는 GPU 기계어 코드를 
		// 다음 그리기 명령에서 사용하도록 파이프라인에 연결합니다. (포인터 스왑)
		glUseProgram(shaderProgram);

		// (2) 사용할 '조립 설명서(VAO)'를 상태 머신에 설정
		// ★ 가장 큰 오버헤드 절감 ★
		// 이 한 줄이, 우리가 초기화 때 설정한 모든 것
		// (glBindBuffer(VBO), glVertexAttribPointer, glEnableVertexAttribArray)
		// 을 '한 번에' 드라이버에게 상기시킵니다.
		// VAO가 없었다면 이 모든 함수를 여기서 매번 다시 호출해야 했습니다.
		glBindVertexArray(VAO);

		// (3) ★ '점화 스위치' (Draw Call) ★
		// "현재 바인딩된 '설계도(Program)'와 '조립 설명서(VAO)'를 사용해서,
		//  0번 정점부터 총 3개의 정점을 그려라!"
		//
		// 이 명령은 CPU에서 드라이버로, 드라이버에서 커널을 통해
		// GPU의 커맨드 큐(Command Queue)로 전송됩니다.
		// 이 커맨드 자체는 수십 바이트(Byte)에 불과한 '작업 지시서'입니다.
		// 실제 데이터(VBO)는 이미 VRAM에 있습니다.
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// Buffer Swap
		glfwSwapBuffers(window);

		// Get event(Keyboard, Mouse input, window closing etc.) from OS
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}