#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>

void glfw_error_callback(int error, const char* description)
{
	std::cerr << "GLFW Error (" << error << "): " << description << std::endl;
}

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
	std::cout << "GLFW initialized successfully" << std::endl;

	// Set GLFW window hints for OpenGL version and profile
	const char* glsl_version = "#version 330 core";
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
	std::cout << "GLFW window created successfully" << std::endl;

	// !important
	// Make the OpenGL context binding
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0); // V-Sync Off

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
	// Print OpenGL version
	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

	// [ImGui] Create ImGui Context
	// ---------------------------------------------------
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;

	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Viewport Mode on

	// ImGui Style
	ImGui::StyleColorsDark();

	// 'GLFW' 와 'OpenGL3' imgui 초기화
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	float myParam = 0.5f;
	float color[3] = { 0.0f, 0.5f, 0.5f };

	// Set Viewport
	glViewport(0, 0, 800, 600);


	// C++ �ڵ� ���� ���ڿ��� ���̴��� �����մϴ�.
	// 1. Vertex Shader (GLSL 3.30 Core)
		const char* vertexShaderSource = R"glsl(
		#version 330 core
    
		// �� ���̴��� 1���� '�Է�(in)'�� �޽��ϴ�.
		// layout (location = 0)�� '0�� ����'�̶�� ���Դϴ�. (VAO�� ����� ����)
		layout (location = 0) in vec3 aPos; // "aPos"��� �̸��� 3���� ����(vec3) �Է�

		void main()
		{
			// �Է¹��� aPos�� '�״��' gl_Position�̶�� '���'���� �������ϴ�.
			// gl_Position�� VS�� ���� ���� ��ġ�� ��Ÿ���� '���� ����'�Դϴ�.
			// (x, y, z, w) 4�����̾�� �ϹǷ� vec4�� ��ȯ�մϴ�. w=1.0�� ���� ��ȯ�� �����Դϴ�.
			gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
		}
	)glsl";

		// 2. Fragment Shader (GLSL 3.30 Core)
		const char* fragmentShaderSource = R"glsl(
		#version 330 core
    
		// �� ���̴��� 1���� '���(out)'�� �����ϴ�.
		out vec4 FragColor; // "FragColor"��� �̸��� 4���� ����(RGBA) ���

		void main()
		{
			// �� �ȼ��� '��Ȳ��' (R=1.0, G=0.5, B=0.2)���� ĥ�մϴ�.
			FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
		}
	)glsl";

	// Shader Compilation
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Attach Shader Source Code
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);

	// Compile Shader
	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	// Shader Program Linking
	unsigned int shaderProgram = glCreateProgram();

	// Attach Shaders to the Program
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Link the Shader Program
	glLinkProgram(shaderProgram);

	// Delete Shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	float vertices[] = {
		-0.5f, -0.5f, 0.0f, 
		 0.5f, -0.5f, 0.0f, 
		 0.0f,  0.5f, 0.0f  
	};

	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glEnableVertexAttribArray(0);

	glBindVertexArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

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
		}

		ImGui::End();
		ImGui::Render();

		// Render Color
		glClearColor(color[0], color[1], color[2], 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			// '상태 머신'의 핵심:
			// ImGui가 '다른 창'을 그리면서 OpenGL의 '현재 컨텍스트'를
			// 오염시킬 수 있으므로, '메인 윈도우'의 컨텍스트를 백업합니다.
			GLFWwindow* backup_current_context = glfwGetCurrentContext();

			// ImGui에게 "메인 창 말고, '다른 모든' 플랫폼 창들을 
			// 업데이트하고 그려라"고 명시적으로 명령합니다.
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();

			// '현재 컨텍스트'를 메인 윈도우의 것으로 '복원'합니다.
			// 이래야만 'glfwSwapBuffers'가 올바른 창(메인 윈도우)에서
			// 동작하는 것을 '보장'할 수 있습니다.
			glfwMakeContextCurrent(backup_current_context);
		}

		// Buffer Swap
		glfwSwapBuffers(window);

		
	}

	glfwTerminate();
	return 0;
}