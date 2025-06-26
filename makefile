# Makefile for hello_cuda_fixed

# 编译器
NVCC = nvcc

# 编译选项
NVCC_FLAGS = -arch=native -O2 -Wno-deprecated-gpu-targets

# 源文件与目标文件
TARGET = hello
SRC = hello.cu

# 默认目标：编译可执行程序
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# 清理构建产物
clean:
	rm -f $(TARGET)

