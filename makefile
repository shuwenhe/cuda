# Makefile for compiling all .cu files

# 编译器
NVCC = nvcc

# 编译选项
NVCC_FLAGS = -arch=native -O2 -Wno-deprecated-gpu-targets

# 查找当前目录下所有 .cu 文件
SRC = $(wildcard *.cu)

# 将源文件名 (.cu) 转换为目标文件名 (.out)
TARGET = $(SRC:.cu=.out)

# 默认目标
all: $(TARGET)

# 编译规则
%.out: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# 清理目标
clean:
	rm -f $(TARGET)

