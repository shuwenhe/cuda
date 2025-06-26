#include <iostream> // 用于标准输入输出
#include <cuda_runtime.h> // 包含 CUDA 运行时 API
#include <string>     // 用于字符串操作

// 错误检查宏，用于 CUDA API 调用
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA 核函数：在 GPU 上执行的函数
// 每个线程将打印一个独特的 "helloN"
__global__ void helloCores(int num_threads) {
    // 计算当前线程的全局唯一索引
    // threadIdx.x: 线程在当前block中的索引
    // blockIdx.x: block在grid中的索引
    // blockDim.x: 每个block中的线程数量
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 只有在索引小于我们希望打印的总线程数时才打印
    if (idx < num_threads) {
        printf("hello%d\n", idx + 1); // 打印 "hello1", "hello2", ...
    }
}

int main() {
    // 1. 查询 CUDA 设备数量
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "错误：未找到任何 CUDA 兼容设备。请确保已安装 NVIDIA GPU 和 CUDA Toolkit。" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "系统中共找到 " << deviceCount << " 个 CUDA 设备。" << std::endl;

    // 2. 选择第一个 CUDA 设备 (通常是默认设备)
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId)); // 明确设置要使用的设备

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

    std::cout << "正在使用设备 " << deviceId << ": " << deviceProp.name << std::endl;
    // 注意：deviceProp.multiProcessorCount 和 deviceProp.coresPerSM
    // 可以用来计算理论上的 CUDA 核心总数，但对于简单的线程打印，我们直接启动线程。
    // printf("CUDA Cores: %d (Approx. calculation based on SMs * CoresPerSM)\n", deviceProp.multiProcessorCount * ???);
    // 实际的 CUDA 核心数量计算比较复杂，因为它取决于具体的架构，这里不直接计算并打印。

    // 3. 定义我们希望启动的逻辑“核心”（线程）的总数
    // 我们可以根据需要调整这个数字。
    // 为了更好地演示并行性，我们通常会启动比物理核心多得多的线程。
    int total_threads_to_launch = 1024 * 32; // 启动 32768 个线程进行演示

    // 4. 定义每个块的线程数量
    // 常见的每个块的线程数量是 256、512 或 1024。
    // 这里我们选择 256，因为这是一个常用的值，且大多数 GPU 支持。
    int threads_per_block = 256;

    // 5. 计算需要的块数量
    // 向上取整以确保所有线程都能被启动
    int num_blocks = (total_threads_to_launch + threads_per_block - 1) / threads_per_block;

    std::cout << "计划启动 " << total_threads_to_launch << " 个逻辑线程。" << std::endl;
    std::cout << "每个块 " << threads_per_block << " 个线程，共 " << num_blocks << " 个块。" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    // 6. 启动 CUDA 核函数
    // <<<num_blocks, threads_per_block>>> 是 CUDA 的核函数启动语法
    helloCores<<<num_blocks, threads_per_block>>>(total_threads_to_launch);

    // 7. 同步设备：等待所有 GPU 上的操作完成
    // 这很重要，以确保所有 printf 输出都已刷新到主机。
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. 检查是否有任何核函数执行时的错误
    CUDA_CHECK(cudaGetLastError());

    std::cout << "---------------------------------------" << std::endl;
    std::cout << "程序执行完毕。" << std::endl;

    return EXIT_SUCCESS;
}

