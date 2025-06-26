// hello_cuda_fixed.cu
// 编译：nvcc -arch=native hello_cuda_fixed.cu -o hello_cuda_fixed -Wno-deprecated-gpu-targets
// 运行： ./hello_cuda_fixed

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

/* --------- 将 SM 版本映射为 “每个 SM 的 CUDA 核心数” --------- */
int convertSMVer2Cores(int major, int minor)
{
    struct { int sm; int cores; } table[] = {
        {0x89,128}, {0x87,128}, {0x86,128}, {0x80, 64}, // Ada & Ampere
        {0x75, 64}, {0x70, 64},                         // Turing
        {-1, -1}
    };
    int sm = (major << 4) + minor;
    for (int i = 0; table[i].sm != -1; ++i)
        if (table[i].sm == sm) return table[i].cores;
    fprintf(stderr, "⚠️ 未知 SM %d.%d，默认 128 核/SM\n", major, minor);
    return 128;
}

/* --------- GPU Kernel：把线程全局编号写进数组 --------- */
__global__ void writeIndexKernel(int* out, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) out[tid] = tid;   // 0-based 索引
}

int main()
{
    /* 1. 查询 GPU 属性 */
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "无法获取 GPU 属性！\n");
        return -1;
    }
    int coresPerSM = convertSMVer2Cores(prop.major, prop.minor);
    int totalCores = coresPerSM * prop.multiProcessorCount;

    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("共 %d SM，每 SM %d 核，总 CUDA 核心数：%d\n\n",
           prop.multiProcessorCount, coresPerSM, totalCores);

    /* 2. 分配 GPU / CPU 内存 */
    int* d_idx = nullptr;
    cudaMalloc(&d_idx, totalCores * sizeof(int));
    std::vector<int> h_idx(totalCores);

    /* 3. 启动 kernel */
    const int blockSize = 256;
    const int gridSize  = (totalCores + blockSize - 1) / blockSize;
    writeIndexKernel<<<gridSize, blockSize>>>(d_idx, totalCores);

    /* 4. 检错 + 同步 */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_idx);
        return -1;
    }
    cudaDeviceSynchronize();

    /* 5. 拷贝结果回 Host 并顺序打印 */
    cudaMemcpy(h_idx.data(), d_idx, totalCores * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < totalCores; ++i) {
        printf("hello%d\n", h_idx[i] + 1);  // +1 变成 1-based 序号
    }

    cudaFree(d_idx);
    return 0;
}

