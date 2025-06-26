// cuda_linear_regression.cu
// A simple linear regression trainer on GPU with CUDA
//
// Compile:
//    nvcc -arch=native cuda_linear_regression.cu -o cuda_linear_regression
//
// Run:
//    ./cuda_linear_regression
//
// The program generates synthetic data y = 2*x + 1 + noise,
// then trains w,b using batch gradient descent on GPU.
// After training, it prints the learned parameters and compares to ground truth.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void compute_gradients(const float* x, const float* y,
                       float w, float b,
                       float* grad_w, float* grad_b,
                       int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float xi = x[idx];
        float yi = y[idx];
        float pred = w * xi + b;
        float err  = pred - yi;
        atomicAdd(grad_w, err * xi);
        atomicAdd(grad_b, err);
    }
}

int main()
{
    // 1. Generate synthetic dataset
    const int N = 1 << 20; // 1,048,576 samples
    std::vector<float> h_x(N);
    std::vector<float> h_y(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_x(0.f, 10.f);
    std::normal_distribution<float> dist_noise(0.f, 0.1f);

    const float TRUE_W = 2.0f;
    const float TRUE_B = 1.0f;

    for (int i = 0; i < N; ++i) {
        float x = dist_x(rng);
        float noise = dist_noise(rng);
        h_x[i] = x;
        h_y[i] = TRUE_W * x + TRUE_B + noise;
    }

    // 2. Allocate device memory
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float *d_grad_w = nullptr, *d_grad_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grad_w, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));

    // 3. Hyperparameters
    const int epochs = 100;
    const float lr = 0.0001f;

    // 4. Initialize parameters
    float w = 0.0f;
    float b = 0.0f;

    // 5. Training loop
    const int blockSize = 256;
    const int gridSize  = (N + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // zero gradients on device
        CUDA_CHECK(cudaMemset(d_grad_w, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));

        // launch kernel
        compute_gradients<<<gridSize, blockSize>>>(d_x, d_y, w, b, d_grad_w, d_grad_b, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy grads back
        float h_grad_w = 0, h_grad_b = 0;
        CUDA_CHECK(cudaMemcpy(&h_grad_w, d_grad_w, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

        // average gradients
        h_grad_w /= N;
        h_grad_b /= N;

        // update parameters
        w -= lr * h_grad_w;
        b -= lr * h_grad_b;

        if ((epoch + 1) % 10 == 0 || epoch == 0)
        {
            printf("Epoch %3d: w=%.4f b=%.4f grad_w=%.4f grad_b=%.4f\n",
                   epoch + 1, w, b, h_grad_w, h_grad_b);
        }
    }

    // 6. Print final parameters
    printf("\nTraining complete.\nLearned parameters: w=%.4f, b=%.4f\n", w, b);
    printf("Ground truth:      w=%.4f, b=%.4f\n", TRUE_W, TRUE_B);

    // cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);

    return 0;
}
