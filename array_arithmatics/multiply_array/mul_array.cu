#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>

// CUDA kernel for 2D matrix multiplication
__global__ void MatMul2D(float* A, float* B, float* C, int N, int M, int K) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < N && x < K) {
        float value = 0;
        for (int i = 0; i < M; i++) {
            value += A[y * M + i] * B[i * K + x];
        }
        C[y * K + x] = value;
    }
}

// CUDA kernel for 3D matrix multiplication
__global__ void MatMul3D(float* A, float* B, float* C, int N, int M, int P, int Q) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < P && z < Q) {
        float value = 0;
        for (int i = 0; i < M; i++) {
            value += A[x * M * P + i * P + y] * B[i * P * Q + y * Q + z];
        }
        C[x * P * Q + y * Q + z] = value;
    }
}

// CPU version for 2D matrix multiplication
void MatMul2D_CPU(float* A, float* B, float* C, int N, int M, int K) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float value = 0;
            for (int k = 0; k < M; k++) {
                value += A[i * M + k] * B[k * K + j];
            }
            C[i * K + j] = value;
        }
    }
}

// CPU version for 3D matrix multiplication
void MatMul3D_CPU(float* A, float* B, float* C, int N, int M, int P, int Q) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < P; y++) {
            for (int z = 0; z < Q; z++) {
                float value = 0;
                for (int i = 0; i < M; i++) {
                    value += A[x * M * P + i * P + y] * B[i * P * Q + y * Q + z];
                }
                C[x * P * Q + y * Q + z] = value;
            }
        }
    }
}

int main() {
    int N = 256, M = 256, K = 256, P = 256, Q = 256;  // Example sizes

    // Allocate host memory for 2D matrices
    size_t size_2d_A = N * M * sizeof(float);
    size_t size_2d_B = M * K * sizeof(float);
    size_t size_2d_C = N * K * sizeof(float);

    float *h_A_2D = (float*)malloc(size_2d_A);
    float *h_B_2D = (float*)malloc(size_2d_B);
    float *h_C_2D = (float*)malloc(size_2d_C);
    float *h_C_CPU_2D = (float*)malloc(size_2d_C);  // For CPU result

    // Allocate host memory for 3D matrices
    size_t size_3d_A = N * M * P * sizeof(float);
    size_t size_3d_B = M * P * Q * sizeof(float);
    size_t size_3d_C = N * P * Q * sizeof(float);

    float *h_A_3D = (float*)malloc(size_3d_A);
    float *h_B_3D = (float*)malloc(size_3d_B);
    float *h_C_3D = (float*)malloc(size_3d_C);
    float *h_C_CPU_3D = (float*)malloc(size_3d_C);  // For CPU result

    // Initialize input matrices with random data
    for (int i = 0; i < N * M; i++) h_A_2D[i] = rand() % 100;
    for (int i = 0; i < M * K; i++) h_B_2D[i] = rand() % 100;
    for (int i = 0; i < N * M * P; i++) h_A_3D[i] = rand() % 100;
    for (int i = 0; i < M * P * Q; i++) h_B_3D[i] = rand() % 100;

    // Timing CPU 2D multiplication
    clock_t start_cpu_2d = clock();
    MatMul2D_CPU(h_A_2D, h_B_2D, h_C_CPU_2D, N, M, K);
    clock_t end_cpu_2d = clock();
    double cpu_time_2d = double(end_cpu_2d - start_cpu_2d) / CLOCKS_PER_SEC;
    printf("CPU Time taken for 2D multiplication: %f seconds\n", cpu_time_2d);

    // Timing CPU 3D multiplication
    clock_t start_cpu_3d = clock();
    MatMul3D_CPU(h_A_3D, h_B_3D, h_C_CPU_3D, N, M, P, Q);
    clock_t end_cpu_3d = clock();
    double cpu_time_3d = double(end_cpu_3d - start_cpu_3d) / CLOCKS_PER_SEC;
    printf("CPU Time taken for 3D multiplication: %f seconds\n", cpu_time_3d);

    // Allocate device memory for 2D and 3D matrices
    float *d_A_2D, *d_B_2D, *d_C_2D;
    float *d_A_3D, *d_B_3D, *d_C_3D;
    
    cudaMalloc(&d_A_2D, size_2d_A);
    cudaMalloc(&d_B_2D, size_2d_B);
    cudaMalloc(&d_C_2D, size_2d_C);

    cudaMalloc(&d_A_3D, size_3d_A);
    cudaMalloc(&d_B_3D, size_3d_B);
    cudaMalloc(&d_C_3D, size_3d_C);

    // Copy 2D and 3D matrices to device
    cudaMemcpy(d_A_2D, h_A_2D, size_2d_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_2D, h_B_2D, size_2d_B, cudaMemcpyHostToDevice);

    cudaMemcpy(d_A_3D, h_A_3D, size_3d_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_3D, h_B_3D, size_3d_B, cudaMemcpyHostToDevice);

    // CUDA timing for 2D matrix multiplication
    cudaEvent_t start_2d, stop_2d;
    cudaEventCreate(&start_2d);
    cudaEventCreate(&stop_2d);

    dim3 threadsPerBlock_2D(16, 16);
    dim3 blocksPerGrid_2D((K + 15) / 16, (N + 15) / 16);

    cudaEventRecord(start_2d);
    MatMul2D<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(d_A_2D, d_B_2D, d_C_2D, N, M, K);
    cudaEventRecord(stop_2d);

    cudaEventSynchronize(stop_2d);
    float gpu_time_2d = 0;
    cudaEventElapsedTime(&gpu_time_2d, start_2d, stop_2d);
    printf("GPU Time taken for 2D multiplication: %f milliseconds\n", gpu_time_2d);

    // CUDA timing for 3D matrix multiplication
    cudaEvent_t start_3d, stop_3d;
    cudaEventCreate(&start_3d);
    cudaEventCreate(&stop_3d);

    dim3 threadsPerBlock_3D(8, 8, 8);
    dim3 blocksPerGrid_3D((N + 7) / 8, (P + 7) / 8, (Q + 7) / 8);

    cudaEventRecord(start_3d);
    MatMul3D<<<blocksPerGrid_3D, threadsPerBlock_3D>>>(d_A_3D, d_B_3D, d_C_3D, N, M, P, Q);
    cudaEventRecord(stop_3d);

    cudaEventSynchronize(stop_3d);
    float gpu_time_3d = 0;
    cudaEventElapsedTime(&gpu_time_3d, start_3d, stop_3d);
    printf("GPU Time taken for 3D multiplication: %f milliseconds\n", gpu_time_3d);

    // Cleanup
    free(h_A_2D); free(h_B_2D); free(h_C_2D); free(h_C_CPU_2D);
    free(h_A_3D); free(h_B_3D); free(h_C_3D); free(h_C_CPU_3D);
    cudaFree(d_A_2D); cudaFree(d_B_2D); cudaFree(d_C_2D);
    cudaFree(d_A_3D); cudaFree(d_B_3D); cudaFree(d_C_3D);

    return 0;
}