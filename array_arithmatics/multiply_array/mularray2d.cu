#include <stdio.h>
#include <cuda.h>
#include <time.h>

// Function to initialize matrices
void initMatrix(float *mat, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

// CPU 2D matrix multiplication
void MatMul2D_CPU(float* A, float* B, float* C, int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// CUDA kernel for 2D matrix multiplication
__global__ void MatMul2D(float* A, float* B, float* C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float value = 0;
        for (int i = 0; i < M; i++) {
            value += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

// GPU 2D matrix multiplication kernel using pitched memory
__global__ void MatMul2D_Pitched(float* A, size_t pitch_A, float* B, size_t pitch_B, float* C, size_t pitch_C, int N, int M, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        float sum = 0;
        float* row_A = (float*)((char*)A + row * pitch_A);
        for (int i = 0; i < M; ++i) {
            float* col_B = (float*)((char*)B + i * pitch_B);
            sum += row_A[i] * col_B[col];
        }
        float* row_C = (float*)((char*)C + row * pitch_C);
        row_C[col] = sum;
    }
}

// Function to compare CPU and GPU performance using pitched memory
void comparePerformancePitched(int N, int M, int P) {
    size_t size2D_CPU = N * M * sizeof(float);
    size_t sizeResult_CPU = N * P * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size2D_CPU);
    float* h_B = (float*)malloc(size2D_CPU);
    float* h_C_CPU = (float*)malloc(sizeResult_CPU);
    float* h_C_GPU = (float*)malloc(sizeResult_CPU);

    // Initialize matrices
    initMatrix(h_A, N, M);
    initMatrix(h_B, M, P);

    // CPU computation
    clock_t cpu_start = clock();
    MatMul2D_CPU(h_A, h_B, h_C_CPU, N, M, P);
    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // GPU computation with pitched memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_A, *d_B, *d_C;
    size_t pitch_A, pitch_B, pitch_C;

    // Allocate pitched memory on the device
    cudaMallocPitch(&d_A, &pitch_A, M * sizeof(float), N);
    cudaMallocPitch(&d_B, &pitch_B, P * sizeof(float), M);
    cudaMallocPitch(&d_C, &pitch_C, P * sizeof(float), N);

    // Copy data to the device
    cudaMemcpy2D(d_A, pitch_A, h_A, M * sizeof(float), M * sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitch_B, h_B, P * sizeof(float), P * sizeof(float), M, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Record the start event
    cudaEventRecord(start);
    
    MatMul2D_Pitched<<<blocksPerGrid, threadsPerBlock>>>(d_A, pitch_A, d_B, pitch_B, d_C, pitch_C, N, M, P);
    
    // Record the stop event
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy2D(h_C_GPU, P * sizeof(float), d_C, pitch_C, P * sizeof(float), N, cudaMemcpyDeviceToHost);

    // Wait for the event to complete
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float gpu_time_pitched = 0;
    cudaEventElapsedTime(&gpu_time_pitched, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,sizeResult_CPU);
    cudaMalloc(&d_b,sizeResult_CPU);
    cudaMalloc(&d_c,sizeResult_CPU);

    cudaMemcpy(&d_a,h_A,sizeResult_CPU,cudaMemcpyHostToDevice);
    cudaMemcpy(&d_b,h_B,sizeResult_CPU,cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    MatMul2D<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, 512, 512, 512);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpu_time=0;
    cudaEventElapsedTime(&gpu_time, start, stop);




    // Print CPU and GPU execution times
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time (Pitched): %f milliseconds\n", gpu_time_pitched);
    printf("GPU Time : %f milliseconds\n", gpu_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    int N = 512, M = 512, P = 480;

    printf("Comparing 2D Matrix Multiplication with Pitched Memory (CPU vs GPU)...\n");
    comparePerformancePitched(N, M, P);

    return 0;
}
