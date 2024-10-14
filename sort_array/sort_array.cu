#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024  // Assume array size is 1024, a power of 2

// CUDA kernel for bitonic sorting step
__global__ void bitonicSortKernel(float *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;  // XOR i with j

    if (ixj > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ixj]) {
                // Swap elements
                float temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        } else {
            if (arr[i] < arr[ixj]) {
                // Swap elements
                float temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

// Host function to call the kernel
void bitonicSort(float *arr) {
    float *d_arr;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_arr, size);
    
    // Copy array from host to device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Launch bitonic sort kernel
    dim3 blocks(N / 1024);
    dim3 threads(1024);

    int j, k;
    for (k = 2; k <= N; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<blocks, threads>>>(d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    // Copy sorted array back to host
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
}

int main() {
    // Create array of random numbers
    float arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
    }

    // Print original array
    printf("Original array:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");

    // Sort the array using Bitonic Sort
    bitonicSort(arr);

    // Print sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");

    return 0;
}
