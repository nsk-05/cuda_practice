#include <stdio.h>
#include <cuda.h>
#include <ctime>
#include "iostream"

void mat_SUB(float* matA, float* matB, float* matC, int ROWsize, int COLsize){
    
    for(int row=0; row<ROWsize; row++){
        for(int col=0; col<COLsize; col++){
                matC[row * ROWsize + col] = matA[row * ROWsize + col] - matB[row * ROWsize + col];
        }
    }
}

__global__ void mat_SUB_cuda(float* matA, float* matB, float* matC, int ROWsize, int COLsize){
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the matrix bounds
    if (row < ROWsize && col < COLsize) {
        int idx = row * COLsize + col; // Compute the linear index for 2D matrix
        matC[idx] = matA[idx] - matB[idx]; // Perform element-wise subtraction
    }
    
}

int main(){

    int ROWsize = 256, COLsize = 256;
    size_t sizeofarray = ROWsize * COLsize * sizeof(float);


    float* matA= (float*)malloc(sizeofarray);
    float* matB= (float*)malloc(sizeofarray);
    float* matC= (float*)malloc(sizeofarray);

    for(int i=0; i< ROWsize * COLsize; i++) {matA[i] = (float)(rand() % 10);}
    for(int i=0; i< ROWsize * COLsize; i++) {matB[i] = (float)(rand() % 10);}

    mat_SUB(matA,matB,matC,ROWsize,COLsize);

    float *matA_gpu,*matB_gpu,*matC_gpu;

    cudaMalloc(&matA_gpu,sizeofarray);
    cudaMalloc(&matB_gpu,sizeofarray);
    cudaMalloc(&matC_gpu,sizeofarray);

    cudaMemcpy(matA_gpu,matA,sizeofarray,cudaMemcpyHostToDevice);
    cudaMemcpy(matB_gpu,matB,sizeofarray,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock_2D(16, 16);
    dim3 blocksPerGrid_2D((ROWsize + 15) / 16, (COLsize + 15) / 16);

    mat_SUB_cuda<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(matA_gpu, matB_gpu, matC_gpu, ROWsize,COLsize);

    cudaMemcpy(matA,matA_gpu,sizeofarray,cudaMemcpyDeviceToHost);
    cudaMemcpy(matB,matB_gpu,sizeofarray,cudaMemcpyDeviceToHost);

    cudaMemcpy(matC, matC_gpu, sizeofarray, cudaMemcpyDeviceToHost);

    // Print the result
    // std::cout << "Matrix C (Result of A - B):\n";
    // for (int i = 0; i < ROWsize; i++) {
    //     for (int j = 0; j < COLsize; j++) {
    //         std::cout << matA[i * COLsize + j] << " " << matB[i * COLsize + j]<< " " << matC[i * COLsize + j] << "          ";
    //     }
    //     std::cout << std::endl;
    // }

    cudaFree(matA_gpu);
    cudaFree(matB_gpu);
    cudaFree(matC_gpu);

    free(matA);
    free(matB);
    free(matC);

}
