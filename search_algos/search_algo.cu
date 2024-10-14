#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda.h>
#include <vector>


__global__ void searchNumber(float *arr, int *found_idx, float search_val, int *found_flag, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        // Check if the number is already found by any other thread
        if (atomicAdd(found_flag, 0) == 1) {
            return; // Exit early if the number is already found
        }

        // If current thread finds the search value
        if (arr[idx] == search_val) {
            // Store the index of the found element
            *found_idx = idx;
            // Set the found flag to signal that the search is over
            atomicExch(found_flag, 1);
        }
    }
}

int main(){

    std::vector<float> h_numbers;
    std::ifstream inFile("random_numbers.txt");

    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open the file!" << std::endl;
        return 1;
    }

    float num;
    while (inFile >> num) {
        h_numbers.push_back(num);
    }
    inFile.close();

    int N = h_numbers.size();

    float search_value;
    std::cout << "Enter the number to search for: ";
    std::cin >> search_value;

    int h_found_idx = -1;
    int h_found_flag = 0;

    float *d_numbers;
    int *d_found_idx;
    int *d_found_flag;
    cudaMalloc((void **)&d_numbers, N * sizeof(float));
    cudaMalloc((void **)&d_found_idx, sizeof(int));
    cudaMalloc((void **)&d_found_flag, sizeof(int));

    cudaMemcpy(d_numbers,h_numbers.data(),N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_idx,&h_found_idx,sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_flag,&h_found_flag,sizeof(float),cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    searchNumber<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_found_idx, search_value, d_found_flag, N);

    cudaDeviceSynchronize();
    // Copy the result back to the host
    cudaMemcpy(&h_found_idx, d_found_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

    // Check if the number was found
    if (h_found_flag == 1) {
        std::cout << "Number " << search_value << " found at index " << h_found_idx << std::endl;
    } else {
        std::cout << "Number " << search_value << " not found in the array." << std::endl;
    }

    // Free the device memory
    cudaFree(d_numbers);
    cudaFree(d_found_idx);
    cudaFree(d_found_flag);

    return 0;
}
