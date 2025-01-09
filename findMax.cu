
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <chrono>
#include <random>
#include <iostream>

/*
__device__ float warpReduceMax(float val)
{
    const unsigned int FULL_MASK = 0xffffffff;

    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }

    return val;
}

__device__ int warpBroadcast(int val, int predicate)
{
    const unsigned int FULL_MASK = 0xffffffff;

    unsigned int mask = __ballot_sync(FULL_MASK, predicate);

    int lane = 0;
    for (;!(mask & 1); ++lane)
    {
        mask >>= 1;
    }

    return __shfl_sync(FULL_MASK, val, lane);
}

__global__ void reduceMaxIdxOptimizedWarp(const int* __restrict__ input, const int size, int* maxOut, int* maxIdxOut)
{
    int localMax = 0;
    int localMaxIdx = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        int val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }

    const int warpMax = warpReduceMax(localMax);

    const int warpMaxIdx = warpBroadcast(localMaxIdx, warpMax == localMax);

    const int lane = threadIdx.x % warpSize;

    if (lane == 0)
    {
        int warpIdx = threadIdx.x / warpSize;
        maxOut[warpIdx] = warpMax;
        maxIdxOut[warpIdx] = warpMaxIdx;
    }
}*/

int main() {
    // Define the size of the input array
    const int size = 1024;

    // Allocate host memory for input, output max, and output max index
    int* h_input;
    int* h_maxOut;
    int* h_maxIdxOut;
    cudaMallocHost((void**)&h_input, size * sizeof(int));
    cudaMallocHost((void**)&h_maxOut, gridDim.x * sizeof(int));
    cudaMallocHost((void**)&h_maxIdxOut, gridDim.x * sizeof(int));

    // Initialize input data (replace with your actual data)
    for (int i = 0; i < size; ++i) {
        h_input[i] = rand() % 100 - 50; // Example: Random values between -50 and 49
    }

    // Allocate device memory
    int* d_input;
    int* d_maxOut;
    int* d_maxIdxOut;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_maxOut, gridDim.x * sizeof(int));
    cudaMalloc((void**)&d_maxIdxOut, gridDim.x * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(128); // Adjust grid size as needed
    dim3 blockDim(32); // Adjust block size as needed

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch the kernel
    //reduceMaxIdxOptimizedWarp <<<gridDim, blockDim>>> (d_input, size, d_maxOut, d_maxIdxOut);

    // Synchronize the device
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    
    
    // Copy output data from device to host
    cudaMemcpy(h_maxOut, d_maxOut, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxIdxOut, d_maxIdxOut, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);

    // Find the global maximum and its index
    int globalMax = h_maxOut[0];
    int globalMaxIdx = h_maxIdxOut[0];
    for (int i = 1; i < gridDim.x; ++i) {
        if (globalMax < h_maxOut[i]) {
            globalMax = h_maxOut[i];
            globalMaxIdx = h_maxIdxOut[i];
        }
    }

    // Print the results
    printf("Global Maximum: %d\n", globalMax);
    printf("Global Maximum Index: %d\n", globalMaxIdx);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_maxOut);
    cudaFree(d_maxIdxOut);
    cudaFreeHost(h_input);
    cudaFreeHost(h_maxOut);
    cudaFreeHost(h_maxIdxOut);

    return 0;
}
