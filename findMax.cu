#include <chrono>
#include <random>
#include <cuda_runtime.h>

__device__ float warpReduceMax(int val)
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
}


int main() {
    const int size = 10;
    int *h_input, *h_maxOut, *h_maxIdxOut;
    int *d_input, *d_maxOut, *d_maxIdxOut;
    int warpSize = 32;

    // Allocate host memory
    h_input = (int*)malloc(size * sizeof(int));
    h_maxOut = (int*)malloc(ceil(size / (float)warpSize) * sizeof(int));
    h_maxIdxOut = (int*)malloc(ceil(size / (float)warpSize) * sizeof(int));

    // Initialize host data with some example values
    for (int i = 0; i < size; ++i) {
        h_input[i] = rand() % 100 - 50; // Generate random numbers between -50 and 49
        printf("%d ",h_input[i]);
    }
    printf("\n");

    // Copy data to device
    cudaMemcpy(d_data, h_data, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((arraySize + blockDim.x - 1) / blockDim.x);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel
    reduceMaxIdxOptimizedBlocks<<<gridDim, blockDim>>>(d_data, arraySize, d_maxOut, d_maxIdxOut);

    // Copy results back to host
    cudaMemcpy(h_maxOut, d_maxOut, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxIdxOut, d_maxIdxOut, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Find the global maximum and its index
    int globalMax = h_maxOut[0];
    int globalMaxIdx = h_maxIdxOut[0];
    for (int i = 1; i < gridDim.x; ++i) {
        if (h_maxOut[i] > globalMax) {
            globalMax = h_maxOut[i];
            globalMaxIdx = h_maxIdxOut[i];
        }
    }

    std::cout << "Global Maximum: " << globalMax << std::endl;
    std::cout << "Global Maximum Index: " << globalMaxIdx << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    // Free memory
    cudaFree(d_data);
    cudaFree(d_maxOut);
    cudaFree(d_maxIdxOut);
    free(h_data);
    free(h_maxOut);
    free(h_maxIdxOut);

    return 0;
}
