#include <cuda_runtime.h>
#include <iostream>
__inline__ __device__ int warpReduceMax(int val)
{
    const unsigned int FULL_MASK = 0xffffffff;
  
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }
      
    return val;
}
  
__inline__ __device__ int warpBroadcast(int val, int predicate)
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
  
    const float warpMax = warpReduceMax(localMax);
  
    const int warpMaxIdx = warpBroadcast(localMaxIdx, warpMax == localMax);
  
    const int lane = threadIdx.x % warpSize;
  
    if (lane == 0)
    {
        int warpIdx = threadIdx.x / warpSize;
        maxOut[warpIdx] = warpMax;
        maxIdxOut[warpIdx] = warpMaxIdx;
    }
}

int main(){
    const int arraySize = 1000000;
    int *h_input, *h_maxOut, *h_maxIdxOut;
    int *d_input, *d_maxOut, *d_maxIdxOut;

    // Allocate host memory
    h_input = (int*)malloc(arraySize * sizeof(int));
    h_maxOut = (int*)malloc(arraySize * sizeof(int)); 
    h_maxIdxOut = (int*)malloc(arraySize * sizeof(int)); 

    // Initialize input array (replace with your actual data)
    for (int i = 0; i < arraySize; ++i) {
        h_input[i] = rand() % 100 - 50; // Example: Random values between -50 and 49
    }
    h_input[arraySize - 1] = 300;
    for (int i = 0; i < arraySize; ++i) {
        printf("%d ",h_input[i]);
    }

    // Allocate device memory
    cudaMalloc(&d_input, arraySize * sizeof(int));
    cudaMalloc(&d_maxOut, arraySize * sizeof(int));
    cudaMalloc(&d_maxIdxOut,  arraySize* sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    const int threadsPerBlock = 1024; 
    const int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock; 

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    reduceMaxIdxOptimizedWarp<<<blocksPerGrid, threadsPerBlock>>>(d_input, arraySize, d_maxOut, d_maxIdxOut);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize to ensure all kernel executions have completed
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back to host
    cudaMemcpy(h_maxOut, d_maxOut, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxIdxOut, d_maxIdxOut, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Find the overall maximum across all warps
    int globalMax = h_maxOut[0];
    int globalMaxIdx = h_maxIdxOut[0];
    for (int i = 1; i < arraySize; ++i) {
        if (h_maxOut[i] > globalMax) {
            globalMax = h_maxOut[i];
            globalMaxIdx = h_maxIdxOut[i];
        }
    }

    printf("Global Maximum: %d\n", globalMax);
    printf("Global Maximum Index: %d\n", globalMaxIdx);
    printf("GPU Execution Time: %f milliseconds\n", milliseconds);

    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_maxOut);
    cudaFree(d_maxIdxOut);
    free(h_input);
    free(h_maxOut);
    free(h_maxIdxOut);

    return 0;
  
}
