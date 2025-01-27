%%cuda
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <utility>
#include <cstdint>

#define DIMENSION 25000

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

__device__ short atomicAddShort(short* address, short val)

{

    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);

    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;

unsigned int long_old = atomicAdd(base_address, long_val);

    if((size_t)address & 2) {

        return (short)(long_old >> 16);

    } else {

        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

        if (overflow)

            atomicSub(base_address, overflow);

        return (short)(long_old & 0xffff);

    }

}

typedef std::unordered_set<std::pair<short, short>, PairHash> Unordered_pairs;

__global__ void find_max_indices_row_warp(const char* matrix, short* row_max_indices,
                                    short* row_max_indices_count, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows) {
        int max_value = matrix[row * cols];
        row_max_indices[row * cols] = 0;
        row_max_indices_count[row] = 1;

        // Warp-level synchronization
        __syncthreads();

        for (int j = 1; j < cols; ++j) {
            if (matrix[row * cols + j] > max_value) {
                max_value = matrix[row * cols + j];
                row_max_indices[row * cols] = j;
                row_max_indices_count[row] = 1;
            } else if (matrix[row * cols + j] == max_value) {
                // Find the next available index in row_max_indices
                int index = atomicAddShort(&row_max_indices_count[row], 1);
                row_max_indices[row * cols + index] = j;
            }

            // Warp-level synchronization
            __syncthreads();
        }
    }
}

__global__ void find_max_indices_col_warp(const char* matrix, short* col_max_indices,
                                    short* col_max_indices_count, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        int max_value = matrix[col];
        col_max_indices[col * cols] = 0;
        col_max_indices_count[col] = 1;

        // Warp-level synchronization
        __syncthreads();

        for (int i = 1; i < rows; ++i) {
            if (matrix[i * cols + col] > max_value) {
                max_value = matrix[i * cols + col];
                col_max_indices[col * cols] = i;
                col_max_indices_count[col] = 1;
            } else if (matrix[i * cols + col] == max_value) {
                // Find the next available index in col_max_indices
                int index = atomicAdd(&col_max_indices_count[col], 1);
                col_max_indices[col * cols + index] = i;
            }

            // Warp-level synchronization
            __syncthreads();
        }
    }
}

Unordered_pairs findNashEquilibirum(Unordered_pairs unordered_pairs_1,Unordered_pairs unordered_pairs_2 ) {

  Unordered_pairs nashEquilibria;

  // Find the intersection of the two sets
  for (const auto& pair : unordered_pairs_1) {
    if (unordered_pairs_2.count(pair) > 0) {
      nashEquilibria.insert(pair);
    }
  }

  return nashEquilibria;
}

char *generateMatrix(int dimension) {

  char *matrix = nullptr;

  try {
    matrix = new char[dimension * dimension];
  } catch (std::bad_alloc) {
    std::cout << "Alokacija pala";
    delete[] matrix;
    exit(0);
  }
  // Define the minimum and maximum values for the random number range
  int min_value = 0;
  int max_value = 50;

  for (int i = 0; i < dimension; i++)
    for (int j = 0; j < dimension; j++)
      matrix[i * dimension + j] =
          min_value + rand() % (max_value - min_value + 1);

  return matrix;
}

int main() {

  int dimension(DIMENSION);

  char *matrix1(generateMatrix(dimension));
  char *matrix2(generateMatrix(dimension));

  for(int i=0;i<dimension;i++){
      for(int j=0;j<dimension;j++){
          if(i == j){
              matrix1[i*dimension+j] =100;
              matrix2[i*dimension+j] = 100;
          }
          if(i == j && j+1 != dimension){
              matrix1[i*dimension+j+1] = 100;
              matrix2[i*dimension+j+1] = 100;
          }
          if(i == j && j + 2 != dimension){
              matrix1[i*dimension+j+2] = 100;
              matrix2[i*dimension+j+2] = 100;
          }
          //std::cout<<matrix1[i*dimension+j]<<" ";
      }
      //std::cout<<std::endl;
  }

  auto start_seq = std::chrono::high_resolution_clock::now();

  // Allocate device memory
  char *d_matrix1, *d_matrix2;
  short *d_row_max_indices, *d_col_max_indices;
  short *d_row_max_indices_count, *d_col_max_indices_count;
  cudaMalloc(&d_matrix1, dimension * dimension * sizeof(char));
  cudaMalloc(&d_matrix2, dimension * dimension * sizeof(char));
  cudaMalloc(&d_row_max_indices, dimension * dimension * sizeof(short));
  cudaMalloc(&d_col_max_indices, dimension * dimension * sizeof(short));
  cudaMalloc(&d_row_max_indices_count, dimension * sizeof(short));
  cudaMalloc(&d_col_max_indices_count, dimension * sizeof(short));

  // Copy data to device
  cudaMemcpy(d_matrix1, matrix1, dimension * dimension * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix2, matrix2, dimension * dimension * sizeof(char), cudaMemcpyHostToDevice);

  int num_of_threads = 1024;
  int num_of_blocks = (dimension / 1024) + 1;

  // Define grid and block dimensions
  dim3 blockDim_row(1, num_of_threads);
  dim3 gridDim_row(1, num_of_blocks);
  dim3 blockDim_col(num_of_threads, 1);
  dim3 gridDim_col(num_of_blocks, 1);


  // Launch kernels
  find_max_indices_row_warp<<<gridDim_row, blockDim_row>>>(d_matrix1, d_row_max_indices,
                                                     d_row_max_indices_count, dimension, dimension);
  find_max_indices_col_warp<<<gridDim_col, blockDim_col>>>(d_matrix2, d_col_max_indices,
                                                     d_col_max_indices_count, dimension, dimension);



  // Copy results from device to host
  short *h_row_max_indices = (short*)malloc(dimension * dimension * sizeof(short));
  short *h_col_max_indices = (short*)malloc(dimension * dimension * sizeof(short));
  short *h_row_max_indices_count = (short*)malloc(dimension * sizeof(short));
  short *h_col_max_indices_count = (short*)malloc(dimension * sizeof(short));
  cudaMemcpy(h_row_max_indices, d_row_max_indices, dimension * dimension * sizeof(short), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_col_max_indices, d_col_max_indices, dimension * dimension * sizeof(short), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_row_max_indices_count, d_row_max_indices_count, dimension * sizeof(short), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_col_max_indices_count, d_col_max_indices_count, dimension * sizeof(short), cudaMemcpyDeviceToHost);

  // Find Nash equilibria


  Unordered_pairs unordered_pairs_1, unordered_pairs_2;

  // Print row_max_indices
  std::cout << "Row max indices: " << std::endl;

  for (int i = 0; i < dimension; ++i) {

      int num_of_indices = h_row_max_indices_count[i];

      for(int j=0; j < num_of_indices; j++){
        unordered_pairs_1.insert(std::make_pair(i, h_row_max_indices[i*dimension + j]));
        std::cout << "(" << i << ", " << h_row_max_indices[i*dimension + j] << ") ";
      }

  }
  std::cout << std::endl;

  // Print col_max_indices
  std::cout << "Column max indices: " << std::endl;

  for (int i = 0; i < dimension; ++i) {

      int num_of_indices = h_col_max_indices_count[i];

      for(int j=0; j < num_of_indices; j++){

        unordered_pairs_2.insert(std::make_pair(h_col_max_indices[i*dimension + j],i));

        std::cout << "(" << h_col_max_indices[i*dimension+j] << ", " << i << ") ";

      }

  }
  std::cout << std::endl;

  // Find Nash equilibria
  auto nashEquilibria = findNashEquilibirum(unordered_pairs_1,unordered_pairs_2);

  auto end_seq = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);

  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
  std::cout << "Number of Nash equilibria: " << nashEquilibria.size() << std::endl;

  // Print Nash equilibria
  /*std::cout << "Nash equilibria: " << std::endl;
  for (const auto& pair : nashEquilibria) {
      std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
  }*/

  // Free device memory
  cudaFree(d_matrix1);
  cudaFree(d_matrix2);
  cudaFree(d_row_max_indices);
  cudaFree(d_col_max_indices);
  cudaFree(d_row_max_indices_count);
  cudaFree(d_col_max_indices_count);

  // Free host memory
  free(matrix1);
  free(matrix2);
  free(h_row_max_indices);
  free(h_col_max_indices);
  free(h_row_max_indices_count);
  free(h_col_max_indices_count);

  return 0;
}
