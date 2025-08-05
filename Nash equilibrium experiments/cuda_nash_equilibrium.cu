#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#define DIMENSION 1000

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

typedef std::unordered_set<std::pair<int, int>, PairHash> Unordered_pairs;

__global__ void find_max_indices(const int* matrix1, const int* matrix2, int* row_max_indices,
                                int* row_max_indices_count, int* col_max_indices,
                                int* col_max_indices_count, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows) {

        int max_value = matrix1[row*rows];
        row_max_indices[row*rows] = 0;
        row_max_indices_count[row] = 1;

        for (int j = 1; j < cols; ++j) {

            if (matrix1[row*rows+j] > max_value) {

                max_value = matrix1[row*rows+j];
                row_max_indices[row*rows] = j;
                row_max_indices_count[row] = 1;

            } else if (matrix1[row*rows+j] == max_value) {

                row_max_indices[row*rows + row_max_indices_count[row]++] = j;

            }

        }

    }

    if (col < cols) {

        int max_value = matrix2[col];
        col_max_indices[col*cols] = 0;
        col_max_indices_count[col] = 1;

        for (int i = 1; i < rows; ++i) {

            if (matrix2[i*cols + col] > max_value) {

                max_value = matrix2[i*cols+col];
                col_max_indices[col*cols] = i;
                col_max_indices_count[col] = 1;

            } else if (matrix2[i*cols + col] == max_value) {

                col_max_indices[col*cols + col_max_indices_count[col]++] = i;

            }
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

int *generateMatrix(int dimension) {

  int *matrix = nullptr;

  try {
    matrix = new int[dimension * dimension];
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

  srand(time(NULL));

  int *matrix1(generateMatrix(dimension));
  int *matrix2(generateMatrix(dimension));

  for(int i=0;i<dimension;i++){
      for(int j=0;j<dimension;j++){
          if(i == j){
              matrix1[i*dimension+j] =1000;
              matrix2[i*dimension+j] = 1000;
          }
          if(i == j && j+1 != dimension){
              matrix1[i*dimension+j+1] = 1000;
              matrix2[i*dimension+j+1] = 1000;
          }
          if(i == j && j + 2 != dimension){
              matrix1[i*dimension+j+2] = 1000;
              matrix2[i*dimension+j+2] = 1000;
          }
          std::cout<<matrix1[i*dimension+j]<<" ";
      }
      std::cout<<std::endl;
  }


  // Allocate device memory
  int *d_matrix1, *d_matrix2, *d_row_max_indices, *d_col_max_indices;
  int *d_row_max_indices_count, *d_col_max_indices_count;
  cudaMalloc(&d_matrix1, dimension * dimension * sizeof(int));
  cudaMalloc(&d_matrix2, dimension * dimension * sizeof(int));
  cudaMalloc(&d_row_max_indices, dimension * dimension * sizeof(int));
  cudaMalloc(&d_col_max_indices, dimension * dimension * sizeof(int));
  cudaMalloc(&d_row_max_indices_count, dimension * sizeof(int));
  cudaMalloc(&d_col_max_indices_count, dimension * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_matrix1, matrix1, dimension * dimension * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix2, matrix2, dimension * dimension * sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(16, 1024);
  dim3 gridDim((dimension + blockDim.x - 1) / blockDim.x, (dimension + blockDim.y - 1) / blockDim.y);

  // Measure execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventRecord(start);

  // Launch kernel
  find_max_indices<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_row_max_indices,
                                        d_row_max_indices_count, d_col_max_indices,
                                        d_col_max_indices_count, dimension, dimension);

  // Record stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  // Copy results from device to host
  int *h_row_max_indices = (int*)malloc(dimension * dimension * sizeof(int));
  int *h_col_max_indices = (int*)malloc(dimension * dimension * sizeof(int));
  int *h_row_max_indices_count = (int*)malloc(dimension * sizeof(int));
  int *h_col_max_indices_count = (int*)malloc(dimension * sizeof(int));
  cudaMemcpy(h_row_max_indices, d_row_max_indices, dimension * dimension * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_col_max_indices, d_col_max_indices, dimension * dimension * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_row_max_indices_count, d_row_max_indices_count, dimension * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_col_max_indices_count, d_col_max_indices_count, dimension * sizeof(int), cudaMemcpyDeviceToHost);

  auto start_seq = std::chrono::high_resolution_clock::now();

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

  std::cout << "Execution time: " << milliseconds + duration.count() << " ms" << std::endl;
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