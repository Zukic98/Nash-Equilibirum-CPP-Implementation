#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <utility>
#include <iomanip>
#include <cstdint>
#include <fstream>

#define DIMENSION 8000

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

std::unordered_set<std::pair<short, short>, PairHash>
findNashEquilibirum(char *matrix1, char *matrix2, int dimension) {

  std::unordered_set<std::pair<short, short>, PairHash> global_best_of_column,
      global_best_of_row;
  std::unordered_set<std::pair<short, short>, PairHash> intersection;

  for (int i = 0; i < dimension; i++) {

    int maximumColumnValue = -std::numeric_limits<char>::max();
    int maximumRowValue = -std::numeric_limits<char>::max();

    std::unordered_set<std::pair<short, short>, PairHash> best_of_column,
        best_of_row;

    for (int j = 0; j < dimension; j++) {

      if (matrix1[j * dimension + i] > maximumColumnValue) {

        best_of_column.clear();

        maximumColumnValue = matrix1[j * dimension + i];

        best_of_column.insert({j, i});

      } else if (matrix1[j * dimension + i] == maximumColumnValue) {

        best_of_column.insert({j, i});
      }

      if (matrix2[i * dimension + j] > maximumRowValue) {

        best_of_row.clear();

        maximumRowValue = matrix2[i * dimension + j];

        best_of_row.insert({i, j});

      } else if (matrix2[i * dimension + j] == maximumRowValue) {

        best_of_row.insert({i, j});
      }
    }

    global_best_of_column.insert(best_of_column.begin(), best_of_column.end());
    global_best_of_row.insert(best_of_row.begin(), best_of_row.end());
  }

  if (global_best_of_column.size() > global_best_of_row.size()) {
    for (auto element : global_best_of_row) {
      // If insertion is successful, the element exists in both sets
      if (global_best_of_column.insert(element).second == false) {
        intersection.insert(element);
      }
    }
  } else {
    for (auto element : global_best_of_column) {
      // If insertion is successful, the element exists in both sets
      if (global_best_of_row.insert(element).second == false) {
        intersection.insert(element);
      }
    }
  }

  return intersection;
}
/*
std::unordered_set<std::pair<int, int>, PairHash>
getPositionsWithBiggestValues(int dimension){ int min_value = 0; int max_value =
dimension;

    std::unordered_set<std::pair<int, int>, PairHash> indicesOfEquilibirum;

    int numberOfRandomNumbers( min_value + rand() % (max_value - min_value + 1)
);

    std::cout<<"Indices of equilibrium: "<<std::endl;

    for(int i = 0; i < numberOfRandomNumbers; i++){
        int first(min_value + rand() % (max_value - min_value + 1));
        int second(min_value + rand() % (max_value - min_value + 1));
        indicesOfEquilibirum.insert({first,second});
        //std::cout<< "i: " << first << " j: " << second << std::endl;
    }

    return indicesOfEquilibirum;
}

std::unordered_set<std::pair<int, int>, PairHash>
indicesOfEquilibirum(getPositionsWithBiggestValues(DIMENSION));
*/
/*bool compareIndices(int i, int j, std::unordered_set<std::pair<int, int>,
PairHash> &indices){

    for (auto index : indices)
        if(i == index.first && j == index.second)
            return true;

    return false;
}*/

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

  // indicesOfEquilibirum(getPositionsWithBiggestValues(dimension));

  for (int i = 0; i < dimension; i++)
    for (int j = 0; j < dimension; j++)
      matrix[i * dimension + j] =
          min_value + rand() % (max_value - min_value + 1);

  return matrix;
}

void printMatrix(short *matrix, int dimension) {

  for (int i = 0; i < dimension; i++) {

    for (int j = 0; j < dimension; j++)
      std::cout << std::setw(3) << static_cast<int>(matrix[i * dimension + j]) << " ";

    std::cout << std::endl;
  }
}

int main() {
  
  int dimension(DIMENSION);

  std::ofstream outputFile("output.txt");

  for(int i=500;i<=35000;i+=500){
      
  dimension = i;
  srand(time(NULL));

  char *matrix1(generateMatrix(dimension));
  /*std::cout << "Matrix 1: " << std::endl;*/
  //printMatrix(matrix1,dimension);

  char *matrix2(generateMatrix(dimension));
  /*std::cout << "Matrix 2: " << std::endl;*/
  //printMatrix(matrix2,dimension);

  /*std::cout << "Equilibrium: " << std::endl;*/

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
  
  auto start = std::chrono::high_resolution_clock::now();

  auto equilibrium(findNashEquilibirum(matrix1, matrix2, dimension));

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Execution time of Nash equilibrium with " << dimension
            << " size: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  std::cout<<"Number of Nash equilibria: "<<equilibrium.size()<<".";
  /*for(auto index : equilibrium){
      std::cout<<"i: "<<static_cast<int>(index.first)<<" j:"<<static_cast<int>(index.second)<<"."<<std::endl;
  }*/

  delete[] matrix1;
  delete[] matrix2;
  outputFile<< i << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count() << std::endl;
  }
  return 0;
}
