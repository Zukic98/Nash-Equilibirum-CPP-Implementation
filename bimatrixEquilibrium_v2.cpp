#include <iostream>
#include <deque>
#include <limits>
#include <unordered_set>
#include <ctime>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <chrono>
#define DIMENSION 1000

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

std::unordered_set<std::pair<int, int>, PairHash> findNashEquilibirum(int *matrix1, int *matrix2, int dimension){
    
    std::unordered_set<std::pair<int,int>,PairHash> equilibrium;

    for(int i=0; i < dimension; i++){

        int maximumRowValue = -std::numeric_limits<int>::max();

        std::unordered_set<int> best_of_row;

        for(int j=0; j < dimension; j++){

            if(matrix2[i*dimension +j] > maximumRowValue){
                
                best_of_row.clear();

                maximumRowValue = matrix2[i*dimension +j];

                best_of_row.insert(j);

            }else if(matrix2[i*dimension +j] == maximumRowValue){
                
                best_of_row.insert(j);

            }
        }

        for(auto column_index : best_of_row){
            
            int maximumColumnValue = -std::numeric_limits<int>::max();
            std::unordered_set<int> best_of_column;

            for(int k = 0; k < dimension; k++){
                if(matrix1[k*dimension + column_index] > maximumColumnValue){
                
                    maximumColumnValue = matrix1[k*dimension + column_index];

                    best_of_column.insert(k);

                }       
            }

            for(auto row_index:best_of_column){
                if(row_index == i)
                    equilibrium.insert({row_index,column_index});
                
            }

        }

    }

    return equilibrium;
}
/*
std::unordered_set<std::pair<int, int>, PairHash> getPositionsWithBiggestValues(int dimension){
    int min_value = 0;
    int max_value = dimension;
    
    std::unordered_set<std::pair<int, int>, PairHash> indicesOfEquilibirum;
    
    int numberOfRandomNumbers( min_value + rand() % (max_value - min_value + 1) );

    std::cout<<"Indices of equilibrium: "<<std::endl;

    for(int i = 0; i < numberOfRandomNumbers; i++){
        int first(min_value + rand() % (max_value - min_value + 1));
        int second(min_value + rand() % (max_value - min_value + 1));
        indicesOfEquilibirum.insert({first,second});
        //std::cout<< "i: " << first << " j: " << second << std::endl;
    }

    return indicesOfEquilibirum;
}

std::unordered_set<std::pair<int, int>, PairHash> indicesOfEquilibirum(getPositionsWithBiggestValues(DIMENSION));
*/
/*bool compareIndices(int i, int j, std::unordered_set<std::pair<int, int>, PairHash> &indices){
    
    for (auto index : indices) 
        if(i == index.first && j == index.second)
            return true;

    return false;
}*/

int* generateMatrix(int dimension){

    int *matrix = nullptr;

    try{
        matrix = new int[dimension*dimension];
    }catch(std::bad_alloc){
        std::cout<<"Alokacija pala";
        delete[] matrix;
        exit(0);
    }
    // Define the minimum and maximum values for the random number range
    int min_value = 0;
    int max_value = 100;

    //indicesOfEquilibirum(getPositionsWithBiggestValues(dimension));

    for(int i=0; i<dimension; i++)
        for(int j=0; j<dimension; j++)
            matrix[i * dimension + j ] = min_value + rand() % (max_value - min_value + 1);

    return matrix;

}

void printMatrix(int *matrix, int dimension){
    
    for(int i=0; i < dimension; i++){
        
        for(int j=0; j < dimension; j++)
            std::cout<<std::setw(3)<< matrix[i*dimension + j]<<" ";
        
        std::cout<<std::endl;
    }
}

int main(){

    int dimension(DIMENSION);

    srand(time(NULL));
    
    int *matrix1(generateMatrix(dimension));
    std::cout<<"Matrix 1: "<<std::endl;
    //printMatrix(matrix1,dimension);

    std::cout<<"Matrix 2: "<<std::endl;
    int *matrix2(generateMatrix(dimension));
    //printMatrix(matrix2,dimension);
    
    std::cout<<"Equilibrium: "<<std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto equilibrium(findNashEquilibirum(matrix1, matrix2,dimension));

    auto end = std::chrono::high_resolution_clock::now();

    std::cout<< "Execution time of Nash equilibrium: "
        <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    for(auto index : equilibrium){
        std::cout<<"i: "<<index.first<<" j:"<<index.second<<"."<<std::endl;
    }

    delete[] matrix1;
    delete[] matrix2;

    return 0;
}
