#include <iostream>
#include <deque>
#include <limits>
#include <unordered_set>
#include <ctime>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <chrono>
#define DIMENSION 10000

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

std::unordered_set<std::pair<int, int>, PairHash> findNashEquilibirum(int *matrix1, int *matrix2, int dimension){
    
    std::unordered_set<std::pair<int,int>,PairHash> global_best_of_column, global_best_of_row;
    std::unordered_set<std::pair<int,int>,PairHash> intersection;

    for(int i=0; i < dimension; i++){

        int maximumColumnValue = -std::numeric_limits<int>::max();
        int maximumRowValue = -std::numeric_limits<int>::max();

        std::unordered_set<std::pair<int,int>, PairHash> best_of_column, best_of_row;

        for(int j=0; j < dimension; j++){
            
            if(matrix1[j*dimension + i] > maximumColumnValue){
                
                best_of_column.clear();

                maximumColumnValue = matrix1[j*dimension + i];

                best_of_column.insert({j,i});

            }else if(matrix1[j*dimension + i] == maximumColumnValue){
                
                best_of_column.insert({j,i});

            }

            if(matrix2[i*dimension +j] > maximumRowValue){
                
                best_of_row.clear();

                maximumRowValue = matrix2[i*dimension +j];

                best_of_row.insert({i,j});

            }else if(matrix2[i*dimension +j] == maximumRowValue){
                
                best_of_row.insert({i,j});

            }
        }

        global_best_of_column.insert(best_of_column.begin(),best_of_column.end());
        global_best_of_row.insert(best_of_row.begin(),best_of_row.end());

    }
    
    if(global_best_of_column.size() > global_best_of_row.size()){
        for (auto element : global_best_of_row) {
            // If insertion is successful, the element exists in both sets
            if (global_best_of_column.insert(element).second == false) {
                intersection.insert(element);
            }
        }
    }else{
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

    /*for(auto index : equilibrium){
        std::cout<<"i: "<<index.first<<" j:"<<index.second<<"."<<std::endl;
    }*/

    delete[] matrix1;
    delete[] matrix2;

    return 0;
}
