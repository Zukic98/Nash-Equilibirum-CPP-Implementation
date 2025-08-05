#ifndef BIMATRIX_NASH_EQUILIBRIUM
#define BIMATRIX_NASH_EQUILIBRIUM

#include <iostream>
#include <deque>
#include <limits>
#include <unordered_set>
#include <ctime>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <chrono>

typedef std::deque<std::deque<int>> Matrix;

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

std::unordered_set<std::pair<int, int>, PairHash> findNashEquilibirum(Matrix &matrix1, Matrix &matrix2){
    
    std::unordered_set<std::pair<int,int>,PairHash> best_of_column;
    // best_of_row;
    std::unordered_set<std::pair<int,int>,PairHash> global_best_of_column, global_best_of_row;
    std::unordered_set<std::pair<int,int>,PairHash> intersection;

    int dimension(matrix1.size());

    for(int i=0; i < dimension; i++){

        int maximumColumnValue = -std::numeric_limits<int>::max();
        int maximumRowValue = -std::numeric_limits<int>::max();

        std::unordered_set<std::pair<int,int>, PairHash> best_of_column, best_of_row;

        for(int j=0; j < dimension; j++){
            
            if(matrix1[j][i] > maximumColumnValue){
                
                maximumColumnValue = matrix1[j][i];

                best_of_column.clear();

                best_of_column.insert({j,i});

            }else if(matrix1[j][i] == maximumColumnValue){
                
                best_of_column.insert({j,i});

            }

            if(matrix2[i][j] > maximumRowValue){
                
                maximumRowValue = matrix2[i][j];

                best_of_row.clear();

                best_of_row.insert({i,j});

            }else if(matrix2[i][j] == maximumRowValue){
                
                best_of_row.insert({i,j});

            }
        }

        global_best_of_column.insert(best_of_column.begin(),best_of_column.end());
        global_best_of_row.insert(best_of_row.begin(),best_of_row.end());

    }

    std::set_intersection(global_best_of_column.begin(), global_best_of_column.end(), 
    global_best_of_row.begin(), global_best_of_row.end(), std::inserter(intersection, intersection.begin()));

    return intersection;
}


#endif