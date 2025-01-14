#include <iostream>
#include <deque>
#include <limits>
#include <unordered_set>
#include <ctime>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <fstream>

#define DIMENSION 3000
#define NUM_THREADS 8 // Adjust based on your system

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

// Shared data structures for thread synchronization
std::mutex mtx;
std::condition_variable cv;
std::vector<std::unordered_set<std::pair<int, int>, PairHash>> best_of_column_per_thread(NUM_THREADS);
std::vector<std::unordered_set<std::pair<int, int>, PairHash>> best_of_row_per_thread(NUM_THREADS);

void worker(int thread_id, int *matrix1, int *matrix2, int dimension) {
    for (int i = thread_id; i < dimension; i += NUM_THREADS) {
        int maximumColumnValue = -std::numeric_limits<int>::max();
        int maximumRowValue = -std::numeric_limits<int>::max();

        std::unordered_set<std::pair<int, int>, PairHash> best_of_column, best_of_row;

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

        // Store results in thread-specific data structures
        best_of_column_per_thread[thread_id] = best_of_column;
        best_of_row_per_thread[thread_id] = best_of_row;
    }

    // Signal main thread that this worker has finished
    std::unique_lock<std::mutex> lock(mtx);
    cv.notify_one();
}

std::unordered_set<std::pair<int, int>, PairHash> findNashEquilibirum(int *matrix1, int *matrix2, int dimension) {
    std::vector<std::thread> threads;
    std::unordered_set<std::pair<int, int>, PairHash> global_best_of_column, global_best_of_row;
    std::unordered_set<std::pair<int, int>, PairHash> intersection;

    // Create and start threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker, i, matrix1, matrix2, dimension);
    }

    // Wait for all threads to finish
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return best_of_column_per_thread.size() == NUM_THREADS && best_of_row_per_thread.size() == NUM_THREADS; });
    }

    // Combine results from all threads
    for (const auto& best_of_column : best_of_column_per_thread) {
        global_best_of_column.insert(best_of_column.begin(), best_of_column.end());
    }
    for (const auto& best_of_row : best_of_row_per_thread) {
        global_best_of_row.insert(best_of_row.begin(), best_of_row.end());
    }

    // Find intersection
    if (global_best_of_column.size() > global_best_of_row.size()) {
        for (auto element : global_best_of_row) {
            if (global_best_of_column.count(element) > 0) {
                intersection.insert(element);
            }
        }
    } else {
        for (auto element : global_best_of_column) {
            if (global_best_of_row.count(element) > 0) {
                intersection.insert(element);
            }
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return intersection;
}


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

    std::ofstream outputFile("resultsOfExecution.txt");

    for(int i = 4; i<30000; i++){
    int dimension(i);

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

    outputFile << dimension<<","<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<"\n";

    /*for(auto index : equilibrium){
        std::cout<<"i: "<<index.first<<" j:"<<index.second<<"."<<std::endl;
    }*/

    delete[] matrix1;
    delete[] matrix2;
    }

    outputFile.close();

    return 0;
}

// Rest of the code (generateMatrix, printMatrix, main) remains the same
