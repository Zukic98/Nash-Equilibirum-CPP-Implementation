#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>

#define RAND_MAX 1
#define MAX_PLAYERS 2
#define MAX_STRATEGIES 10

struct IndexNode{
    int i;
    int j;
    struct IndexNode *next;
    struct IndexNode *previous;
};

struct IndexNode *createNode(int i, int j){
    
    struct IndexNode *root =  (struct IndexNode*)malloc(sizeof(struct IndexNode));
    
    root -> i = i;
    root -> j = j;
    root -> next = NULL;
    root -> previous = NULL;
    
    return root;
}

struct IndexNode *deleteList(struct IndexNode *root){
    
    struct IndexNode *iterator = root;
    
    while(iterator != NULL){
        root = root -> next;
        free(iterator);
        iterator = root;
    }

    return NULL;
}

struct IndexNode *findNashEquilibirum(double **matrix1, double **matrix2, int dimension){
    
    struct IndexNode *root_of_best_columns = NULL;
    struct IndexNode *iterator_of_columns = NULL;

    struct IndexNode *best_of_matrix1 = NULL;
    struct IndexNode *end_of_best1 = NULL;
    struct IndexNode *best_of_matrix2 = NULL;
    struct IndexNode *end_of_best2 = NULL;
    
    struct IndexNode *root_of_best_rows = NULL;
    struct IndexNode *iterator_of_rows = NULL;

    for(int i=0; i<dimension; i++){

        int maximumColumnValue = DBL_MIN;
        int maximumRowValue = DBL_MIN;

        root_of_best_rows = NULL;
        iterator_of_rows = NULL;

        root_of_best_columns = NULL;
        iterator_of_columns = NULL;


        for(int j=0; j < dimension; j++){
                        
            if(matrix1[j][i] > maximumColumnValue){
                
                maximumColumnValue = matrix1[j][i];

                root_of_best_columns = deleteList(root_of_best_columns);

                root_of_best_columns = createNode(j,i);

                iterator_of_columns = root_of_best_columns;

            }else if(matrix1[j][i] == maximumColumnValue){
            
                iterator_of_columns -> next =  createNode(j,i);

                iterator_of_columns -> previous = iterator_of_columns -> next -> previous;

                iterator_of_columns = iterator_of_columns -> next;

            }

            if(matrix2[i][j] > maximumRowValue){
                
                maximumRowValue = matrix1[i][j];

                root_of_best_rows = deleteList(root_of_best_rows);

                root_of_best_rows = createNode(i,j);

                iterator_of_columns = root_of_best_rows;

            }else if(matrix2[i][j] == maximumRowValue){
            
                iterator_of_columns -> next =  createNode(i,j);

                iterator_of_rows = iterator_of_rows -> next;

            }

        }

        if(best_of_matrix1 != NULL){
            
            end_of_best1 -> next = root_of_best_columns;
            end_of_best1 = iterator_of_columns;

        }else{
            
            best_of_matrix1 = root_of_best_columns;
            end_of_best1 = iterator_of_columns;
        }

        if(best_of_matrix2 != NULL){
            
            end_of_best2 -> next = root_of_best_rows;
            end_of_best2 = iterator_of_rows;

        }else{
            
            best_of_matrix2 = root_of_best_rows;
            end_of_best2 = iterator_of_rows;
        }

    }

    // Find intersection between best cells
    for(struct IndexNode *it1 = best_of_matrix1; it1 != NULL;){
        
        int exist = 0;
        struct IndexNode *tmp = NULL;

        for(struct IndexNode *it2 = best_of_matrix2; it2 != NULL; it2 = it2 -> next){
            if(it1 -> i == it2 -> i && it1 -> j == it2 -> j){
                exist = 1;
                break;
            }
        }

        if(!exist){
            
            it1 -> previous -> next = it1 -> next;
            tmp = it1;
            it1 = it1 -> next;
            free(tmp);

        }else{

            it1 = it1 -> next;
            
        }

    }

    free(best_of_matrix2);

    return best_of_matrix1;
}

double **genereateMatrix(int dimension){

    // Generate a random double between 0.0 (inclusive) and 1.0 (exclusive)
    double random_double = (double)rand() / RAND_MAX;

    // Generate a random double between a specific range (e.g., -10.0 to 10.0)
    double min = DBL_MIN, max = DBL_MAX;
    random_double = min + random_double * (max - min);

    double **matrix = (double**)malloc(dimension * sizeof(double*));
    for(int i=0;i<dimension;i++)
        matrix[i] = (double*)malloc(dimension * sizeof(double*));
    
    for(int i=0; i<dimension; i++)
        for(int j=0; j<dimension; j++)
            matrix[i][j] = min + random_double * (max - min);
    
    return matrix;

}

int main(){
    int dimension = 100;
    srand(time(NULL)); // Seed the random number generator
    
    double **matrix = genereateMatrix(dimension);

    for(int i=0; i < dimension; i++)
        for(int j=0; j < dimension; j++)
            print(matrix[i][j]);

    for(int i=0; i<dimension; i++)
        free(matrix[i]);    
    
    free(matrix);
}