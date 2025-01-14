#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tuple>
#include <iterator>
#include <exception>
#include <iomanip>
#include <chrono>

using namespace std;

// Function to solve a system of linear equations using Gaussian Elimination
vector<double> solve(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    vector<vector<double>> augmented_matrix(n, vector<double>(n + 1));

    // Create augmented matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented_matrix[i][j] = A[i][j];
        }
        augmented_matrix[i][n] = b[i];
    }

    // Gaussian Elimination
    for (int i = 0; i < n; ++i) {
        // Find pivot row
        int pivot_row = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(augmented_matrix[j][i]) > abs(augmented_matrix[pivot_row][i])) {
                pivot_row = j;
            }
        }

        // Swap rows (if necessary)
        if (pivot_row != i) {
            swap(augmented_matrix[i], augmented_matrix[pivot_row]);
        }

        // Make all elements below the pivot zero
        for (int j = i + 1; j < n; ++j) {
            double factor = augmented_matrix[j][i] / augmented_matrix[i][i];
            for (int k = i; k <= n; ++k) {
                augmented_matrix[j][k] -= factor * augmented_matrix[i][k];
            }
        }
    }

    // Back-substitution
    vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = augmented_matrix[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= augmented_matrix[i][j] * x[j];
        }
        x[i] /= augmented_matrix[i][i];
    }

    return x;
}

// Function to check if a value is close to zero within a tolerance
bool is_zero(double value, double tol = 1e-16) {
  return std::abs(value) <= tol;
}

// Function to calculate the dot product of two vectors
vector<double> dot(const vector<vector<double>>& A, const vector<double>& v) {
  int m = A.size();
  int n = v.size();
  vector<double> result(m, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i] += A[i][j] * v[j];
    }
  }
  return result;
}

// Function to check if a strategy obeys its support
bool obey_support(const vector<double>& strategy, const vector<int>& support, double tol = 1e-16) {
  for (int i = 0; i < strategy.size(); ++i) {
    if ((find(support.begin(), support.end(), i) != support.end() && strategy[i] <= tol) ||
        (find(support.begin(), support.end(), i) == support.end() && strategy[i] > tol)) {
      return false;
    }
  }
  return true;
}

// Function to solve indifference for a payoff matrix assuming support for the strategies
vector<double> solve_indifference(const vector<vector<double>>& A, const vector<int>& rows) {
  // Ensure differences between pairs of pure strategies are the same
  int num_rows = rows.size();
  vector<vector<double>> M(num_rows - 1, vector<double>(A[0].size()));
  for (int i = 0; i < num_rows - 1; ++i) {
    for (int j = 0; j < A[0].size(); ++j) {
      M[i][j] = A[rows[i]][j] - A[rows[i + 1]][j];
    }
  }

  // Add a row to ensure the strategy sums to 1
  vector<double> ones(A[0].size(), 1.0);
  M.push_back(ones);

  // Solve the linear system for probabilities
  vector<double> b(num_rows, 0.0);
  b[num_rows - 1] = 1.0; 
  try {
    vector<double> prob = solve(M, b); 
    if (all_of(prob.begin(), prob.end(), [](double x) { return x >= 0; })) {
      return prob;
    }
  } catch (const exception& e) {
    // Handle potential exceptions during solving (e.g., singular matrix)
    return vector<double>(); // Empty vector indicates no solution found
  }
  return vector<double>(); // Empty vector indicates no solution found
}

// Function to generate power set (all subsets) of a set
vector<vector<int>> powerset(int n) {
  vector<vector<int>> subsets;
  for (int i = 1; i <= n; ++i) {
    for (int j = 0; j < (1 << n); ++j) {
      vector<int> subset;
      for (int k = 0; k < n; ++k) {
        if (j & (1 << k)) {
          subset.push_back(k);
        }
      }
      if (subset.size() == i) {
        subsets.push_back(subset);
      }
    }
  }
  return subsets;
}

// Function to generate potential support pairs
vector<pair<vector<int>, vector<int>>> potential_support_pairs(
    const vector<vector<double>>& A, const vector<vector<double>>& B, bool non_degenerate = false) {
  int m = A.size();
  int n = B[0].size();
  vector<pair<vector<int>, vector<int>>> pairs;
  for (const vector<int>& support1 : powerset(m)) {
    if (support1.empty()) {
      continue;
    }
    for (const vector<int>& support2 : powerset(n)) {
      if (support2.empty()) {
        continue;
      }
      if (!non_degenerate || support1.size() == support2.size()) {
        pairs.push_back(make_pair(support1, support2));
      }
    }
  }
  return pairs;
}

// Function to check if a given strategy pair is a pair of best responses
bool is_ne(
    const vector<double>& s1, const vector<double>& s2,
    const vector<int>& support1, const vector<int>& support2,
    const vector<vector<double>>& A, const vector<vector<double>>& B) {
  // Payoff against opponents strategies:
  vector<double> row_payoffs = dot(A, s2);
  vector<double> column_payoffs = dot(B, s1);

  // Extract payoffs for the current support
  vector<double> row_support_payoffs;
  for (int i : support1) {
    row_support_payoffs.push_back(row_payoffs[i]);
  }
  vector<double> column_support_payoffs;
  for (int j : support2) {
    column_support_payoffs.push_back(column_payoffs[j]);
  }

  // Check if maximum payoffs are achieved within the support
  return *max_element(row_payoffs.begin(), row_payoffs.end()) == *max_element(row_support_payoffs.begin(), row_support_payoffs.end()) &&
         *max_element(column_payoffs.begin(), column_payoffs.end()) == *max_element(column_support_payoffs.begin(), column_support_payoffs.end());
}

// Function to perform support enumeration and find Nash equilibria
vector<pair<vector<double>, vector<double>>> support_enumeration(
    const vector<vector<double>>& A, const vector<vector<double>>& B, bool non_degenerate = false, double tol = 1e-16) {
  vector<pair<vector<double>, vector<double>>> equilibria;
  for (const auto& pair : potential_support_pairs(A, B, non_degenerate)) {
    const vector<int>& support1 = pair.first;
    const vector<int>& support2 = pair.second;

    vector<double> s1 = solve_indifference(B, support2);
    vector<double> s2 = solve_indifference(A, support1);

    if (obey_support(s1, support1, tol) && obey_support(s2, support2, tol) && is_ne(s1, s2, support1, support2, A, B)) {
      equilibria.push_back(make_pair(s1, s2));
    }
  }
  return equilibria;
}

// Helper function to print a vector
void print_vector(const vector<double>& v) {
  for (double x : v) {
    cout << fixed << setprecision(4) << x << " ";
  }
  cout << endl;
}
std::vector<std::vector<double>> generateMatrix(int dimension){
    vector<vector<double>> matrix(dimension, vector<double>(dimension, 1.0));
    matrix[0][0] = 5;
    matrix[55][30] = 5;
    matrix[20][10] = 5;
    matrix[90][70] = 5;
    matrix[50][50] = 5;
    matrix[95][88] = 5;
    return matrix;

}
// Example usage
int main() {
  // Define a simple 2-player game (Matching Pennies)
  vector<vector<double>> A(generateMatrix(500));//{{1, -1}, {-1, 1}}; // Payoffs for player 1
  vector<vector<double>> B(generateMatrix(500));//{{-1, 1}, {1, -1}}; // Payoffs for player 2
    std::cout<<"EKV";
     
     auto start = std::chrono::high_resolution_clock::now();

  vector<pair<vector<double>, vector<double>>> equilibria = support_enumeration(A, B);
    
    auto end = std::chrono::high_resolution_clock::now();

    std::cout<< "Execution time of Nash equilibrium: "
        <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  cout << "Nash Equilibria:" << endl;
  for (const auto& equilibrium : equilibria) {
    cout << "Player 1: ";
    print_vector(equilibrium.first);
    cout << "Player 2: ";
    print_vector(equilibrium.second);
    cout << endl;
  }

  return 0;
}