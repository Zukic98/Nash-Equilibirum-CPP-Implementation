#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <limits>
#include "Geometry.hpp" // Using the same structures

// __device__ versions of helper functions for use on the GPU
__device__ inline double d_distance_sq(const Point& p1, const Point& p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

__device__ inline double d_distance_point_to_segment_sq(const Point& p, const Segment& s) {
    double l2 = d_distance_sq(s.p1, s.p2);
    if (l2 < EPSILON) return d_distance_sq(p, s.p1);
    double t = ((p.x - s.p1.x) * (s.p2.x - s.p1.x) + (p.y - s.p1.y) * (s.p2.y - s.p1.y)) / l2;
    t = max(0.0, min(1.0, t));
    Point projection = {s.p1.x + t * (s.p2.x - s.p1.x), s.p1.y + t * (s.p2.y - s.p1.y)};
    return d_distance_sq(p, projection);
}

// Structure for weights - same as before
struct NavigationWeights {
    double goal = 4.0;
    double deviation_from_path = 1.5;
    double dynamic_avoid_warn = 20.0;
    double static_edge_nudge = 0.5;
};

// CUDA Kernel: Each thread calculates a score for one direction
__global__ void calculate_direction_scores_kernel(
    double* out_scores,
    Point agent_pos,
    Point target_pos,
    Circle* dynamic_obstacles, int num_dynamic,
    Polygon* static_obstacles, int num_static, int* static_vertex_counts, Segment* all_static_edges, int num_static_edges,
    float agent_radius, float agent_speed,
    NavigationWeights weights, int num_directions) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_directions) return;

    // --- The logic is almost identical to the C++ version, but executed on the GPU ---

    const double PI_d = 3.14159265358979323846;
    double angle = 2.0 * PI_d * i / num_directions;
    Point direction = {cos(angle), sin(angle)};

    Point vector_to_target = {target_pos.x - agent_pos.x, target_pos.y - agent_pos.y};
    double target_dist = hypot(vector_to_target.x, vector_to_target.y);
    if (target_dist > EPSILON) {
        vector_to_target.x /= target_dist;
        vector_to_target.y /= target_dist;
    }

    // --- VETO CHECKS ---
    // Veto 1: Collision with a dynamic circle
    double prediction_time = 0.5;
    Point agent_future_pos = {agent_pos.x + direction.x * agent_speed * prediction_time,
                              agent_pos.y + direction.y * agent_speed * prediction_time};

    for (int k = 0; k < num_dynamic; ++k) {
        Point circle_future_pos = {dynamic_obstacles[k].center.x + dynamic_obstacles[k].velocity.x * prediction_time,
                                   dynamic_obstacles[k].center.y + dynamic_obstacles[k].velocity.y * prediction_time};
        double min_safe_dist = agent_radius + dynamic_obstacles[k].radius;
        if (d_distance_sq(agent_future_pos, circle_future_pos) < (min_safe_dist * min_safe_dist)) {
            out_scores[i] = -1.0/0.0; // Negative infinity
            return;
        }
    }
    
    // Unfortunately, checking for polygons is complex on the GPU due to irregular memory structures.
    // For now, it is left to the CPU or simplified. It is omitted in this example for simplicity.
    // In a real application, this would require advanced CUDA techniques.

    // --- SCORING ---
    double current_score = 0.0;
    double goal_projection = (direction.x * vector_to_target.x + direction.y * vector_to_target.y);
    current_score += weights.goal * goal_projection;
    current_score -= weights.deviation_from_path * (1.0 - goal_projection);

    double dynamic_warning_penalty = 0.0;
    for (int k = 0; k < num_dynamic; ++k) {
        double warning_dist = (agent_radius + dynamic_obstacles[k].radius) * 2.0;
        if (d_distance_sq(agent_future_pos, dynamic_obstacles[k].center) < (warning_dist * warning_dist)) {
            dynamic_warning_penalty += weights.dynamic_avoid_warn;
        }
    }
    current_score -= dynamic_warning_penalty;

    out_scores[i] = current_score;
}

// Wrapper function: Manages memory and calls the kernel
extern "C" Point findBestDirectionGPU(
    const Point& current_pos, const Point& target_pos,
    const std::vector<Circle>& dynamic_obstacles,
    const std::vector<Polygon>& static_obstacles, // Retained for future development
    float agent_radius, float agent_speed)
{
    const int NUM_DIRECTIONS = 72;
    NavigationWeights weights;

    // Allocate memory on the GPU
    double* d_scores;
    Circle* d_dynamic_obstacles;
    cudaMalloc(&d_scores, NUM_DIRECTIONS * sizeof(double));
    cudaMalloc(&d_dynamic_obstacles, dynamic_obstacles.size() * sizeof(Circle));

    // Copy data to the GPU
    cudaMemcpy(d_dynamic_obstacles, dynamic_obstacles.data(), dynamic_obstacles.size() * sizeof(Circle), cudaMemcpyHostToDevice);

    // Call the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_DIRECTIONS + threadsPerBlock - 1) / threadsPerBlock;
    calculate_direction_scores_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_scores, current_pos, target_pos,
        d_dynamic_obstacles, dynamic_obstacles.size(),
        nullptr, 0, nullptr, nullptr, 0, // Static obstacles are ignored on the GPU for now
        agent_radius, agent_speed, weights, NUM_DIRECTIONS
    );

    // Copy the results (scores) back to the CPU
    std::vector<double> h_scores(NUM_DIRECTIONS);
    cudaMemcpy(h_scores.data(), d_scores, NUM_DIRECTIONS * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_scores);
    cudaFree(d_dynamic_obstacles);

    // Find the best direction on the CPU (very fast since the array is small)
    double max_score = -std::numeric_limits<double>::infinity();
    int best_direction_index = 0;
    for (int i = 0; i < NUM_DIRECTIONS; ++i) {
        if (h_scores[i] > max_score) {
            max_score = h_scores[i];
            best_direction_index = i;
        }
    }
    
    // If all directions are bad, don't move (or have some fallback logic)
    if (max_score == -std::numeric_limits<double>::infinity()) {
        return {0, 0};
    }

    double best_angle = 2.0 * PI * best_direction_index / NUM_DIRECTIONS;
    return {cos(best_angle), sin(best_angle)};
}