#ifndef NAVIGATION_HPP
#define NAVIGATION_HPP

#include "Geometry.hpp"
#include <vector>

// Declaration of C-style wrapper function which is define in Navigation.cu                  
// This make avaivable main.cpp to call CUDA implementation
extern "C" Point findBestDirectionGPU(
    const Point& current_pos,
    const Point& target_pos,
    const std::vector<Circle>& dynamic_obstacles,
    const std::vector<Polygon>& static_obstacles,
    float agent_radius,
    float agent_speed);

#endif // NAVIGATION_HPP