#ifndef NAVIGATION_HPP
#define NAVIGATION_HPP

#include "Geometry.hpp"
#include "VisibilityGraph.hpp"
#include <vector>
#include <limits>

// --- Helper functions for navigation ---
inline double distance_sq(const Point& p1, const Point& p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// Calculates the squared distance from a point to a line segment
inline double distance_point_to_segment_sq(const Point& p, const Segment& s) {
    double l2 = distance_sq(s.p1, s.p2);
    if (l2 < EPSILON) return distance_sq(p, s.p1);
    double t = ((p.x - s.p1.x) * (s.p2.x - s.p1.x) + (p.y - s.p1.y) * (s.p2.y - s.p1.y)) / l2;
    t = std::max(0.0, std::min(1.0, t));
    Point projection = {s.p1.x + t * (s.p2.x - s.p1.x), s.p1.y + t * (s.p2.y - s.p1.y)};
    return distance_sq(p, projection);
}

// --- Navigation weights structure ---
struct NavigationWeights {
    double goal = 4.0;                  // Attraction to the goal
    double deviation_from_path = 1.5;   // Penalty for deviating from the ideal path
    double dynamic_avoid_warn = 20.0;   // Penalty for entering a dynamic obstacle's "warning zone"
    double static_edge_nudge = 0.5;     // Gentle "nudge" away from walls to prevent sticking
};

class AgentNavigator {
public:
    static Point findBestDirection(
        const Point& current_pos,
        const Point& target_pos,
        const std::vector<Circle>& dynamic_obstacles,
        const std::vector<Polygon>& static_obstacles,
        float agent_radius,
        float agent_speed,
        const NavigationWeights& weights = NavigationWeights{}
    ) {
        const int num_directions = 72;
        const double prediction_time = 0.5;
        const double static_safety_margin = 1.1;

        Point best_direction = {0, 0};
        double max_score = -std::numeric_limits<double>::infinity();

        Point vector_to_target = {target_pos.x - current_pos.x, target_pos.y - current_pos.y};
        double target_dist = std::hypot(vector_to_target.x, vector_to_target.y);
        if (target_dist > EPSILON) {
            vector_to_target.x /= target_dist;
            vector_to_target.y /= target_dist;
        }

        for (int i = 0; i < num_directions; ++i) {
            double angle = 2.0 * PI * i / num_directions;
            Point direction = {std::cos(angle), std::sin(angle)};
            
            double current_score = 0.0;

            // --- CHECK FOR ABSOLUTE VETOES ---

            // VETO 1: Collision with a dynamic circle
            Point agent_future_pos = {current_pos.x + direction.x * agent_speed * prediction_time,
                                      current_pos.y + direction.y * agent_speed * prediction_time};
            bool collision_predicted = false;
            for (const auto& circle : dynamic_obstacles) {
                Point circle_future_pos = {circle.center.x + circle.velocity.x * prediction_time,
                                           circle.center.y + circle.velocity.y * prediction_time};
                double min_safe_dist = agent_radius + circle.radius;
                if (distance_sq(agent_future_pos, circle_future_pos) < (min_safe_dist * min_safe_dist)) {
                    collision_predicted = true;
                    break;
                }
            }
            if (collision_predicted) {
                continue; // Disqualify this direction completely
            }

            // VETO 2: Entering a static polygon
            Point agent_next_step_pos = {current_pos.x + direction.x * agent_radius, 
                                         current_pos.y + direction.y * agent_radius};
            bool inside_polygon = false;
            for (const auto& poly : static_obstacles) {
                if (is_point_inside_polygon(agent_next_step_pos, poly)) {
                    inside_polygon = true;
                    break;
                }
            }
            if (inside_polygon) {
                continue; // Disqualify this direction completely
            }


            // --- SCORING IF THE DIRECTION IS PERMITTED ---
            
            // 1. REWARD FOR MOVING TOWARDS THE GOAL
            double goal_projection = (direction.x * vector_to_target.x + direction.y * vector_to_target.y);
            current_score += weights.goal * goal_projection;

            // 2. PENALTY FOR DEVIATION FROM THE IDEAL PATH
            current_score -= weights.deviation_from_path * (1.0 - goal_projection);
            
            // 3. PENALTY FOR THE WARNING ZONE AROUND DYNAMIC OBSTACLES
            double dynamic_warning_penalty = 0.0;
            for (const auto& circle : dynamic_obstacles) {
                double warning_dist = (agent_radius + circle.radius) * 2.0;
                if (distance_sq(agent_future_pos, circle.center) < (warning_dist * warning_dist)) {
                    dynamic_warning_penalty += weights.dynamic_avoid_warn;
                }
            }
            current_score -= dynamic_warning_penalty;
            
            // 4. GENTLE PENALTY FOR PROXIMITY TO WALLS
            double static_nudge_penalty = 0.0;
             for (const auto& poly : static_obstacles) {
                for (const auto& edge : poly.get_edges()) {
                    double dist_to_edge_sq = distance_point_to_segment_sq(agent_next_step_pos, edge);
                    double min_safe_dist_static_sq = (agent_radius * static_safety_margin) * (agent_radius * static_safety_margin);
                    if (dist_to_edge_sq < min_safe_dist_static_sq) {
                        static_nudge_penalty += weights.static_edge_nudge * (1.0 - sqrt(dist_to_edge_sq) / (agent_radius * static_safety_margin));
                    }
                }
            }
            current_score -= static_nudge_penalty;


            if (current_score > max_score) {
                max_score = current_score;
                best_direction = direction;
            }
        }

        // If ALL directions are disqualified (the agent is in a no-win situation),
        // it will try to move in the direction of the ideal path.
        if (max_score == -std::numeric_limits<double>::infinity()) {
            return vector_to_target;
        }

        return best_direction;
    }
};

#endif // NAVIGATION_HPP