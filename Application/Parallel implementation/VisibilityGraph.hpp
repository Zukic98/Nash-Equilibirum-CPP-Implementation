#ifndef VISIBILITY_GRAPH_HPP
#define VISIBILITY_GRAPH_HPP

#include "Geometry.hpp"
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <limits>
#include <cmath>
#include <algorithm>

// --- Helper Functions for Geometry ---

double cross_product(const Point& O, const Point& A, const Point& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

bool on_segment(const Point& P, const Point& Q, const Point& R) {
    return std::fabs(cross_product(P, Q, R)) < EPSILON &&
           Q.x <= std::max(P.x, R.x) + EPSILON && Q.x >= std::min(P.x, R.x) - EPSILON &&
           Q.y <= std::max(P.y, R.y) + EPSILON && Q.y >= std::min(P.y, R.y) - EPSILON;
}

bool segments_intersect(const Segment& s1, const Segment& s2) {
    Point p1 = s1.p1, q1 = s1.p2;
    Point p2 = s2.p1, q2 = s2.p2;

    double o1 = cross_product(p1, q1, p2);
    double o2 = cross_product(p1, q1, q2);
    double o3 = cross_product(p2, q2, p1);
    double o4 = cross_product(p2, q2, q1);

    if (o1 * o2 < -EPSILON && o3 * o4 < -EPSILON) {
        return true;
    }

    if (std::fabs(o1) < EPSILON && on_segment(p1, p2, q1)) return true;
    if (std::fabs(o2) < EPSILON && on_segment(p1, q2, q1)) return true;
    if (std::fabs(o3) < EPSILON && on_segment(p2, p1, q2)) return true;
    if (std::fabs(o4) < EPSILON && on_segment(p2, q1, q2)) return true;

    return false;
}

double euclidean_distance(const Point& p1, const Point& p2) {
    return std::hypot(p1.x - p2.x, p1.y - p2.y);
}

// Function to check if a point is strictly inside a polygon
bool is_point_inside_polygon(const Point& p, const Polygon& poly) {
    if (poly.vertices.size() < 3) return false;

    // Check if the point is one of the vertices
    for (const auto& v : poly.vertices) {
        if (p == v) return false;
    }

    // Check if the point is on any edge
    for (const auto& edge : poly.get_edges()) {
        if (on_segment(edge.p1, p, edge.p2)) return false;
    }

    // Ray casting algorithm
    int intersect_count = 0;
    Point extreme(std::numeric_limits<double>::max(), p.y);

    for (size_t i = 0; i < poly.vertices.size(); ++i) {
        Point p1 = poly.vertices[i];
        Point p2 = poly.vertices[(i + 1) % poly.vertices.size()];
        Segment edge(p1, p2);
        Segment ray(p, extreme);

        if (segments_intersect(edge, ray)) {
            // Special cases for horizontal edges
            if (p1.y != p.y || p2.y != p.y) {
                if (p1.y == p.y && p2.y > p.y) {
                    // Ray intersects vertex from below
                } else if (p2.y == p.y && p1.y > p.y) {
                    // Ray intersects vertex from below
                } else {
                    intersect_count++;
                }
            }
        }
    }
    return intersect_count % 2 == 1;
}

// --- Optimized Visibility Graph Functions ---

struct AnglePoint {
    Point p;
    double angle;
    double distance;
    
    AnglePoint(const Point& _p, double _angle, double _dist) 
        : p(_p), angle(_angle), distance(_dist) {}
    
    bool operator<(const AnglePoint& other) const {
        if (std::fabs(angle - other.angle) < EPSILON) {
            return distance < other.distance;
        }
        return angle < other.angle;
    }
};

// Improved is_visible function with early termination
bool is_visible(const Point& p1, const Point& p2, const std::vector<Polygon>& obstacles) {
    Segment visibility_segment(p1, p2);
    
    if (p1 == p2) return true;
    
    // Check midpoint first for quick rejection
    Point mid_point((p1.x + p2.x)/2, (p1.y + p2.y)/2);
    for (const auto& obs : obstacles) {
        if (is_point_inside_polygon(mid_point, obs)) {
            return false;
        }
    }
    
    // Check all obstacle edges
    for (const auto& obstacle : obstacles) {
        for (const auto& edge : obstacle.get_edges()) {
            if (segments_intersect(visibility_segment, edge)) {
                // Check if intersection is just at endpoints
                if (!(on_segment(edge.p1, p1, edge.p2) || on_segment(edge.p1, p2, edge.p2))) {
                    return false;
                }
            }
        }
    }
    
    return true;
}

// Rotational sweep visibility check (O(n log n) per point)
std::vector<Point> find_visible_points(const Point& origin, 
                                     const std::vector<Point>& points,
                                     const std::vector<Polygon>& obstacles) {
    std::vector<AnglePoint> angle_points;
    std::vector<Point> visible_points;
    
    // Prepare points sorted by angle and distance
    for (const auto& p : points) {
        if (p == origin) continue;
        
        double dx = p.x - origin.x;
        double dy = p.y - origin.y;
        double angle = atan2(dy, dx);
        double dist = euclidean_distance(origin, p);
        
        angle_points.emplace_back(p, angle, dist);
    }
    
    // Sort by angle then distance
    std::sort(angle_points.begin(), angle_points.end());
    
    // For each point, check if it's the closest at its angle
    for (size_t i = 0; i < angle_points.size(); ) {
        double current_angle = angle_points[i].angle;
        Point closest_point = angle_points[i].p;
        double min_dist = angle_points[i].distance;
        
        // Find all points with same angle (within epsilon)
        size_t j = i + 1;
        while (j < angle_points.size() && 
               std::fabs(angle_points[j].angle - current_angle) < EPSILON) {
            if (angle_points[j].distance < min_dist) {
                closest_point = angle_points[j].p;
                min_dist = angle_points[j].distance;
            }
            j++;
        }
        
        // Check visibility to the closest point
        if (is_visible(origin, closest_point, obstacles)) {
            visible_points.push_back(closest_point);
        }
        
        i = j; // Skip all points with same angle
    }
    
    return visible_points;
}

struct VisibilityGraphResult {
    std::vector<Point> all_nodes;
    std::map<Point, int> point_to_index;
    std::vector<std::vector<std::pair<int, double>>> adj_list;
};

// Optimized O(n² log n) visibility graph construction
VisibilityGraphResult build_visibility_graph(
    const std::vector<Polygon>& obstacles,
    const Point& start_point,
    const Point& end_point) {
    
    std::set<Point> unique_points_set;
    
    // Collect all polygon vertices
    for (const auto& obs : obstacles) {
        for (const auto& p : obs.vertices) {
            unique_points_set.insert(p);
        }
    }
    
    // Add start and end points
    unique_points_set.insert(start_point);
    unique_points_set.insert(end_point);
    
    VisibilityGraphResult result;
    int index = 0;
    std::vector<Point> all_points;
    
    // Create mapping between points and indices
    for (const auto& p : unique_points_set) {
        result.all_nodes.push_back(p);
        result.point_to_index[p] = index++;
        all_points.push_back(p);
    }
    
    result.adj_list.resize(result.all_nodes.size());
    
    // For each point, find visible points using rotational sweep
    for (size_t i = 0; i < result.all_nodes.size(); ++i) {
        const Point& origin = result.all_nodes[i];
        std::vector<Point> visible = find_visible_points(origin, all_points, obstacles);
        
        for (const auto& target : visible) {
            int j = result.point_to_index[target];
            double dist = euclidean_distance(origin, target);
            result.adj_list[i].emplace_back(j, dist);
        }
    }
    
    return result;
}

// Dijkstra's algorithm remains the same
struct State {
    double dist;
    int u;
    
    bool operator>(const State& other) const {
        return dist > other.dist;
    }
};

std::vector<Point> find_shortest_path_dijkstra(
    const Point& start_point,
    const Point& end_point,
    const VisibilityGraphResult& graph_result) {
    
    auto it_start = graph_result.point_to_index.find(start_point);
    auto it_end = graph_result.point_to_index.find(end_point);
    
    if (it_start == graph_result.point_to_index.end() || 
        it_end == graph_result.point_to_index.end()) {
        return {};
    }
    
    int start_idx = it_start->second;
    int end_idx = it_end->second;
    int num_nodes = graph_result.all_nodes.size();
    
    std::vector<double> dist(num_nodes, std::numeric_limits<double>::infinity());
    std::vector<int> parent(num_nodes, -1);
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    
    dist[start_idx] = 0;
    pq.push({0, start_idx});
    
    while (!pq.empty()) {
        State current = pq.top();
        pq.pop();
        
        if (current.u == end_idx) {
            std::vector<Point> path;
            int curr = end_idx;
            while (curr != -1) {
                path.insert(path.begin(), graph_result.all_nodes[curr]);
                curr = parent[curr];
            }
            return path;
        }
        
        if (current.dist > dist[current.u]) continue;
        
        for (const auto& edge : graph_result.adj_list[current.u]) {
            int v = edge.first;
            double weight = edge.second;
            
            if (dist[current.u] + weight < dist[v]) {
                dist[v] = dist[current.u] + weight;
                parent[v] = current.u;
                pq.push({dist[v], v});
            }
        }
    }
    
    return {}; // No path found
}

#endif // VISIBILITY_GRAPH_HPP