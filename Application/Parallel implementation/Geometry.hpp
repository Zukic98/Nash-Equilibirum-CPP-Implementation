#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <tuple>

constexpr double EPSILON = 1e-9;
const double PI = 3.14159265358979323846;

struct Point {
    double x, y;
    Point(double _x = 0.0, double _y = 0.0) : x(_x), y(_y) {}

    bool operator==(const Point& other) const {
        return std::fabs(x - other.x) < EPSILON &&
               std::fabs(y - other.y) < EPSILON;
    }

    bool operator!=(const Point& other) const {
        return !(*this == other);
    }
    
    bool operator<(const Point& other) const {
        return std::tie(x, y) < std::tie(other.x, other.y);
    }
};

struct Segment {
    Point p1, p2;
    Segment(const Point& _p1, const Point& _p2) : p1(_p1), p2(_p2) {}
};

struct Polygon {
    std::vector<Point> vertices;
    
    std::vector<Segment> get_edges() const {
        std::vector<Segment> edges;
        if (vertices.size() < 2) return edges;
        for (size_t i = 0; i < vertices.size(); ++i) {
            edges.emplace_back(vertices[i], vertices[(i + 1) % vertices.size()]);
        }
        return edges;
    }
};

struct Circle {
    Point center;
    double radius;
    Point velocity;
    bool isColliding = false;
};

#endif // GEOMETRY_HPP