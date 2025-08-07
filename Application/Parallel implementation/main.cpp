#include <SFML/Graphics.hpp>
#include <SFML/System/Clock.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Geometry.hpp"
#include "Navigation.hpp"
#include "VisibilityGraph.hpp"

using namespace std::chrono;

// --- Application State ---
enum class AppState {
  DRAWING_POLYGONS,
  SELECTING_POINTS,
  SETTING_START,
  SETTING_END,
  PATH_READY,
  SIMULATING,
  SIMULATION_ENDED
};

// --- Simulation Constants ---
const float AGENT_SPEED = 100.0f;
const float AGENT_RADIUS = 10.0f;
const int NUM_DYNAMIC_OBSTACLES = 40;
const float OBSTACLE_MIN_SPEED = 20.0f;
const float OBSTACLE_MAX_SPEED = 50.0f;
const float OBSTACLE_RADIUS = 20.0f;

// --- Global State Variables ---
AppState currentAppState = AppState::DRAWING_POLYGONS;
std::vector<Polygon> obstacles;
Polygon currentDrawingPolygon;
Point startPoint(-1.0, -1.0), endPoint(-1.0, -1.0);
VisibilityGraphResult currentGraphResult;
std::vector<Point> shortestPath;
bool pathFound = false;

// --- Simulation State ---
Point agentPosition;
float agentRotation = 0.0f;
int currentWaypointIndex = 0;
std::vector<Circle> dynamicObstacles;
std::vector<Circle> initialDynamicObstacles;
sf::Clock simulationClock;
sf::Time travelTime;
int collisionCount = 0; // NEW: Collision counter

// --- UI Elements ---
struct Button {
  sf::RectangleShape rect;
  sf::Text text;
  bool isHovered = false;

  Button(const std::string &label, sf::Vector2f size, sf::Vector2f position,
         const sf::Font &font, int charSize)
      : text("", font, charSize) {
    rect.setSize(size);
    rect.setPosition(position);
    rect.setFillColor(sf::Color(70, 70, 70));
    rect.setOutlineThickness(2);
    rect.setOutlineColor(sf::Color::Black);

    text.setString(label);
    text.setFillColor(sf::Color::White);
    sf::FloatRect textRect = text.getLocalBounds();
    text.setOrigin(textRect.left + textRect.width / 2.0f,
                   textRect.top + textRect.height / 2.0f);
    text.setPosition(
        sf::Vector2f(position.x + size.x / 2.0f, position.y + size.y / 2.0f));
  }

  void draw(sf::RenderWindow &window) {
    window.draw(rect);
    window.draw(text);
  }

  bool contains(sf::Vector2f point) const {
    return rect.getGlobalBounds().contains(point);
  }

  void setHovered(bool hover) {
    isHovered = hover;
    if (isHovered) {
      rect.setFillColor(sf::Color(100, 100, 100));
    } else {
      rect.setFillColor(sf::Color(70, 70, 70));
    }
  }
};

// --- Helper Functions (same as before) ---
void drawDashedLine(sf::RenderWindow &window, sf::Vector2f p1, sf::Vector2f p2,
                    sf::Color color, float dashLength = 5.0f,
                    float gapLength = 5.0f);
bool loadPolygonsFromFile(const std::string &filename,
                          std::vector<Polygon> &polygons);

// --- Simulation Management Functions ---
void resetSimulationState() {
  agentPosition = startPoint;
  currentWaypointIndex = 1;
  collisionCount = 0; // NEW: Reset the counter
  simulationClock.restart();
  currentAppState = AppState::SIMULATING;
}

void setupSimulation(float plot_width, float plot_height,
                     bool use_initial_state) {
  if (use_initial_state && !initialDynamicObstacles.empty()) {
    dynamicObstacles = initialDynamicObstacles;
  } else {
    initialDynamicObstacles.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disX(0, plot_width);
    std::uniform_real_distribution<> disY(0, plot_height);
    std::uniform_real_distribution<> disSpeed(OBSTACLE_MIN_SPEED,
                                              OBSTACLE_MAX_SPEED);
    std::uniform_real_distribution<> disAngle(0, 2 * PI);

    while (initialDynamicObstacles.size() < NUM_DYNAMIC_OBSTACLES) {
      Point center = {disX(gen), disY(gen)};
      bool is_valid_spawn = true;
      for (const auto &poly : obstacles) {
        if (is_point_inside_polygon(center, poly)) {
          is_valid_spawn = false;
          break;
        }
      }
      if (euclidean_distance(center, startPoint) < OBSTACLE_RADIUS * 4 ||
          euclidean_distance(center, endPoint) < OBSTACLE_RADIUS * 4) {
        is_valid_spawn = false;
      }
      if (is_valid_spawn) {
        float speed = disSpeed(gen);
        float angle = disAngle(gen);
        Point velocity = {std::cos(angle) * speed, std::sin(angle) * speed};
        initialDynamicObstacles.push_back(
            {center, OBSTACLE_RADIUS, velocity, false});
      }
    }
    dynamicObstacles = initialDynamicObstacles;
  }
  resetSimulationState();
}

void updateSimulation(float dt, float plot_width, float plot_height) {
  if (shortestPath.empty() || currentWaypointIndex >= shortestPath.size()) {
    currentAppState = AppState::SIMULATION_ENDED;
    travelTime = simulationClock.getElapsedTime();
    return;
  }

  // Move the circles
  for (auto &circle : dynamicObstacles) {
    circle.center.x += circle.velocity.x * dt;
    circle.center.y += circle.velocity.y * dt;
    if (circle.center.x - circle.radius < 0) {
      circle.center.x = circle.radius;
      circle.velocity.x *= -1;
    }
    if (circle.center.x + circle.radius > plot_width) {
      circle.center.x = plot_width - circle.radius;
      circle.velocity.x *= -1;
    }
    if (circle.center.y - circle.radius < 0) {
      circle.center.y = circle.radius;
      circle.velocity.y *= -1;
    }
    if (circle.center.y + circle.radius > plot_height) {
      circle.center.y = plot_height - circle.radius;
      circle.velocity.y *= -1;
    }
  }

  // Agent navigation
  Point targetWaypoint = shortestPath[currentWaypointIndex];
  Point best_direction =
      findBestDirectionGPU(agentPosition, targetWaypoint, dynamicObstacles,
                           obstacles, AGENT_RADIUS, AGENT_SPEED);
  float norm = std::hypot(best_direction.x, best_direction.y);
  if (norm > EPSILON) {
    best_direction.x /= norm;
    best_direction.y /= norm;
  }
  agentPosition.x += best_direction.x * AGENT_SPEED * dt;
  agentPosition.y += best_direction.y * AGENT_SPEED * dt;
  if (norm > EPSILON) {
    agentRotation =
        std::atan2(best_direction.y, best_direction.x) * 180.0f / PI;
  }

  // NEW: Collision detection
  for (auto &circle : dynamicObstacles) {
    double min_dist_sq =
        (AGENT_RADIUS + circle.radius) * (AGENT_RADIUS + circle.radius);
    if (d_distance_sq(agentPosition, circle.center) < min_dist_sq) {
      // Collision occurred. Check if it's already logged.
      if (!circle.isColliding) {
        collisionCount++;
        circle.isColliding = true; // Mark that a collision is in progress
      }
    } else {
      // No collision, reset the flag
      circle.isColliding = false;
    }
  }

  // Check for goal
  if (euclidean_distance(agentPosition, targetWaypoint) < AGENT_RADIUS) {
    currentWaypointIndex++;
    if (currentWaypointIndex >= shortestPath.size()) {
      currentAppState = AppState::SIMULATION_ENDED;
      travelTime = simulationClock.getElapsedTime();
    }
  }
}

int main() {
  sf::RenderWindow window(sf::VideoMode(1280, 720),
                          "Dynamic Pathfinding Simulation");
  window.setFramerateLimit(60);

  sf::Font font;
  if (!font.loadFromFile("fonts/DejaVuSans.ttf")) {
    return 1;
  }

  // ... [UI setup code, same as before, I won't repeat it for brevity] ...
  // ... [Definitions of all buttons] ...
  float ui_sidebar_width = 280.0f;
  float plot_area_width = window.getSize().x - ui_sidebar_width;
  sf::View plotView(sf::FloatRect(0, 0, plot_area_width, window.getSize().y));
  sf::View uiView(sf::FloatRect(0, 0, window.getSize().x, window.getSize().y));

  float button_width = ui_sidebar_width - 40.0f;
  float button_height = 35.0f;
  float button_start_x = plot_area_width + 20.0f;
  float button_y_spacing = 10.0f;
  float current_button_y = 10.0f;

  Button finishDrawingButton("Finish Drawing", {button_width, button_height},
                             {button_start_x, current_button_y}, font, 14);
  current_button_y += button_height + button_y_spacing;
  Button selectStartButton("Set Start Point", {button_width, button_height},
                           {button_start_x, current_button_y}, font, 14);
  current_button_y += button_height + button_y_spacing;
  Button selectEndButton("Set End Point", {button_width, button_height},
                         {button_start_x, current_button_y}, font, 14);
  current_button_y += button_height + button_y_spacing;
  Button loadFromFileButton("Load from File", {button_width, button_height},
                            {button_start_x, current_button_y}, font, 14);
  current_button_y += button_height + button_y_spacing * 2;
  Button runAlgorithmButton("1. Find Path", {button_width, button_height},
                            {button_start_x, current_button_y}, font, 16);
  runAlgorithmButton.rect.setFillColor(sf::Color(30, 100, 200));
  current_button_y += button_height + button_y_spacing;
  Button playButton("2. PLAY", {button_width, button_height},
                    {button_start_x, current_button_y}, font, 16);
  playButton.rect.setFillColor(sf::Color(30, 150, 30));
  current_button_y += button_height + button_y_spacing;
  Button repeatButton("3. Repeat", {button_width, button_height},
                      {button_start_x, current_button_y}, font, 16);
  repeatButton.rect.setFillColor(sf::Color(230, 126, 34)); // Orange color
  current_button_y += button_height + button_y_spacing * 2;
  Button resetButton("RESET ALL", {button_width, button_height},
                     {button_start_x, current_button_y}, font, 16);
  resetButton.rect.setFillColor(sf::Color(200, 50, 50));

  std::vector<Button *> allButtons = {&finishDrawingButton, &selectStartButton,
                                      &selectEndButton,     &loadFromFileButton,
                                      &runAlgorithmButton,  &playButton,
                                      &repeatButton,        &resetButton};

  sf::Text userMessage("", font, 16);
  userMessage.setFillColor(sf::Color::Black);
  userMessage.setPosition(10, window.getSize().y - 30);
  userMessage.setString(
      "Left-click to add polygon vertex. Right-click to finish polygon.");

  sf::ConvexShape agentShape;
  agentShape.setPointCount(3);
  agentShape.setPoint(0, sf::Vector2f(AGENT_RADIUS, 0));
  agentShape.setPoint(1,
                      sf::Vector2f(-AGENT_RADIUS * 0.7f, -AGENT_RADIUS * 0.7f));
  agentShape.setPoint(2,
                      sf::Vector2f(-AGENT_RADIUS * 0.7f, AGENT_RADIUS * 0.7f));
  agentShape.setFillColor(sf::Color::Red);
  agentShape.setOrigin(0, 0);

  std::vector<sf::CircleShape> obstacleShapes(NUM_DYNAMIC_OBSTACLES);
  for (auto &shape : obstacleShapes) {
    shape.setRadius(OBSTACLE_RADIUS);
    shape.setFillColor(sf::Color(0, 0, 200, 150));
    shape.setOrigin(OBSTACLE_RADIUS, OBSTACLE_RADIUS);
  }

  sf::Clock deltaClock;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();

      sf::Vector2i mousePosScreen = sf::Mouse::getPosition(window);
      sf::Vector2f mousePosUI = window.mapPixelToCoords(mousePosScreen, uiView);
      sf::Vector2f mousePosWorld =
          window.mapPixelToCoords(mousePosScreen, plotView);

      if (event.type == sf::Event::MouseMoved) {
        for (auto &btn : allButtons)
          btn->setHovered(btn->contains(mousePosUI));
      }

      if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
          // Button logic
          if (resetButton.contains(mousePosUI)) {
            currentAppState = AppState::DRAWING_POLYGONS;
            obstacles.clear();
            dynamicObstacles.clear();
            initialDynamicObstacles.clear();
            currentDrawingPolygon.vertices.clear();
            shortestPath.clear();
            pathFound = false;
            startPoint = {-1, -1};
            endPoint = {-1, -1};
            collisionCount = 0; // NEW: Reset the counter
            userMessage.setString(
                "All reset. Left-click to add polygon vertex.");
          }
          // ... [The rest of the button logic, same as before] ...
          else if (finishDrawingButton.contains(mousePosUI)) {
            if (!currentDrawingPolygon.vertices.empty()) {
              obstacles.push_back(currentDrawingPolygon);
              currentDrawingPolygon.vertices.clear();
            }
            currentAppState = AppState::SELECTING_POINTS;
            userMessage.setString(
                "Drawing finished. Use buttons to set Start/End points.");
          } else if (selectStartButton.contains(mousePosUI)) {
            currentAppState = AppState::SETTING_START;
            userMessage.setString("Click on the map to set the START point.");
          } else if (selectEndButton.contains(mousePosUI)) {
            currentAppState = AppState::SETTING_END;
            userMessage.setString("Click on the map to set the END point.");
          } else if (loadFromFileButton.contains(mousePosUI)) {
            if (loadPolygonsFromFile("polygons.txt", obstacles)) {
              currentAppState = AppState::SELECTING_POINTS;
              userMessage.setString("Polygons loaded. Set Start/End points.");
            } else {
              userMessage.setString("Failed to load 'polygons.txt'.");
            }
          } else if (runAlgorithmButton.contains(mousePosUI)) {
            if (startPoint.x != -1.0 && endPoint.x != -1.0) {
              userMessage.setString("Calculating optimal path...");
              window.display();
              shortestPath.clear();
              pathFound = false;
              initialDynamicObstacles.clear();
              currentGraphResult =
                  build_visibility_graph(obstacles, startPoint, endPoint);
              shortestPath = find_shortest_path_dijkstra(startPoint, endPoint,
                                                         currentGraphResult);
              pathFound = !shortestPath.empty();
              if (pathFound) {
                currentAppState = AppState::PATH_READY;
                userMessage.setString(
                    "Path found. Press PLAY to start simulation.");
              } else {
                userMessage.setString("Path NOT found! Check point positions.");
              }
            } else {
              userMessage.setString("Set Start and End points first!");
            }
          } else if (playButton.contains(mousePosUI)) {
            if (currentAppState == AppState::PATH_READY) {
              setupSimulation(plotView.getSize().x, plotView.getSize().y,
                              false);
            }
          } else if (repeatButton.contains(mousePosUI)) {
            if (currentAppState == AppState::SIMULATION_ENDED ||
                currentAppState == AppState::SIMULATING) {
              if (!initialDynamicObstacles.empty()) {
                setupSimulation(plotView.getSize().x, plotView.getSize().y,
                                true);
              }
            }
          }
          // Logic for clicking on the scene
          else if (mousePosScreen.x < plot_area_width) {
            if (currentAppState == AppState::DRAWING_POLYGONS) {
              currentDrawingPolygon.vertices.emplace_back(mousePosWorld.x,
                                                          mousePosWorld.y);
            } else if (currentAppState == AppState::SETTING_START) {
              startPoint = {mousePosWorld.x, mousePosWorld.y};
              currentAppState = AppState::SELECTING_POINTS;
              userMessage.setString("Start point set. Now set End point.");
            } else if (currentAppState == AppState::SETTING_END) {
              endPoint = {mousePosWorld.x, mousePosWorld.y};
              currentAppState = AppState::SELECTING_POINTS;
              userMessage.setString("End point set. Now find path.");
            }
          }
        } else if (event.mouseButton.button == sf::Mouse::Right) {
          if (currentAppState == AppState::DRAWING_POLYGONS &&
              !currentDrawingPolygon.vertices.empty()) {
            obstacles.push_back(currentDrawingPolygon);
            currentDrawingPolygon.vertices.clear();
          }
        }
      }
    }

    // --- Update ---
    float dt = deltaClock.restart().asSeconds();
    if (currentAppState == AppState::SIMULATING) {
      updateSimulation(dt, plotView.getSize().x, plotView.getSize().y);
      // NEW: Updated message with collision counter
      std::ostringstream ss;
      ss << "Time: " << std::fixed << std::setprecision(1)
         << simulationClock.getElapsedTime().asSeconds()
         << "s | Collisions: " << collisionCount;
      userMessage.setString(ss.str());
    } else if (currentAppState == AppState::SIMULATION_ENDED) {
      // NEW: Updated message with collision counter
      std::ostringstream ss;
      ss << "Goal reached! Time: " << std::fixed << std::setprecision(2)
         << travelTime.asSeconds() << "s | Collisions: " << collisionCount
         << ". Press Repeat or Reset.";
      userMessage.setString(ss.str());
    }

    // --- Draw ---
    window.clear(sf::Color(240, 240, 240));

    // UI
    window.setView(uiView);
    for (const auto &btn : allButtons)
      btn->draw(window);
    sf::RectangleShape separator(sf::Vector2f(2, window.getSize().y));
    separator.setPosition(plot_area_width, 0);
    separator.setFillColor(sf::Color(150, 150, 150));
    window.draw(separator);
    window.draw(userMessage);

    // Scene
    window.setView(plotView);
    for (const auto &obs : obstacles) {
      sf::ConvexShape sfmlPolygon;
      sfmlPolygon.setPointCount(obs.vertices.size());
      for (size_t i = 0; i < obs.vertices.size(); ++i) {
        sfmlPolygon.setPoint(
            i, {(float)obs.vertices[i].x, (float)obs.vertices[i].y});
      }
      sfmlPolygon.setFillColor(sf::Color(100, 100, 100, 150));
      sfmlPolygon.setOutlineThickness(2);
      sfmlPolygon.setOutlineColor(sf::Color::Black);
      window.draw(sfmlPolygon);
    }
    if (currentAppState == AppState::DRAWING_POLYGONS &&
        !currentDrawingPolygon.vertices.empty()) {
      for (size_t i = 0; i < currentDrawingPolygon.vertices.size() - 1; ++i) {
        sf::Vertex line[] = {
            sf::Vertex({(float)currentDrawingPolygon.vertices[i].x,
                        (float)currentDrawingPolygon.vertices[i].y},
                       sf::Color::Blue),
            sf::Vertex({(float)currentDrawingPolygon.vertices[i + 1].x,
                        (float)currentDrawingPolygon.vertices[i + 1].y},
                       sf::Color::Blue)};
        window.draw(line, 2, sf::Lines);
      }
    }
    if (startPoint.x != -1.0) {
      sf::CircleShape startShape(6);
      startShape.setFillColor(sf::Color::Green);
      startShape.setPosition(startPoint.x - 6, startPoint.y - 6);
      window.draw(startShape);
    }
    if (endPoint.x != -1.0) {
      sf::CircleShape endShape(6);
      endShape.setFillColor(sf::Color(255, 0, 0));
      endShape.setPosition(endPoint.x - 6, endPoint.y - 6);
      window.draw(endShape);
    }
    if (pathFound) {
      for (size_t i = 0; i < shortestPath.size() - 1; ++i) {
        drawDashedLine(
            window, {(float)shortestPath[i].x, (float)shortestPath[i].y},
            {(float)shortestPath[i + 1].x, (float)shortestPath[i + 1].y},
            sf::Color(180, 180, 180));
      }
    }
    if (currentAppState == AppState::SIMULATING ||
        currentAppState == AppState::SIMULATION_ENDED) {
      for (size_t i = 0; i < dynamicObstacles.size(); ++i) {
        obstacleShapes[i].setPosition(dynamicObstacles[i].center.x,
                                      dynamicObstacles[i].center.y);
        // Optional: color the circle red if it is in a collision
        if (dynamicObstacles[i].isColliding) {
          obstacleShapes[i].setFillColor(sf::Color(255, 80, 80, 200));
        } else {
          obstacleShapes[i].setFillColor(sf::Color(0, 0, 200, 150));
        }
        window.draw(obstacleShapes[i]);
      }
      agentShape.setPosition(agentPosition.x, agentPosition.y);
      agentShape.setRotation(agentRotation);
      window.draw(agentShape);
    }

    window.display();
  }

  return 0;
}

// Implementations of helper functions that were shortened
void drawDashedLine(sf::RenderWindow &window, sf::Vector2f p1, sf::Vector2f p2,
                    sf::Color color, float dashLength, float gapLength) {
  sf::Vector2f direction = p2 - p1;
  float length = std::hypot(direction.x, direction.y);
  if (length < EPSILON)
    return;
  direction /= length;

  float currentLength = 0.0f;
  while (currentLength < length) {
    sf::Vector2f dashStart = p1 + direction * currentLength;
    sf::Vector2f dashEnd =
        p1 + direction * std::min(currentLength + dashLength, length);

    sf::Vertex line[] = {sf::Vertex(dashStart, color),
                         sf::Vertex(dashEnd, color)};
    window.draw(line, 2, sf::Lines);
    currentLength += dashLength + gapLength;
  }
}

bool loadPolygonsFromFile(const std::string &filename,
                          std::vector<Polygon> &polygons) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return false;
  }
  polygons.clear();
  std::string line;
  Polygon currentPolygon;
  while (std::getline(file, line)) {
    if (line.empty()) {
      if (!currentPolygon.vertices.empty()) {
        polygons.push_back(currentPolygon);
        currentPolygon.vertices.clear();
      }
      continue;
    }
    std::istringstream iss(line);
    double x, y;
    if (iss >> x >> y) {
      currentPolygon.vertices.emplace_back(x, y);
    }
  }
  if (!currentPolygon.vertices.empty()) {
    polygons.push_back(currentPolygon);
  }
  file.close();
  return true;
}
