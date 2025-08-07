CXX := g++
CXXFLAGS := -std=c++20
LDFLAGS := -lsfml-graphics -lsfml-window -lsfml-system

SRC := main.cpp
TARGET := simulation

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
