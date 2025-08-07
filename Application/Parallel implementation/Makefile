NVCC := nvcc
CXXFLAGS := -std=c++17
LDFLAGS := -lsfml-graphics -lsfml-window -lsfml-system

CU_SRC := Navigation.cu
CPP_SRC := main.cpp
TARGET := simulation_gpu

all: $(TARGET)

$(TARGET): $(CU_SRC) $(CPP_SRC)
	$(NVCC) $(CU_SRC) $(CPP_SRC) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

