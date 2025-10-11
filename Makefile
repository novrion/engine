# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

# Vulkan setup
VULKAN_SDK = $(shell echo $$VULKAN_SDK)
CXXFLAGS += -I$(VULKAN_SDK)/include
CXXFLAGS += -DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
CXXFLAGS += -DVULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1

# Find package includes (adjust paths as needed)
CXXFLAGS += $(shell pkg-config --cflags glfw3 2>/dev/null || echo "")

# Project settings
TARGET = Engine
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Shader compilation
GLSLANG = glslangValidator
SHADER_DIR = shaders
SHADERS = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag $(SHADER_DIR)/*.comp)
SHADER_SPVS = $(SHADERS:.vert=.vert.spv)
SHADER_SPVS := $(SHADER_SPVS:.frag=.frag.spv)
SHADER_SPVS := $(SHADER_SPVS:.comp=.comp.spv)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Shader compilation rules
$(SHADER_DIR)/%.vert.spv: $(SHADER_DIR)/%.vert
	$(GLSLANG) --target-env vulkan1.0 -o $@ $

$(SHADER_DIR)/%.frag.spv: $(SHADER_DIR)/%.frag
	$(GLSLANG) --target-env vulkan1.0 -o $@ $

$(SHADER_DIR)/%.comp.spv: $(SHADER_DIR)/%.comp
	$(GLSLANG) --target-env vulkan1.0 -o $@ $

# Phony targets
.PHONY: all shaders test clean run

shaders: $(SHADER_SPVS)

test: $(TARGET)
	./$(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS) $(SHADER_SPVS)

# Dependency tracking
-include $(OBJECTS:.o=.d)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) -MM -MT $(@:.d=.o) $< > $@
