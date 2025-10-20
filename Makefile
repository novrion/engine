# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra
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

# Shader configuration
GLSLC = glslc
SHADER_DIR = shaders
VERT_SHADERS = $(wildcard $(SHADER_DIR)/*.vert)
FRAG_SHADERS = $(wildcard $(SHADER_DIR)/*.frag)
COMP_SHADERS = $(wildcard $(SHADER_DIR)/*.comp)
SHADER_OUTPUTS = $(VERT_SHADERS:.vert=.vert.spv) $(FRAG_SHADERS:.frag=.frag.spv) $(COMP_SHADERS:.comp=.comp.spv)



# Phony targets
.PHONY: all shaders test clean run

# Default target
all: shaders $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@



# Compile all shaders
shaders: $(SHADER_OUTPUTS)

$(SHADER_DIR)/%.vert.spv: $(SHADER_DIR)/%.vert
	@mkdir -p $(SHADER_DIR)
	@echo "compiling vertex shader $<..."
	$(GLSLC) -o $@ $<
$(SHADER_DIR)/%.frag.spv: $(SHADER_DIR)/%.frag
	@mkdir -p $(SHADER_DIR)
	@echo "compiling fragment shader $<..."
	$(GLSLC) -o $@ $<
$(SHADER_DIR)/%.comp.spv: $(SHADER_DIR)/%.comp
	@mkdir -p $(SHADER_DIR)
	@echo "compiling compute shader $<..."
	$(GLSLC) -o $@ $<



run: shaders $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS) $(SHADER_OUTPUTS)

# Dependency tracking
-include $(OBJECTS:.o=.d)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) -MM -MT $(@:.d=.o) $< > $@
