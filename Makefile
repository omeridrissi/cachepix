# Compiler and tools
CC      := gcc
AR      := ar
ARFLAGS := rcs

# Directories
SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj

# Library name
LIB_NAME := libcachepix.a
LIB_PATH := $(BUILD_DIR)/$(LIB_NAME)

# Flags
CFLAGS := -O3 -Wall -Wextra -I$(INC_DIR) -lm

ifeq ($(ARCH),x86)
    CFLAGS += -msse2
endif

ifeq ($(ARCH),x86_avx2)
    CFLAGS += -mavx2
endif

ifeq ($(ARCH),arm)
    CFLAGS += -mfpu=neon
endif

# SIMD flags (adjust as needed)
# Examples:
#   -msse4.2
#   -mavx2
#   -march=native
SIMD_FLAGS := -march=native

# Source and object files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Default target
all: $(LIB_PATH)

# Build static library
$(LIB_PATH): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(AR) $(ARFLAGS) $@ $^

# Compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(SIMD_FLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

