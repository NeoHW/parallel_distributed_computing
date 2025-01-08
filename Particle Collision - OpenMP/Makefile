CXX := g++
CXXFLAGS := -Wall -Wextra -Werror -pedantic -std=c++20 -fopenmp -g -Og
RELEASEFLAGS := -O3

# List of source files
SRCS := main.cc io.cc
HEADERS := io.h collision.h sim_validator.h

# Object files
OBJS := $(SRCS:.cc=.o)

.PHONY: all clean

all: release

# List of executables (actual binary names)
EXECUTABLES := sim 
PERF_EXECUTABLES := $(EXECUTABLES:%=%.perf)

TARGETS := $(EXECUTABLES) 
PERF_TARGETS := $(PERF_EXECUTABLES)

# Release build target
release: CXXFLAGS += $(RELEASEFLAGS)
release: $(TARGETS) $(PERF_TARGETS)

# Debug build target
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: $(TARGETS) $(PERF_TARGETS)

# How to compile .o and .o.perf object files
%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -DCHECK=1 $(RELEASEFLAGS) -c $< -o $@
%.o.perf: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -DCHECK=0 $(RELEASEFLAGS) -c $< -o $@

# How to compile non-perf and perf executables
$(EXECUTABLES): %: %.o io.o sim_validator.a
	$(CXX) $(CXXFLAGS) -DCHECK=1 $(RELEASEFLAGS) -o $@ $^
$(PERF_EXECUTABLES): %.perf: %.o.perf io.o
	$(CXX) $(CXXFLAGS) -DCHECK=0 $(RELEASEFLAGS) -o $@ $^
clean:
	$(RM) *.o *.o.perf $(EXECUTABLES) $(PERF_EXECUTABLES)
