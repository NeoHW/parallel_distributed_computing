# High-Performance Computing Projects in C++

This repository contains a collection of high-performance computing projects implemented in C++. These projects leverage different parallel computing techniques such as CUDA, OpenMP, and MPI to achieve efficient execution for computationally intensive tasks.

---

## Projects

### 1. **DNA Matching - CUDA**

- **Overview**: 
  - Implements DNA sequence matching using NVIDIA's CUDA for GPU acceleration.
  - Efficiently matches DNA patterns against large datasets.

- **Key Features**:
  - Parallelized sequence matching using CUDA threads.
  - Optimized memory access using shared memory and coalesced reads.
  - Handles large DNA datasets efficiently.

- **Technologies Used**: 
  - CUDA
  - C++

- **Performance**:
  - Significant speed-up compared to traditional CPU-based approaches.
  - Scalable to handle large datasets with thousands of sequences.

---

### 2. **Particle Collision Simulation - OpenMP**

- **Overview**: 
  - Simulates particle collisions in a defined space using OpenMP for multi-threaded execution.
  - Designed to model and analyze collision events in a physics-based environment.

- **Key Features**:
  - Multi-threaded parallelism with OpenMP.
  - Supports dynamic and static scheduling for thread distribution.
  - Optimized data structures for efficient memory usage.

- **Technologies Used**: 
  - OpenMP
  - C++

- **Performance**:
  - Leverages multi-core CPUs for parallel execution.
  - Configurable workload distribution to adapt to various hardware setups.

---

### 3. **Train Network Simulation - MPI**

- **Overview**: 
  - Simulates a train network with stations, routes, and trains using MPI for distributed memory parallelism.
  - Models real-world train scheduling, routing, and station operations.

- **Key Features**:
  - Distributed simulation of train stations across MPI processes.
  - Inter-process communication for train movements and scheduling.
  - Includes queueing systems for managing train operations at stations.

- **Technologies Used**:
  - MPI
  - C++

- **Performance**:
  - Designed to scale across multiple nodes in a cluster.
  - Efficient communication between processes using MPI.

---
