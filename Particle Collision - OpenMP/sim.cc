#include <omp.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "collision.h"
#include "io.h"
#include "sim_validator.h"

// calculate grid cell index for a given location
inline int calculate_grid_index(Vec2 loc, int num_cell_in_grid_row, int cell_size) {
    int total_length = cell_size * num_cell_in_grid_row;
    double xLoc = std::max(0.1, std::min(loc.x, static_cast<double>(total_length - 0.1)));
    double yLoc = std::max(0.1, std::min(loc.y, static_cast<double>(total_length - 0.1)));

    int x_cell_index = std::floor(xLoc / cell_size);
    int y_cell_index = (total_length - yLoc) / cell_size;

    return y_cell_index * num_cell_in_grid_row + x_cell_index;
}

// returns a map of grid index to neighbouring grid indices, that are only >= the current grid index
std::vector<std::vector<int>> get_neighbouring_grid_indices(int num_cell_in_grid_row, int total_cells) {
    std::vector<std::vector<int>> neighbours_map(total_cells);
    
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < num_cell_in_grid_row; row++) {
        for (int col = 0; col < num_cell_in_grid_row; col++) {
            int grid_index = row * num_cell_in_grid_row + col;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int new_row = row + i;
                    int new_col = col + j;
                    if (new_row >= 0 && new_row < num_cell_in_grid_row && new_col >= 0 &&
                        new_col < num_cell_in_grid_row) {
                        // calculate the index in 1D flattened array
                        int neighbour_index = new_row * num_cell_in_grid_row + new_col;
                        if (neighbour_index < grid_index) continue;
                        neighbours_map[grid_index].push_back(neighbour_index);
                    }
                }
            }
        }
    }

    return neighbours_map;
}

int main(int argc, char* argv[]) {
    /* debug output file*/
    // std::ofstream ofs{"debug_log.txt"};
    // auto cout_buff = std::cout.rdbuf();
    // std::cout.rdbuf(ofs.rdbuf());
    /* debug output file*/

    // Read arguments and input file
    Params params{};
    std::vector<Particle> particles;
    read_args(argc, argv, params, particles);

    // Set number of threads
    omp_set_num_threads(params.param_threads);

#if CHECK == 1
    // Initialize collision checker
    SimulationValidator validator(params.param_particles, params.square_size, params.param_radius);
    // Initialize with starting positions
    validator.initialize(particles);
    // Uncomment the line below to enable visualization (makes program much slower)
    // validator.enable_viz_output("test.out");
#endif

    // Define grid parameters
    int cell_size = 2 * params.param_radius;  // Cell size slightly larger than particle diameter
    int num_cell_in_grid_row = std::ceil(params.square_size / static_cast<double>(cell_size));  // Number of cells along one dimension
    int total_cells = num_cell_in_grid_row * num_cell_in_grid_row;
    
    std::vector<std::vector<int>> neighbours_map = get_neighbouring_grid_indices(num_cell_in_grid_row, total_cells);

    // simulate time steps
    for (int i = 1; i < params.param_steps + 1; i++) {

        // Update positions of all particles
        #pragma omp parallel for
        for (size_t p = 0; p < particles.size(); p++) {
            particles[p].loc.x += particles[p].vel.x;
            particles[p].loc.y += particles[p].vel.y;
        }

        // Initialize thread-local grids
        std::vector<std::vector<std::vector<int>>> thread_local_grids(params.param_threads, std::vector<std::vector<int>>(total_cells));

        // Assign each particle to their grid index
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::vector<std::vector<int>>& local_grid = thread_local_grids[thread_id];

            // Assign particles to grid cells
            #pragma omp for nowait
            for (size_t p = 0; p < particles.size(); p++) {
                int grid_index = calculate_grid_index(particles[p].loc, num_cell_in_grid_row, cell_size);
                local_grid[grid_index].push_back(p);
            }
        }

        std::vector<std::vector<int>> grid(total_cells); // each cell in grid will be stored in this 1D flattened array, each contains a list of particle indices in that cell

        // Merge thread-local grids into the global grid
        #pragma omp parallel for
        for (int idx = 0; idx < total_cells; idx++) {
            for (int t = 0; t < params.param_threads; t++) {
                grid[idx].insert(grid[idx].end(), thread_local_grids[t][idx].begin(), thread_local_grids[t][idx].end());
            }
        }

        /*
        At this point, each cell in grid will have a vector of particles that currently is inside the cell
        */

        // Initialize thread caches for overlaps (each vector holds a thread local cache)
        std::vector<std::vector<int>> thread_particle_wall_overlap_caches(params.param_threads);
        std::vector<std::vector<std::vector<std::pair<int, int>>>> thread_particle_particle_overlap_caches(
                params.param_threads, std::vector<std::vector<std::pair<int, int>>>(total_cells));
        
        // Populate the caches
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

            // Each thread's per-thread caches
            std::vector<int>& local_particle_wall_overlap_cache = thread_particle_wall_overlap_caches[thread_id];
            std::vector<std::vector<std::pair<int, int>>>& local_particle_particle_overlap_cache = thread_particle_particle_overlap_caches[thread_id];

            // Check for particle-wall overlaps
            #pragma omp for nowait
            for (size_t p = 0; p < particles.size(); p++) {
                if (is_wall_overlap(particles[p].loc, params.square_size, params.param_radius)) {
                    local_particle_wall_overlap_cache.push_back(p);
                }
            }

            // Check for particle-particle overlaps
            #pragma omp for schedule(static)
            for (int cell_index = 0; cell_index < total_cells; cell_index++) {
                // For each particle in the current cell
                const std::vector<int>& cell_particles = grid[cell_index];
                for (size_t i = 0; i < cell_particles.size(); i++) {
                    int particle1_index = cell_particles[i];
                    Particle& particle1 = particles[particle1_index];

                    // Check overlaps within the same cell
                    for (size_t j = i + 1; j < cell_particles.size(); j++) {
                        int particle2_index = cell_particles[j];
                        Particle& particle2 = particles[particle2_index];

                        if (is_particle_overlap(particle1.loc, particle2.loc, params.param_radius)) {
                            local_particle_particle_overlap_cache[cell_index].emplace_back(particle1_index, particle2_index);
                        }
                    }

                    // Check overlaps with particles in neighboring cells
                    for (int neighbour_index : neighbours_map[cell_index]) {
                        if (neighbour_index == cell_index) continue;  // Skip the current cell

                        const std::vector<int>& neighbour_particles = grid[neighbour_index];
                        for (int particle2_index : neighbour_particles) {
                            Particle& particle2 = particles[particle2_index];

                            if (is_particle_overlap(particle1.loc, particle2.loc, params.param_radius)) {
                                local_particle_particle_overlap_cache[cell_index].emplace_back(particle1_index, particle2_index);
                            }
                        }
                    }
                }
            }
        }

        // Merge per-thread caches into global caches
        std::vector<int> particle_wall_overlap_cache;
        std::vector<std::vector<std::pair<int, int>>> particle_particle_overlap_cache(total_cells);

        // Merge particle-wall overlap caches
        for (int t = 0; t < params.param_threads; t++) {
            particle_wall_overlap_cache.insert(particle_wall_overlap_cache.end(),
                    thread_particle_wall_overlap_caches[t].begin(),
                    thread_particle_wall_overlap_caches[t].end());
        }

        // Merge particle-particle overlap caches
        for (int t = 0; t < params.param_threads; t++) {
            for (int cell_idx = 0; cell_idx < total_cells; cell_idx++) {
                particle_particle_overlap_cache[cell_idx].insert(
                    particle_particle_overlap_cache[cell_idx].end(),
                    thread_particle_particle_overlap_caches[t][cell_idx].begin(),
                    thread_particle_particle_overlap_caches[t][cell_idx].end());
            }
        }

        bool have_collisions = true;

        while (have_collisions) {
            have_collisions = false;

            // Check for wall collision
            #pragma omp parallel
            {
                bool local_have_collisions = false;

                #pragma omp for nowait
                for (size_t idx = 0; idx < particle_wall_overlap_cache.size(); idx++) {
                    int particle_index = particle_wall_overlap_cache[idx];
                    Particle& particle = particles[particle_index];

                    if (is_wall_collision(particle.loc, particle.vel, params.square_size, params.param_radius)) {
                        resolve_wall_collision(particle.loc, particle.vel, params.square_size, params.param_radius);
                        local_have_collisions = true;
                    }
                }

                // Combine results
                if (local_have_collisions) {
                    have_collisions = true;
                }
            }

            // Check for particle collision in same cell and neighbouring cells
            #pragma omp parallel
            {
                bool local_have_collisions = false;
                #pragma omp for schedule(static)
                for (size_t cell_index = 0; cell_index < particle_particle_overlap_cache.size(); cell_index++) {
                    for (const auto& overlap_pair : particle_particle_overlap_cache[cell_index]) {
                        int p1_index = overlap_pair.first;
                        int p2_index = overlap_pair.second;

                        Particle& particle1 = particles[p1_index];
                        Particle& particle2 = particles[p2_index];

                        if (is_particle_moving_closer(particle1.loc, particle1.vel, particle2.loc, particle2.vel)) {
                            resolve_particle_collision(particle1.loc, particle1.vel, particle2.loc, particle2.vel);
                            local_have_collisions = true;
                        }
                    }
                }

                if (local_have_collisions) {
                    have_collisions = true;
                }
            }
        }

// After simulating each timestep, you must call:
#if CHECK == 1
        validator.validate_step(particles);
#endif
    }

    /* For Debug */
    // std::cout.rdbuf(cout_buff);
    /* For Debug */
}