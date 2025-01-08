#pragma once

#include <unordered_map>
#include <vector>

#include "io.h"  // Include the shared header for Vec2 and Particle

class SimulationValidator {
   public:
    // Constructor
    SimulationValidator(int num_particles, int square_size, int radius);

    // Destructor
    ~SimulationValidator();

    // Initialize the first set of particles
    void initialize(const std::vector<Particle>& initial_particles);

    // Check the current step's particles vs. the previous step
    void validate_step(const std::vector<Particle>& current_particles);

    // Print output for visualization
    void enable_viz_output(const std::string& filename);

   private:
    // Private helper functions
    bool is_nearly_wall_overlap(Vec2 loc, int square_size, int radius) const;
    bool is_nearly_particle_overlap(Vec2 loc1, Vec2 loc2, int radius) const;
    bool is_definitely_wall_collision(Vec2 loc, Vec2 vel, int square_size, int radius) const;
    double distance(Vec2 loc1, Vec2 loc2) const;
    bool is_definitely_particle_overlap(Vec2 loc1, Vec2 loc2, int radius) const;
    bool is_definitely_particle_moving_closer(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2) const;
    bool is_definitely_particle_collision(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2, int radius) const;
    bool exceedThreshold(double newValue, double oldValue) const;
    void logParticle(const Particle& p, const char* prefix = "") const;
    void logParticleGroup(const std::vector<Particle>& group, const char* prefix = "") const;

    // Private energy and momentum calculations
    double getEnergy(const std::vector<Particle>& particles) const;
    Vec2 getMomentum(const std::vector<Particle>& particles) const;

    // Private helpers
    size_t hash(int pos) const;
    int to_grid_pos(Vec2 loc) const;
    void validate_step_firstpass(const std::vector<Particle>& current_particles);
    void validate_step_secondpass();
    bool bfs_group(int grid_pos, Particle particle);
    void checkEnergyConservation(std::vector<Particle>& groupInPrevStep, std::vector<Particle>& groupInCurrStep) const;
    void checkMomentumConservation(std::vector<Particle>& groupInPrevStep,
                                   std::vector<Particle>& groupInCurrStep) const;
    void checkUnhandledCollisions(std::vector<Particle>& groupInPrevStep, std::vector<Particle>& groupInCurrStep) const;

    // Private member variables
    int num_particles;
    int square_size;
    int radius;

    double tile_size;
    int tiles_in_square;
    size_t grid_bits;

    int step = 0;
    std::vector<Particle> previous_particles;

    std::vector<Particle> current_particles_ordered;
    std::vector<bool> seen_ids;
    std::vector<int> all_pos;
    std::vector<std::vector<Particle>> grid;
    std::vector<bool> checked;
    std::vector<std::pair<int, int>> group;
    std::vector<Particle> groupInPrevStep;
    std::vector<Particle> groupInCurrStep;

    // For output file stream
    std::ofstream viz_output;
};
