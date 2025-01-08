#pragma once

/**
 * This file contains default collision detection and resolution logic.
 */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>      // std::setprecision

#include "io.h"

// Is particle overlapping with a wall
inline bool is_wall_overlap(Vec2 loc, int square_size, int radius);

// Is particle colliding with a wall
inline bool is_wall_collision(Vec2 loc, Vec2 vel, int square_size, int radius);

// Make particle not collide with wall
// PRECONDITION: Can be called with no preconditions
inline void resolve_wall_collision(Vec2 loc, Vec2& vel, int square_size, int radius);

// Are the particles overlapping
inline bool is_particle_overlap(Vec2 loc1, Vec2 loc2, int radius);

// Are the particles moving closer together
inline bool is_particle_moving_closer(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2);

// Are the particles colliding
inline bool is_particle_collision(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2, int radius);

// Make particles not collide with each other
// PRECONDITION: Must only be called if particles are overlapping
inline void resolve_particle_collision(Vec2 loc1, Vec2& vel1, Vec2 loc2, Vec2& vel2);

// Get the total energy of a group of particles
inline double getEnergy(std::vector<Particle>& particles);

// Get the total momentum of a group of particles
inline Vec2 getMomentum(std::vector<Particle>& particles);

/*******************
 * IMPLEMENTATIONS *
 *******************/

/**
 * Whether a particle is overlapping with a wall
 * - loc: The location of the particle
 * - square_size: The length of the simulation area
 * - radius: The radius of the particle
 */
inline bool is_wall_overlap(Vec2 loc, int square_size, int radius) {
    return loc.x - radius <= 0 || loc.x + radius >= square_size || loc.y - radius <= 0 || loc.y + radius >= square_size;
}

/**
 * Whether a particle is colliding with a wall
 * - loc: The location of the particle
 * - square_size: The length of the simulation area
 * - radius: The radius of the particle
 */
inline bool is_wall_collision(Vec2 loc, Vec2 vel, int square_size, int radius) {
    if (loc.x - radius <= 0 && vel.x < 0) {
        return true;
    } else if (loc.x + radius >= square_size && vel.x > 0) {
        return true;
    }
    if (loc.y - radius <= 0 && vel.y < 0) {
        return true;
    } else if (loc.y + radius >= square_size && vel.y > 0) {
        return true;
    }
    return false;
}

/**
 * Implements the rules to change the velocity of a particle that collides with a wall
 * - loc: The location of the particle
 * - vel: The velocity of the particle
 * - square_size: The length of the simulation area
 * - radius: The radius of the particle
 */
inline void resolve_wall_collision(Vec2 loc, Vec2& vel, int square_size, int radius) {
    if (loc.x - radius <= 0) {
        vel.x = std::abs(vel.x);
    } else if (loc.x + radius >= square_size) {
        vel.x = -std::abs(vel.x);
    }
    if (loc.y - radius <= 0) {
        vel.y = std::abs(vel.y);
    } else if (loc.y + radius >= square_size) {
        vel.y = -std::abs(vel.y);
    }
}

/**
 * Whether two particles are overlapping
 * - loc1: The location of the first particle
 * - loc2: The location of the second particle
 * - radius: The radius of the particles
 */
inline bool is_particle_overlap(Vec2 loc1, Vec2 loc2, int radius) {
    double dx = loc2.x - loc1.x;
    double dy = loc2.y - loc1.y;
    double sq_distance = dx * dx + dy * dy;
    return sq_distance <= (radius * 2) * (radius * 2);
}

/**
 * Whether two particles are moving closer together, with some tolerance
 * - loc1: The location of the first particle
 * - vel1: The velocity of the first particle
 * - loc2: The location of the second particle
 * - vel2: The velocity of the second particle
 */
inline bool is_particle_moving_closer(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2) {
    double dx = loc2.x - loc1.x;
    double dy = loc2.y - loc1.y;
    double dvx = vel2.x - vel1.x;
    double dvy = vel2.y - vel1.y;
    double dot_product = dvx * dx + dvy * dy;
    return dot_product < -0.0000001;
}

/**
 * Whether two particles are colliding
 * - loc1: The location of the first particle
 * - vel1: The velocity of the first particle
 * - loc2: The location of the second particle
 * - vel2: The velocity of the second particle
 * - radius: The radius of the particles
 */
inline bool is_particle_collision(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2, int radius) {
    return is_particle_overlap(loc1, loc2, radius) && is_particle_moving_closer(loc1, vel1, loc2, vel2);
}

/**
 * Implements the rules to change the velocity of two particles that collide
 * - loc1: The location of the first particle
 * - vel1: The velocity of the first particle
 * - loc2: The location of the second particle
 * - vel2: The velocity of the second particle
 */
inline void resolve_particle_collision(Vec2 loc1, Vec2& vel1, Vec2 loc2, Vec2& vel2) {
    double dx = loc2.x - loc1.x;
    double dy = loc2.y - loc1.y;
    double dvx = vel2.x - vel1.x;
    double dvy = vel2.y - vel1.y;
    double dot_product = dvx * dx + dvy * dy;
    if (dot_product >= 0) {
        dot_product = 0;
    }

    // Calculate the new velocities of the particles after the collision
    double collision_scale = dot_product / (dx * dx + dy * dy);
    double collision_x = collision_scale * dx;
    double collision_y = collision_scale * dy;

    vel1.x += collision_x;
    vel1.y += collision_y;
    vel2.x -= collision_x;
    vel2.y -= collision_y;
}

/**
 * Get the total energy of a group of particles
 * - particles: The particles to calculate the energy of
 */
inline double getEnergy(std::vector<Particle>& particles) {
    double energy = 0;
    for (Particle& p : particles) {
        energy += p.vel.x * p.vel.x + p.vel.y * p.vel.y;
    }
    return energy;
}

/**
 * Get the total momentum of a group of particles
 * - particles: The particles to calculate the momentum of
 */
inline Vec2 getMomentum(std::vector<Particle>& particles) {
    Vec2 momentum = {0.0, 0.0};
    for (Particle& p : particles) {
        momentum.x += p.vel.x;
        momentum.y += p.vel.y;
    }
    return momentum;
}
