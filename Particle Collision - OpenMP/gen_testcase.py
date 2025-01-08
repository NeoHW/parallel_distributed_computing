#!/usr/bin/env python3

import argparse
import random
import sys


def main(num_particles, grid_length, particle_radius, num_time_steps, min_velocity, max_velocity):
    print(num_particles)
    print(grid_length)
    print(particle_radius)
    print(num_time_steps)

    for n in range(num_particles):
        print(
            n,
            random.random() * grid_length,
            random.random() * grid_length,
            random.random() * (max_velocity - min_velocity) + min_velocity,
            random.random() * (max_velocity - min_velocity) + min_velocity,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Particle simulation input file generator.")

    parser.add_argument("num_particles", type=int,
                        help="Number of particles (N)")
    parser.add_argument("grid_length", type=int, help="Length of the grid (L)")
    parser.add_argument("particle_radius", type=int,
                        help="Radius of the particles (R)")
    parser.add_argument("num_time_steps", type=int,
                        help="Number of time steps (T)")
    parser.add_argument("min_velocity", type=float,
                        help="Minimum velocity of the particles")
    parser.add_argument("max_velocity", type=float,
                        help="Maximum velocity of the particles")

    args = parser.parse_args()

    main(
        args.num_particles,
        args.grid_length,
        args.particle_radius,
        args.num_time_steps,
        args.min_velocity,
        args.max_velocity
    )

    # Print particle density: (num_particles * pi * R ^ 2) / L ^ 2
    density = ((args.num_particles * 3.14159 *
               (args.particle_radius**2))) / (args.grid_length**2)
    print(f"\n\nParticle density: {density}\n\n", file=sys.stderr)

    if density < 0.7 or args.num_particles < 10000 or args.num_time_steps < 100:
        print(
            "\033[93mWarning: This testcase is not valid for performance checks.\033[0m",
            file=sys.stderr)
        if density < 0.7:
            print(
                "\033[93mReason: Particle density is less than 0.7.\033[0m",
                file=sys.stderr)
        if args.num_particles < 10000:
            print(
                "\033[93mReason: Number of particles is less than 10,000.\033[0m",
                file=sys.stderr)
        if args.num_time_steps < 100:
            print(
                "\033[93mReason: Number of time steps is less than 100.\033[0m",
                file=sys.stderr)
    else:
        print(
            "\033[92mThis testcase is valid for performance checks.\033[0m",
            file=sys.stderr)
