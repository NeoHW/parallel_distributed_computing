#include "io.h"

/*

    WARNING  - do not change this file!

*/

#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>

void read_args(int argc, char *argv[], Params &params, std::vector<Particle> &particles) {
    int param_threads{};
    int square_size{};
    int param_particles{};
    int param_radius{};
    int param_steps{};

    std::ifstream input_file;
    std::string curr_line{};

    // Check for basic usage
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <INPUT_PATH> <NUM_THREADS>" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "<INPUT_PATH>: " << argv[1] << std::endl;
    std::cout << "<NUM_THREADS>: " << argv[2] << std::endl;

    input_file.open(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open " << argv[1] << " for reading. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        param_threads = std::stoi(argv[2]);
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid argument for <NUM_THREADS>: " << argv[3] << ". Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::out_of_range const &ex) {
        std::cerr << "<NUM_THREADS> out of int range: " << argv[3] << ". Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (param_threads < 1) {
        std::cerr << "<NUM_THREADS> has invalid value: " << argv[3] << ". Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read number of particles
    if (read_param(input_file, curr_line, param_particles) == -1) {
        std::cerr << "Failed to read number of particles. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read square size
    if (read_param(input_file, curr_line, square_size) == -1) {
        std::cerr << "Failed to read square size. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read radius
    if (read_param(input_file, curr_line, param_radius) == -1) {
        std::cerr << "Failed to read particle radius. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read number of steps
    if (read_param(input_file, curr_line, param_steps) == -1) {
        std::cerr << "Failed to read number of steps. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Update the params
    params.param_threads = param_threads;
    params.square_size = square_size;
    params.param_particles = param_particles;
    params.param_radius = param_radius;
    params.param_steps = param_steps;

    // Read all particles
    // Allocate enough space in the vector and make sure the structs are initialized
    particles.resize(param_particles);
    for (int i = 0; i < param_particles; i++) {
        particles[i] = Particle{};
    }
    if (read_particles(input_file, curr_line, particles, param_particles) == -1) {
        std::cerr << "Failed to read particles info. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Print everything out to stdout neatly to check if everything is read
    std::cout << "Number of particles: " << param_particles << std::endl;
    std::cout << "Square size: " << square_size << std::endl;
    std::cout << "Radius: " << param_radius << std::endl;
    std::cout << "Number of steps: " << param_steps << std::endl;
}

int read_particles(std::ifstream &input_file, std::string &line, std::vector<Particle> &particles,
                   int param_particles) {
    for (int i = 0; i < param_particles; i++) {
        std::getline(input_file, line);
        if (input_file.fail()) {
            std::cerr << "Input file failure. Aborting..." << std::endl;
            return -1;
        }

        // 5 columns since we have 5 attributes for each particle
        // Each integer (can have a sign) is separated by a space
        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
        if (tokens.size() != 5) {
            std::cerr << "Invalid number of columns in particle data. Aborting..." << std::endl;
            return -1;
        }

        try {
            particles[i].i = std::stoi(tokens[0]);
            particles[i].loc.x = std::stod(tokens[1]);
            particles[i].loc.y = std::stod(tokens[2]);
            particles[i].vel.x = std::stod(tokens[3]);
            particles[i].vel.y = std::stod(tokens[4]);
        } catch (const std::exception &e) {
            // What exception is this, print it
            std::cerr << "Encountered exception with message " << e.what()
                      << " while reading particle data. Aborting..." << std::endl;
            return -1;
        }
    }
    return 0;
}

int read_param(std::ifstream &input_file, std::string &line, int &param) {
    std::getline(input_file, line);
    if (input_file.fail()) {
        return -1;
    }

    try {
        param = std::stoi(line);
    } catch (const std::exception &e) {
        return -1;
    }
    return 0;
}

/*
For each particle, show on one line the following information, separated by
space:
• tu – the step in the simulation
• i – the index of the particle
• x – position of particle index i on x axis (in μm) at time tu
• y – position of particle index i on y axis (in μm) at time tu
• vx – velocity on the x axis of particle i (in μm/timeunit) at time tu
• vy – velocity on the y axis of particle i (in μm/timeunit ) at time t

    Print x/y/vx/vy values with 8 decimal places and force positive double output.
*/
void print_particles(int step, std::ofstream &output_file, const std::vector<Particle> &particles) {
    for (const Particle &p : particles) {
        output_file << step << " " << p.i << " " << std::fixed << std::setprecision(8)
                    << (p.loc.x == -0.0 ? 0.0 : p.loc.x) << " " << (p.loc.y == -0.0 ? 0.0 : p.loc.y) << " "
                    << (p.vel.x == -0.0 ? 0.0 : p.vel.x) << " " << (p.vel.y == -0.0 ? 0.0 : p.vel.y) << std::endl;
    }
}