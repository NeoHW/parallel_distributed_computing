#pragma once

/*

    WARNING  - do not change this file!

*/

#include <fstream>
#include <string>
#include <vector>

/*
Parameters from input file.
*/
struct Params {
    int param_threads;
    int square_size;
    int param_particles;
    int param_radius;
    int param_steps;
};

/*
Simple tuple type.
*/
struct Vec2 {
    double x;
    double y;
};

/*
Structure to store the information of a particle.
* i – the index of the particle from 0 to N − 1
* x – intial position of particle index i on x axis
* y – intial position of particle index i on y axis
* vx – initial velocity on the x axis of particle i
* vy – initial velocity on the y axis of particle i
*/
struct Particle {
    int i;
    Vec2 loc;
    Vec2 vel;
};

void read_args(int argc, char *argv[], Params &params, std::vector<Particle> &particles);
int read_param(std::ifstream &input_file, std::string &line, int &param);
int read_particles(std::ifstream &input_file, std::string &line, std::vector<Particle> &particles, int param_particles);
void print_particles(int step, std::ofstream &output_file, const std::vector<Particle> &particles);
