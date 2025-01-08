#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <optional>
#include <assert.h>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <string_view>

using std::cerr;
using std::cout;
using std::endl;
using std::nullopt;
using std::optional;
using std::string;
using std::string_view;
using std::unordered_map;
using std::vector;

using adjacency_matrix = std::vector<std::vector<size_t>>;

// Definition for simulate
void simulate(size_t num_stations, const vector<string> &station_names, const std::vector<size_t> &popularities,
              const adjacency_matrix &mat, const unordered_map<char, vector<string>> &station_lines, size_t ticks,
              const unordered_map<char, size_t> num_trains, size_t num_ticks_to_print, size_t mpi_rank,
              size_t total_processes);

enum LineColor {
    GREEN = 'g',
    YELLOW = 'y',
    BLUE = 'b',
    RED = 'r',
    BROWN = 'w',
    PURPLE = 'p',
    TURQUOISE = 't',
    PINK = 'k',
    LIME = 'l',
    GREY = 'e'
};

const LineColor colors[] = {GREEN, YELLOW, BLUE, RED, BROWN, PURPLE, TURQUOISE, PINK, LIME, GREY};

vector<string> extract_station_names(string &line) {
    constexpr char space_delimiter = ' ';
    vector<string> stations{};
    line += ' '; // add line to make sure last station is processed
    size_t pos;
    while ((pos = line.find(space_delimiter)) != string::npos) {
        stations.push_back(line.substr(0, pos));
        line.erase(0, pos + 1);
    }
    return stations;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << argv[0] << " <input_file>\n";
        std::exit(1);
    }

    // so each process processes the input data and do its own simulation
    int rank, tp;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tp);

    std::ifstream ifs(argv[1], std::ios_base::in);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open " << argv[1] << '\n';
        std::exit(2);
    }

    // Read S & V
    size_t S; // number of stations
    size_t V; // number of lines 
    ifs >> S;
    ifs >> V;

    // Read station names.
    string station;
    std::vector<string> station_names{};
    station_names.reserve(S);
    for (size_t i = 0; i < S; ++i) {
        ifs >> station;
        station_names.emplace_back(station);
    }

    // Read P popularity
    size_t p;
    std::vector<size_t> popularities{};
    popularities.reserve(S);
    for (size_t i = 0; i < S; ++i) {
        ifs >> p;
        popularities.emplace_back(p);
    }

    // Form adjacency mat
    adjacency_matrix mat(S, std::vector<size_t>(S));
    for (size_t src{}; src < S; ++src) {
        for (size_t dst{}; dst < S; ++dst) {
            ifs >> mat[src][dst];
        }
    }

    ifs.ignore(); // skips newline char

    string stations_buf;
    unordered_map<char, vector<string>> station_lines; // stores map of line to stations

    for (size_t i = 0; i < V; ++i) {
        std::getline(ifs, stations_buf);
        vector<string> station_names = extract_station_names(stations_buf);
        station_lines[colors[i]] = std::move(station_names); // move avoids copying, same as emplace_back
    }

    // N time ticks
    size_t N;
    ifs >> N;

    // map of line to number of trains
    unordered_map<char, size_t> num_trains;
    for (size_t i = 0; i < V; ++i) {
        size_t line_num_trains;
        ifs >> line_num_trains;
        num_trains[colors[i]] = line_num_trains;
    }

    size_t num_ticks_to_print;
    ifs >> num_ticks_to_print;

    // Start timing with MPI_Wtime
    double start_time = MPI_Wtime();

    // Call student implementation
    simulate(S, station_names, popularities, mat, station_lines, N, num_trains, num_ticks_to_print, rank, tp);

    // Barrier to make sure all processes are finished before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // End timing with MPI_Wtime
    double end_time = MPI_Wtime();

    // We will use rank 0's timing in the end
    double total_time;
    if (rank == 0) {
        total_time = end_time - start_time;
        cerr << std::fixed << "mpi_time:" << total_time << "s" << endl;
    }

    MPI_Finalize();

    return 0;
}
