#include <vector>
#include <string>
#include <ostream>
#include <sstream>
#include <iostream>
#include <queue>
#include <deque>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <mpi.h>
#include "platform_load_time_gen.hpp"

#define DEBUG_LOG(x) std::cout << x << std::endl
#define OMPI_SKIP_MPICXX 1

using std::string;
using std::unordered_map;
using std::map;
using std::vector;
using std::priority_queue;
using std::deque;
using std::cout;
using std::endl;
using adjacency_matrix = std::vector<std::vector<size_t>>;

struct Train {
    int train_id;
    int line;
    int current_station;
    int next_station;
    int direction; // 1 for forward, -1 for backward
    int station_arrival_time;
    int state; // 0 for holding area, 1 for platform, 2 for link
    int load_time_remaining;

    Train(int id, int line, int station_arrival_time, int start_station, int direction, int next_station)
        : train_id(id), line(line), station_arrival_time(station_arrival_time),
        current_station(start_station), next_station(next_station), direction(direction),
        load_time_remaining(-1), state(0) {}

    // Default constructor
    Train() : train_id(-1), line(-1), current_station(-1), next_station(-1), direction(0), station_arrival_time(-1), load_time_remaining(-1), state(-1) {}

    /// Copy constructor
    Train(const Train& other) {
        train_id = other.train_id;
        line = other.line;
        current_station = other.current_station;
        next_station = other.next_station;
        direction = other.direction;
        station_arrival_time = other.station_arrival_time;
        load_time_remaining = other.load_time_remaining;
        state = other.state;
    }
    
    // Copy assignment operator
    Train& operator=(const Train& other) {
        train_id = other.train_id;
        line = other.line;
        current_station = other.current_station;
        next_station = other.next_station;
        direction = other.direction;
        station_arrival_time = other.station_arrival_time;
        load_time_remaining = other.load_time_remaining;
        state = other.state;
        return *this;
    }

    string toString() const {
        std::ostringstream oss;
        oss << "Train[ID=" << train_id << ", Line=" << line 
            << ", Current=" << current_station << ", Next=" << next_station 
            << ", Dir=" << direction << ", ArrivalTime=" << station_arrival_time 
            << ", LoadTime=" << load_time_remaining 
            << ", State=" << state << "]";
        return oss.str();
    }
};

// Minimal structure for MPI communication
struct TrainComm {
    int train_id;
    int state; // 0 for holding area, 1 for platform, 2 for link
    int direction; // 1 for forward, -1 for backward
    int station_arrival_time;
    int current_station;
    int next_station;

    // Default constructor
    TrainComm() 
        : train_id(-1), current_station(-1), next_station(-1), 
          direction(0), station_arrival_time(-1), state(-1) {}

    // Constructor from Train object
    TrainComm(const Train& t) 
        : train_id(t.train_id), current_station(t.current_station), 
          next_station(t.next_station), direction(t.direction), 
          station_arrival_time(t.station_arrival_time), state(t.state) {}

    // Copy constructor
    TrainComm(const TrainComm& other) 
        : train_id(other.train_id), state(other.state), direction(other.direction), 
          station_arrival_time(other.station_arrival_time), 
          current_station(other.current_station), next_station(other.next_station) {
    }

    // Move constructor
    TrainComm(TrainComm&& other) noexcept 
        : train_id(other.train_id), state(other.state), direction(other.direction), 
          station_arrival_time(other.station_arrival_time), 
          current_station(other.current_station), next_station(other.next_station) {
        // Reset the source object
        other.reset();
    }

    // Copy assignment operator
    TrainComm& operator=(const TrainComm& other) {
        if (this == &other) return *this; // Self-assignment check
        train_id = other.train_id;
        state = other.state;
        direction = other.direction;
        station_arrival_time = other.station_arrival_time;
        current_station = other.current_station;
        next_station = other.next_station;
        return *this;
    }

    // Move assignment operator
    TrainComm& operator=(TrainComm&& other) noexcept {
        if (this == &other) return *this; // Self-assignment check
        train_id = other.train_id;
        state = other.state;
        direction = other.direction;
        station_arrival_time = other.station_arrival_time;
        current_station = other.current_station;
        next_station = other.next_station;
        // Reset the source object
        other.reset();
        return *this;
    }

    // Reset helper function
    void reset() {
        train_id = -1;
        state = -1;
        direction = 0;
        station_arrival_time = -1;
        current_station = -1;
        next_station = -1;
    }

    // Update fields of Train from this TrainComm
    void updateTrain(Train& t) const {
        t.train_id = train_id;
        t.state = state;
        t.direction = direction;
        t.station_arrival_time = station_arrival_time;
        t.current_station = current_station;
        t.next_station = next_station;
    }

    // String representation
    string toString() const {
        std::stringstream ss;
        ss << "TrainComm[ID=" << train_id
           << ", Current=" << current_station
           << ", Next=" << next_station
           << ", Dir=" << direction
           << ", ArrivalTime=" << station_arrival_time
           << ", State=" << state
           << "]";
        return ss.str();
    }
};


void create_mpi_train_type(MPI_Datatype& mpi_train_type) {
    const int    nitems=6;
    int          blocklengths[6] = {1,1,1,1,1,1};
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint     offsets[6];

    offsets[0] = offsetof(TrainComm, train_id);
    offsets[1] = offsetof(TrainComm, state);
    offsets[2] = offsetof(TrainComm, direction);
    offsets[3] = offsetof(TrainComm, station_arrival_time);
    offsets[4] = offsetof(TrainComm, current_station);
    offsets[5] = offsetof(TrainComm, next_station);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_train_type);
    MPI_Type_commit(&mpi_train_type);
}

class TrainCompare {
public:
    bool operator()(const Train& a, const Train& b) const {
        if (a.station_arrival_time == b.station_arrival_time) {
            return a.train_id > b.train_id; // lower ID prioritized
        }
        return a.station_arrival_time > b.station_arrival_time; // earlier arrival time prioritized
    }
};

struct Link {
    int end_station;
    int distance;
    bool is_occupied;
    int next_free_time;
    Train occupied_by;

    Link(int end, int dist): end_station(end), distance(dist), is_occupied(false), next_free_time(0) {}

    string toString() const {
        std::ostringstream oss;
        oss << "Link[End=" << end_station << ", Dist=" << distance 
            << ", Occupied=" << (is_occupied ? "true" : "false") 
            << ", NextFree=" << next_free_time << "]";
        return oss.str();
    }
};

// one platform & link per destination
struct Platform {
    Link link;
    priority_queue<Train, vector<Train>, TrainCompare> holding_area; // Train in holding area
    PlatformLoadTimeGen pltg;
    Train current_train;

    Platform(int popularity, int end_station, int distance)
        : link(end_station, distance), pltg(popularity) {}

    int get_load_time(int train_id) {
        int load_time = pltg.next(train_id);
        return load_time;
    }

    string toString() const {
        std::ostringstream oss;
        oss << link.toString() << "Platform[HoldingArea=" << holding_area.size() << " trains"
            << ", HasCurrentTrain=" << (current_train.train_id != -1) << "]";
        return oss.str();
    }
};

// insert train to flat buffer for specific station
void insert_train_to_buffer(TrainComm* buffer, int process_index, int station_index, int buffer_size_per_process, int buffer_size_per_station, Train train) {
    // Calculate start index based on station index and station buffer size
    int start_index = station_index * buffer_size_per_station;

    // Search for first free slot
    for (int i = 0; i < buffer_size_per_station; ++i) {
        int buffer_index = start_index + i;
        if (buffer[buffer_index].train_id == -1) {
            buffer[buffer_index] = TrainComm(train);  // Insert train here
            return;
        }
    }
    cout << "Error: Could not insert train to buffer for station " << station_index << endl;
}

void insert_train_to_buffer_first_free(TrainComm* buffer, int& next_free_index, Train train) {
    buffer[next_free_index] = TrainComm(train);
    ++next_free_index;
}

struct Station {
    int station_id;
    int popularity;
    vector<Platform> platforms;
    priority_queue<Train, vector<Train>, TrainCompare> incoming_trains; // Trains that are on the way

    Station(int id, int pop, int num_stations) : station_id(id), popularity(pop) {
        platforms.reserve(num_stations);
    }

    Platform* get_platform_for_destination(int dest) {
        for(auto& platform : platforms) {
            if(platform.link.end_station == dest) {
                return &platform;
            }
        }
        cout << "Error: Could not find platform for destination " << dest << " at station " << station_id << endl;
        return nullptr;
    }

    void handle_platform_link_status(Station& station, int tick) {
        for (Platform& platform : station.platforms) {
            if (platform.link.is_occupied && platform.link.next_free_time <= tick) {
                // get rid of the train on the link before freeing link status
                platform.link.occupied_by = Train();
                platform.link.is_occupied = false;
            }
        }
    }

    // puts the train in the holding PQ of the platform, handles direction change
    void handle_train_arrival(Train& train, const vector<unordered_map<int, int>>& forward_routes,
                            const vector<unordered_map<int, int>>& backward_routes, 
                            vector<int> start_stations, vector<int> end_stations, 
                            TrainComm* processToRoot_outgoing_trains, int& processToRoot_outgoing_trains_nextFreeIndex, vector<int>& train_id_to_line) {
        
        Train train_copy = train;
        train_copy.current_station = station_id;
        train_copy.load_time_remaining = -1;
        train_copy.state = 0;
        train_copy.line = train_id_to_line[train_copy.train_id];

        if (train_copy.direction == 1) {
            if (train_copy.current_station == end_stations[train_copy.line]) {
                train_copy.direction = -1;
                train_copy.next_station = backward_routes[train_copy.line].at(train_copy.current_station);
            } else {
                train_copy.next_station = forward_routes[train_copy.line].at(train_copy.current_station);
            }
        } else {
            if (train_copy.current_station == start_stations[train_copy.line]) {
                train_copy.direction = 1;
                train_copy.next_station = forward_routes[train_copy.line].at(train_copy.current_station);
            } else {
                train_copy.next_station = backward_routes[train_copy.line].at(train_copy.current_station);
            }
        }

        Platform* platform = get_platform_for_destination(train_copy.next_station);
        
        insert_train_to_buffer_first_free(processToRoot_outgoing_trains, processToRoot_outgoing_trains_nextFreeIndex, train_copy);
        platform->holding_area.emplace(train_copy);
    }

     void process_platforms(TrainComm* processToRoot_outgoing_trains, int& processToRoot_outgoing_trains_nextFreeIndex, int tick) {
        for (Platform& platform : platforms) {
            if (platform.current_train.train_id != -1) {
                Train& actual_train = platform.current_train;
                
                // if train is loading, decrement load time
                if (actual_train.load_time_remaining > 1) {
                    --actual_train.load_time_remaining;
                    continue;
                }

                // if train is loaded, move to link
                if (!platform.link.is_occupied) {
                    platform.link.is_occupied = true;
                    actual_train.state = 2;
                    actual_train.station_arrival_time = tick + platform.link.distance;
                    platform.link.occupied_by = actual_train;
                    platform.link.next_free_time = tick + platform.link.distance;
                    
                    Train train_copy = actual_train;
                    insert_train_to_buffer_first_free(processToRoot_outgoing_trains, processToRoot_outgoing_trains_nextFreeIndex, train_copy);
                    
                    // reset platform current train to dummy train
                    platform.current_train = Train();
                }
            }

            // moving from holding area to platform
            if (!platform.holding_area.empty() && platform.current_train.train_id == -1) {
                Train next_train = platform.holding_area.top();
                platform.holding_area.pop();
                int load_time = platform.get_load_time(next_train.train_id);
                next_train.load_time_remaining = load_time;
                next_train.state = 1;
                platform.current_train = next_train;
                insert_train_to_buffer_first_free(processToRoot_outgoing_trains, processToRoot_outgoing_trains_nextFreeIndex, next_train);
            }
        }
    }
};

// Function to format train position for output
string format_train_position(const Train& train, const vector<string>& station_names) {
    std::ostringstream oss;
    char line_prefix = (train.line == 0) ? 'g' : (train.line == 1) ? 'y' : 'b';
    oss << line_prefix << train.train_id << "-";

    switch(train.state) {
        // Train is on a link
        case 2:
            oss << station_names.at(train.current_station) << "->" << station_names.at(train.next_station);
            break;
        // Train is at a platform
        case 1:
            oss << station_names.at(train.current_station) << "%";
            break;
        // Train is in a holding area
        case 0:
            oss << station_names.at(train.current_station) << "#";
            break;
    }

    return oss.str();
}

// Function to collect and print train positions for a specific tick
void print_tick_output(int tick, const vector<Train>& all_trains, const vector<string>& station_names) {
    vector<string> train_positions;
    for (const auto& train : all_trains) {
        if (train.train_id != -1) {
            train_positions.push_back(format_train_position(train, station_names));
        }
    }

    std::sort(train_positions.begin(), train_positions.end()); // Sort lexicographically

    cout << tick << ": ";
    for (const auto& pos : train_positions) {
        cout << pos << " ";
    }
    cout << endl;
}

// Function to distribute station indices and create a process-to-station mapping
vector<int> distribute_stations(int num_stations_per_process, int original_num_stations, int mpi_rank) {
    vector<int> local_stations;
    int start_index = mpi_rank * num_stations_per_process;
    for (int i = 0; i < num_stations_per_process; ++i) {
        if (start_index + i >= original_num_stations) {
            break;
        }
        local_stations.push_back(start_index+i);
    }
    return local_stations;
}

/**
 * @param num_stations number of stations
 * @param station_names names of the stations (use as index to access station)
 * @param popularities loading time for each station
 * @param mat adjacency matrix of the stations
 * @param station_lines map of line to stations
 * @param ticks number of time ticks to simulate
 * @param num_trains map of line to number of trains on that line
 * @param num_ticks_to_print number of ticks to print
 * @param mpi_rank rank of the current process
 * @param total_processes total number of processes
 */
void simulate(size_t num_stations, const vector<string> &station_names, const std::vector<size_t> &popularities,
              const adjacency_matrix &mat, const unordered_map<char, vector<string>> &station_lines, size_t ticks,
              unordered_map<char, size_t> num_trains, size_t num_ticks_to_print, size_t mpi_rank,
              size_t total_processes) {

    int TRAIN_SPAWN_NUM = 0;
    int MAX_NUMBER_OF_LINKS = -1;
    int i, j;

    MPI_Datatype mpi_train_type;
    create_mpi_train_type(mpi_train_type);

    for (const auto& [line, num] : num_trains) {
        TRAIN_SPAWN_NUM += num;
    }
    vector<Train> all_trains(TRAIN_SPAWN_NUM); // holds all trains data, updates when root receives new data

    // Create stations
    vector<Station> stations; // mapping of station index to station
    unordered_map<string, int> station_name_to_index; // mapping of station name to index
    
    for (int src = 0; src < num_stations; ++src) {
        int num_links = 0;
        stations.emplace_back(src, popularities[src], num_stations);
        station_name_to_index[station_names[src]] = src;
        
        // adjacency matrix of the stations to create platforms & links
        // different lines share the same platform for same destination
        for (int dst = 0; dst < num_stations; ++dst) {
            if (mat[src][dst] > 0) {
                num_links++;
                stations[src].platforms.emplace_back(popularities[src], dst, mat[src][dst]);
            }
        }
        MAX_NUMBER_OF_LINKS = std::max(MAX_NUMBER_OF_LINKS, num_links);
    }

    char line_keys[3] = {'g', 'y', 'b'};

    // start and end stations for each line
    vector<int> start_stations(3);
    vector<int> end_stations(3);
    for (i = 0; i < 3; ++i) {
        start_stations[i] = station_name_to_index[station_lines.at(line_keys[i]).front()];
        end_stations[i] = station_name_to_index[station_lines.at(line_keys[i]).back()];
    }

    // for each line, map station index to next station on that line
    vector<unordered_map<int, int>> forward_routes(3);
    vector<unordered_map<int, int>> backward_routes(3);
    
    for (i = 0; i < 3; ++i) {
        const auto& route = station_lines.at(line_keys[i]);
        int n = route.size();

        for (j = 0; j < n-1; ++j) {
            forward_routes[i][station_name_to_index[route[j]]] = station_name_to_index[route[j+1]];
            backward_routes[i][station_name_to_index[route[j+1]]] = station_name_to_index[route[j]];
        }
    }

    // create an array that maps train id to its line
    vector<int> train_id_to_line(TRAIN_SPAWN_NUM);
    int train_id = 0;
    int line_0_count = num_trains.at('g');
    int line_1_count = num_trains.at('y');
    int line_2_count = num_trains.at('b');

    while (num_trains.at('g') > 0 || num_trains.at('y') > 0 || num_trains.at('b') > 0) {
        for (int line = 0; line < 3; ++line) {
            if (num_trains.at(line_keys[line]) > 0) {
                train_id_to_line[train_id] = line;
                ++train_id;
                --num_trains[line_keys[line]];
            }
            if (num_trains.at(line_keys[line]) > 0) {
                train_id_to_line[train_id] = line;
                ++train_id;
                --num_trains[line_keys[line]];
            }
        }
    }
    num_trains['g'] = line_0_count;
    num_trains['y'] = line_1_count;
    num_trains['b'] = line_2_count;

    int original_num_stations = num_stations;

    // equation for new number of stations per process instead of ++1 
    num_stations = ((num_stations + total_processes - 1) / total_processes) * total_processes;

    int num_stations_per_process = num_stations / total_processes;
    vector<int> local_stations = distribute_stations(num_stations_per_process, original_num_stations, mpi_rank); // holds index of train stations that current local process will handle

    vector<int> station_to_process_map(original_num_stations); // stores which process has which station index
    for (i = 0; i < original_num_stations; ++i) {
        station_to_process_map[i] = i / num_stations_per_process;
    }

    int BUFFER_SIZE_PER_STATION = (MAX_NUMBER_OF_LINKS * 2 + 3); // 2 * max number of links for each station + 2 for spawning + 1 for if train arrives -> holding area -> platform
    int BUFFER_SIZE_PER_PROCESS = BUFFER_SIZE_PER_STATION * num_stations_per_process;
    int TOTAL_BUFFER_SIZE = BUFFER_SIZE_PER_STATION * num_stations;

    // for scatter: root process will send outgoing trains to other processes
    TrainComm* rootToProcess_outgoing_trains = new TrainComm[TOTAL_BUFFER_SIZE]; // holds total number of station * size of buffer for one station
    TrainComm* rootToProcess_recv_buffer = new TrainComm[BUFFER_SIZE_PER_PROCESS]; // holds number of station for each local process * size of buffer for one station

    int root = 0;
    int train_index = 0;

    // actual simulation
    for (int tick = 0; tick < ticks; ++tick) {
        // for gather: root process will receive incoming trains from other processes
        TrainComm* processToRoot_incoming_trains = new TrainComm[TOTAL_BUFFER_SIZE];
        TrainComm* processToRoot_outgoing_trains = new TrainComm[BUFFER_SIZE_PER_PROCESS];

        int processToRoot_outgoing_trains_nextFreeIndex = 0;

        // spawn trains if possible (rank 0)
        if (mpi_rank == 0) {
            for (int line = 0; line < 3; ++line) {
                // spawn at start station
                if (num_trains.at(line_keys[line]) > 0) {
                    int start_station = start_stations[line];
                    int next_station = start_stations[line];
                    int direction = 1;
                    int arrival_time = tick;
                    Train new_train(train_index, line, arrival_time, start_station, direction, next_station);
                    int process_handling_next_station = station_to_process_map[next_station];
                    insert_train_to_buffer(rootToProcess_outgoing_trains, process_handling_next_station, start_station, BUFFER_SIZE_PER_PROCESS, BUFFER_SIZE_PER_STATION, new_train);
                    all_trains[train_index] = new_train;
                    ++train_index;
                    --num_trains[line_keys[line]];
                }  
                
                // spawn at end station
                if (num_trains.at(line_keys[line]) > 0) {
                    int start_station = end_stations[line];
                    int next_station = end_stations[line];
                    int direction = 1; // we set this direction to forward first as it will be handled on train arrival function
                    int arrival_time = tick;
                    Train new_train(train_index, line, arrival_time, start_station, direction, next_station);
                    int process_handling_next_station = station_to_process_map[next_station];
                    insert_train_to_buffer(rootToProcess_outgoing_trains, process_handling_next_station, start_station, BUFFER_SIZE_PER_PROCESS, BUFFER_SIZE_PER_STATION, new_train);
                    all_trains[train_index] = new_train;
                    ++train_index;
                    --num_trains[line_keys[line]];
                }
            }
        }

        // scatter the outgoing trains to the correct processes
        MPI_Scatter(rootToProcess_outgoing_trains, BUFFER_SIZE_PER_PROCESS, mpi_train_type, rootToProcess_recv_buffer, BUFFER_SIZE_PER_PROCESS, mpi_train_type, root, MPI_COMM_WORLD);

        // process each station locally
        for (i = 0; i < local_stations.size(); ++i) {
            int station_idx = local_stations[i];
            Station& station = stations[station_idx];

            // handle occupancy status of links
            station.handle_platform_link_status(station, tick);

            // handle incoming trains received from scatter by putting them into station's incoming trains
            int incoming_count = 0;
            for (int j = 0; j < BUFFER_SIZE_PER_STATION; ++j) {
                int buffer_index = i*BUFFER_SIZE_PER_STATION + j;
                if (rootToProcess_recv_buffer[buffer_index].train_id != -1) {
                    Train full_train;
                    rootToProcess_recv_buffer[buffer_index].updateTrain(full_train);
                    station.incoming_trains.emplace(full_train);
                    incoming_count++;
                }
            }
            
            // process arrivals
            int arrival_count = 0;
            // if train arrives at station, move from incoming trains -> holding area
            while (!station.incoming_trains.empty() && station.incoming_trains.top().station_arrival_time <= tick) {
                Train train = station.incoming_trains.top();
                station.incoming_trains.pop();
                station.handle_train_arrival(train, forward_routes, backward_routes, start_stations, end_stations, 
                                                processToRoot_outgoing_trains, processToRoot_outgoing_trains_nextFreeIndex, train_id_to_line);
                arrival_count++;
            }
            station.process_platforms(processToRoot_outgoing_trains, processToRoot_outgoing_trains_nextFreeIndex, tick);
        }

        // gather the incoming trains from all processes
        MPI_Gather(processToRoot_outgoing_trains, BUFFER_SIZE_PER_PROCESS, mpi_train_type, processToRoot_incoming_trains, BUFFER_SIZE_PER_PROCESS, mpi_train_type, root, MPI_COMM_WORLD);

        rootToProcess_outgoing_trains = new TrainComm[TOTAL_BUFFER_SIZE];

        if (mpi_rank == 0) {
            // update all_trains with incoming trains from other processes
            for (i = 0; i < TOTAL_BUFFER_SIZE; ++i) {
                if (processToRoot_incoming_trains[i].train_id != -1) {
                    processToRoot_incoming_trains[i].updateTrain(all_trains[processToRoot_incoming_trains[i].train_id]);
                    // if new train status is at link, we need to update the station at the next tick
                    if (processToRoot_incoming_trains[i].state == 2) {
                        int next_station = processToRoot_incoming_trains[i].next_station;
                        int process_handling_next_station = station_to_process_map[next_station];
                        insert_train_to_buffer(rootToProcess_outgoing_trains, process_handling_next_station, next_station, BUFFER_SIZE_PER_PROCESS, BUFFER_SIZE_PER_STATION, all_trains[processToRoot_incoming_trains[i].train_id]);
                    }
                }
            }
        }

        // reset the rootToProcess_recv_buffer
        rootToProcess_recv_buffer = new TrainComm[BUFFER_SIZE_PER_PROCESS];

        // sort and print last num_ticks_to_print ticks
        if (mpi_rank == 0) {
            if (tick >= ticks - num_ticks_to_print) {
                print_tick_output(tick, all_trains, station_names);
            }
        }
    }
}