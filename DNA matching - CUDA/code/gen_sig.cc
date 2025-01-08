#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

/**
 * @brief Generates a random DNA sequence of a given length.
 *
 * @param length Length of the sequence.
 * @param rng Random number generator.
 * @return std::string Random DNA sequence.
 */
std::string generateRandomDna(size_t length, std::mt19937& rng) {
    const char nucleotides[] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<> dist(0, 3);
    std::string seq;
    seq.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        seq += nucleotides[dist(rng)];
    }
    return seq;
}

/**
 * @brief Inserts 'N's into a DNA sequence based on the given ratio.
 *
 * @param sequence DNA sequence.
 * @param n_ratio Ratio of 'N's (0.0 to 1.0).
 * @param rng Random number generator.
 * @return std::string Sequence with 'N's inserted.
 */
std::string insertNs(const std::string& sequence, double n_ratio, std::mt19937& rng) {
    std::string modified_seq = sequence;
    std::bernoulli_distribution dist(n_ratio);
    for (size_t i = 0; i < modified_seq.size(); ++i) {
        if (dist(rng)) {
            modified_seq[i] = 'N';
        }
    }
    return modified_seq;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_signatures> <min_length> <max_length> <n_ratio>\n";
        return 1;
    }

    // Parse command-line arguments
    int num_signatures = std::stoi(argv[1]);
    int min_length = std::stoi(argv[2]);
    int max_length = std::stoi(argv[3]);
    double n_ratio = std::stod(argv[4]);

    // Validate arguments
    if (min_length <= 0 || max_length < min_length) {
        std::cerr << "Error: Invalid length parameters.\n";
        return 1;
    }
    if (n_ratio < 0.0 || n_ratio > 1.0) {
        std::cerr << "Error: 'n_ratio' must be between 0.0 and 1.0.\n";
        return 1;
    }

    // Initialize random number generators
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<> length_dist(min_length, max_length);

    // Get current time for unique sample naming
    auto now = std::chrono::system_clock::now();
    auto now_micro = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto value = now_micro.time_since_epoch();

    // Generate the specified number of signatures
    for (int i = 0; i < num_signatures; ++i) {
        int seq_length = length_dist(rng);
        std::string sequence = generateRandomDna(seq_length, rng);
        sequence = insertNs(sequence, n_ratio, rng);

        // Generate a unique name based on the time and the current signature index
        std::string sample_name = "signature_" + std::to_string(value.count()) + "_" + std::to_string(i + 1);

        // Output the FASTA formatted signature
        std::cout << ">" << sample_name << "\n";
        std::cout << sequence << "\n";
    }

    return 0;
}
