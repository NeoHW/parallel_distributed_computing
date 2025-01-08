
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "kseq/kseq.h"

void gen_no_virus_sample(std::chrono::microseconds& value, int& sample_counter,
                         std::uniform_int_distribution<>& length_dist, std::mt19937& rng, double n_ratio, int min_phred,
                         int max_phred);

void gen_virus_sample(std::chrono::microseconds& value, int& sample_counter,
                      std::uniform_int_distribution<>& num_viruses_dist, std::mt19937& rng,
                      std::uniform_int_distribution<>& virus_dist, std::vector<klibpp::KSeq>& viruses, int min_length,
                      int max_length, double n_ratio, int min_phred, int max_phred);

/**
 * Credits: initial boilerplate from ChatGPT o1-mini
 */

/**
 * @brief Generates a random DNA sequence of given length.
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

/**
 * @brief Generates a Phred quality string using a two-step distribution:
 *        1. Draw a uniform value N between min_phred and max_phred.
 *        2. For each Phred score, draw from a normal distribution with mean N and stddev 3.
 *        3. Clamp the scores to the [min_phred, max_phred] range.
 *
 * @param length     Length of the quality string to generate.
 * @param min_phred  Minimum Phred score (inclusive).
 * @param max_phred  Maximum Phred score (inclusive).
 * @param rng        Reference to a random number generator (e.g., std::mt19937).
 * @return std::string Generated quality string.
 *
 * @throws std::invalid_argument if min_phred is greater than max_phred.
 */
std::string generateQualityString(size_t length, int min_phred, int max_phred, std::mt19937& rng) {
    // Validate input parameters
    if (min_phred > max_phred) {
        throw std::invalid_argument("min_phred cannot be greater than max_phred.");
    }

    // Step 1: Draw N uniformly between min_phred and max_phred
    std::uniform_int_distribution<> uniform_dist(min_phred, max_phred);
    int N = uniform_dist(rng);

    // Step 2: Set up a normal distribution with mean N and standard deviation of 3
    std::normal_distribution<> normal_dist(static_cast<double>(N), 3.0);

    std::string qual;
    qual.reserve(length);  // Optimize memory allocation

    for (size_t i = 0; i < length; ++i) {
        // Draw a Phred score from the normal distribution
        double phred_double = normal_dist(rng);
        int phred = static_cast<int>(std::round(phred_double));

        // Step 3: Clamp the Phred score to [min_phred, max_phred]
        phred = std::max(min_phred, std::min(phred, max_phred));

        // Step 4: Convert the Phred score to a Sanger-encoded ASCII character
        char qual_char = static_cast<char>(phred + 33);
        qual += qual_char;
    }

    return qual;
}

/**
 * @brief Generates a random flanking sequence of given length, inserting 'N's based on ratio.
 *
 * @param length Length of the flanking sequence.
 * @param n_ratio Ratio of 'N's.
 * @param rng Random number generator.
 * @return std::string Flanking sequence with 'N's.
 */
std::string generateFlank(size_t length, double n_ratio, std::mt19937& rng) {
    std::string flank = generateRandomDna(length, rng);
    if (n_ratio > 0.0) {
        flank = insertNs(flank, n_ratio, rng);
    }
    return flank;
}

// Function to generate a FASTQ record and print it
auto generateAndPrintFastq = [](const std::string& sample_name, const std::string& sequence,
                                 const std::string& quality) {
    std::cout << "@" << sample_name << "\n";
    std::cout << sequence << "\n";
    std::cout << "+" << "\n";
    std::cout << quality << "\n";
};

int main(int argc, char* argv[]) {
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <fasta_file> <num_no_virus> <num_with_virus> <min_viruses> <max_viruses> <min_length> "
                     "<max_length> <min_phred> <max_phred> <n_ratio>\n";
        return 1;
    }

    // Parse command-line arguments
    const char* fasta_path = argv[1];
    int num_no_virus = std::stoi(argv[2]);
    int num_with_virus = std::stoi(argv[3]);
    int min_viruses = std::stoi(argv[4]);
    int max_viruses = std::stoi(argv[5]);
    int min_length = std::stoi(argv[6]);
    int max_length = std::stoi(argv[7]);
    int min_phred = std::stoi(argv[8]);
    int max_phred = std::stoi(argv[9]);
    double n_ratio = std::stod(argv[10]);

    // Validate arguments
    if (min_length <= 0 || max_length < min_length) {
        std::cerr << "Error: Invalid sample length parameters.\n";
        return 1;
    }
    if (min_phred < 0 || max_phred > 93 || min_phred > max_phred) {
        std::cerr << "Error: Invalid Phred score parameters.\n";
        return 1;
    }
    if (n_ratio < 0.0 || n_ratio > 1.0) {
        std::cerr << "Error: 'n_ratio' must be between 0.0 and 1.0.\n";
        return 1;
    }
    if (min_viruses <= 0 || max_viruses < min_viruses) {
        std::cerr << "Error: Invalid virus count parameters.\n";
        return 1;
    }

    // Print all the settings to stderr
    std::cerr << "Settings:\n";
    std::cerr << "  FASTA file: " << fasta_path << "\n";
    std::cerr << "  Number of samples without virus: " << num_no_virus << "\n";
    std::cerr << "  Number of samples with virus: " << num_with_virus << "\n";
    std::cerr << "  Number of viruses: minimum " << min_viruses << " to maximum " << max_viruses << "\n";
    std::cerr << "  Requested sample length: minimum " << min_length << " to maximum " << max_length << "\n";
    std::cerr << "  Average Phred score: " << min_phred << " to " << max_phred << "\n";
    std::cerr << "  N ratio: " << n_ratio << "\n";


    // Read FASTA records (signatures)
    int sample_fd = open(fasta_path, O_RDONLY);
    if (sample_fd == -1) {
        std::cerr << "Error: Unable to open FASTA file.\n";
        return 1;
    }
    auto ks_sig = klibpp::make_kstream(sample_fd, read, klibpp::mode::in, close);
    std::vector<klibpp::KSeq> viruses = ks_sig.read();

    // Make the min length no shorter than the longest virus in the file
    // Make the max length no shorter than the longest virus in the file
    for (const auto& virus : viruses) {
        if (virus.seq.length() > static_cast<size_t>(min_length)) {
            min_length = virus.seq.length();
        }
        if (virus.seq.length() > static_cast<size_t>(max_length)) {
            max_length = virus.seq.length();
        }
    }

    // Update the final min and max length after reading the FASTA file
    std::cerr << "  Final sample length values: minimum " << min_length << " to maximum " << max_length
              << " (after reading virus lengths)\n";

    // Initialize random number generators
    std::random_device rd;
    std::mt19937 rng(rd());

    // Distributions for selecting viruses and generating random numbers
    std::uniform_int_distribution<> virus_dist(0, viruses.size() - 1);
    std::uniform_int_distribution<> num_viruses_dist(min_viruses, max_viruses);
    std::uniform_int_distribution<> length_dist(min_length, max_length);

    // Sample name should be sample_{integer value for time in microseconds}_counter
    auto now = std::chrono::system_clock::now();
    auto now_micro = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto value = now_micro.time_since_epoch();
    // Sample counter for unique naming
    int sample_counter = 1;

    // Generate FASTQ samples without viruses

    // Randomly choose between virus and no virus, but make sure we respect the number of samples requested for each
    // category. Make it simple.
    std::uniform_int_distribution<> virus_or_no_virus(0, 1);
    int total_samples = num_no_virus + num_with_virus;
    for (int i = 0; i < total_samples; ++i) {
        if ((virus_or_no_virus(rng) == 0 && num_no_virus > 0) || num_with_virus == 0) {
            gen_no_virus_sample(value, sample_counter, length_dist, rng, n_ratio, min_phred, max_phred);
            num_no_virus--;
        } else {
            gen_virus_sample(value, sample_counter, num_viruses_dist, rng, virus_dist, viruses, min_length, max_length,
                            n_ratio, min_phred, max_phred);
            num_with_virus--;
        }
    }

    return 0;
}

void gen_virus_sample(std::chrono::microseconds& value, int& sample_counter,
                      std::uniform_int_distribution<>& num_viruses_dist, std::mt19937& rng,
                      std::uniform_int_distribution<>& virus_dist, std::vector<klibpp::KSeq>& viruses, int min_length,
                      int max_length, double n_ratio, int min_phred, int max_phred) {
    std::string sample_name = "sample_" + std::to_string(value.count()) + "_" + std::to_string(sample_counter++);
    int num_viruses = num_viruses_dist(rng);

    // Select viruses to insert
    std::vector<std::string> selected_viruses;
    size_t total_virus_length = 0;
    for (int v = 0; v < num_viruses; ++v) {
        int idx = virus_dist(rng);
        selected_viruses.push_back(viruses[idx].seq);
        total_virus_length += viruses[idx].seq.length();
    }

    // Determine flanking lengths
    int remaining_length_min = min_length - static_cast<int>(total_virus_length);
    int remaining_length_max = max_length - static_cast<int>(total_virus_length);

    if (remaining_length_min < 0) remaining_length_min = 0;
    if (remaining_length_max < 0) remaining_length_max = 0;

    if (remaining_length_min > remaining_length_max) {
        remaining_length_min = remaining_length_max;  // Adjust to feasible range
    }

    // Decide on total sample length
    int sample_length = 0;
    if (remaining_length_max > 0) {
        std::uniform_int_distribution<> flank_dist(remaining_length_min, remaining_length_max);
        sample_length = total_virus_length + flank_dist(rng);
    } else {
        sample_length = total_virus_length;
    }

    // Distribute flanking regions around viruses
    // Number of flanks = num_viruses + 1
    int num_flanks = num_viruses + 1;
    int total_flank_length = sample_length - static_cast<int>(total_virus_length);

    std::vector<int> flank_lengths(num_flanks, 0);
    if (num_flanks > 0 && total_flank_length > 0) {
        std::uniform_int_distribution<> split_dist(0, total_flank_length);
        for (int f = 0; f < num_flanks - 1; ++f) {
            int flank = split_dist(rng);
            flank_lengths[f] = flank;
            total_flank_length -= flank;
        }
        flank_lengths[num_flanks - 1] = total_flank_length;
    }

    // Build the sample sequence
    std::string sequence;
    sequence.reserve(sample_length);
    for (int f = 0; f < num_flanks; ++f) {
        if (flank_lengths[f] > 0) {
            std::string flank = generateFlank(flank_lengths[f], n_ratio, rng);
            sequence += flank;
        }
        if (f < num_viruses) {
            sequence += selected_viruses[f];
        }
    }

    // If the generated sequence is shorter than sample_length, pad with random nucleotides
    if (sequence.length() < static_cast<size_t>(sample_length)) {
        size_t pad_length = sample_length - sequence.length();
        std::string padding = generateFlank(pad_length, n_ratio, rng);
        sequence += padding;
    }

    // If the generated sequence is longer than sample_length, truncate it
    if (sequence.length() > static_cast<size_t>(sample_length)) {
        sequence = sequence.substr(0, sample_length);
    }

    // Assign quality scores
    std::string quality = generateQualityString(sequence.length(), min_phred, max_phred, rng);

    // Ensure that all sequence chars with value 'N' have a quality score of 0
    for (size_t i = 0; i < sequence.length(); ++i) {
        if (sequence[i] == 'N') {
            quality[i] = '!';
        }
    }

    generateAndPrintFastq(sample_name, sequence, quality);
}

void gen_no_virus_sample(std::chrono::microseconds& value, int& sample_counter,
                         std::uniform_int_distribution<>& length_dist, std::mt19937& rng, double n_ratio, int min_phred,
                         int max_phred) {
    std::string sample_name = "sample_" + std::to_string(value.count()) + "_" + std::to_string(sample_counter++);
    int sample_length = length_dist(rng);

    std::string sequence = generateRandomDna(sample_length, rng);
    if (n_ratio > 0.0) {
        sequence = insertNs(sequence, n_ratio, rng);
    }

    std::string quality = generateQualityString(sequence.length(), min_phred, max_phred, rng);
    // Ensure that all sequence chars with value 'N' have a quality score of 0
    for (size_t i = 0; i < sequence.length(); ++i) {
        if (sequence[i] == 'N') {
            quality[i] = '!';
        }
    }

    generateAndPrintFastq(sample_name, sequence, quality);
}
