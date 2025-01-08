#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#include "kseq/kseq.h"
#include "common.h"

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches);

int main(int argc, char* argv[]) {
    // Check if the user provided a FASTA file as an argument
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <sample.fastq> <signatures.fasta>" << std::endl;
        return 1;
    }

    const char* sample_file = argv[1];
    const char* sig_file = argv[2];

    int sample_fd = open(sample_file, O_RDONLY);
    // Error handling
    if (sample_fd == -1) {
        std::cerr << "Error: Unable to open file " << sample_file << std::endl;
        return 1;
    }

    int sig_fd = open(sig_file, O_RDONLY);
    // Error handling
    if (sig_fd == -1) {
        std::cerr << "Error: Unable to open file " << sig_file << std::endl;
        return 1;
    }

    // Read FASTQ records (samples)
	std::cerr << "Starting to read sample FASTQ file" << std::endl;
    auto ks_sample = klibpp::make_kstream(sample_fd, read, klibpp::mode::in, close);
    // Error handling - check for .fail()
    if (ks_sample.fail()) {
        std::cerr << "Error: Unable to read FASTQ records from " << sample_file << std::endl;
        return 1;
    }


    std::vector<klibpp::KSeq> samples = ks_sample.read();
    // If we read nothing, throw an error
    if (samples.size() == 0) {
        std::cerr << "Error: No valid FASTQ records found in " << sample_file << std::endl;
        return 1;
    }

    // Read FASTA records (signatures)
	std::cerr << "Starting to read signature FASTA file" << std::endl;
    auto ks_sig = klibpp::make_kstream(sig_fd, read, klibpp::mode::in, close);
    // Error handling - check for .fail()
    if (ks_sig.fail()) {
        std::cerr << "Error: Unable to read FASTA records from " << sample_file << std::endl;
        return 1;
    }
    std::vector<klibpp::KSeq> sigs = ks_sig.read();
    // If we read nothing, throw an error
    if (sigs.size() == 0) {
        std::cerr << "Error: No valid FASTA records found in " << sig_file << std::endl;
        return 1;
    }

    std::cerr << "Read " << samples.size() << " samples and " << sigs.size() << " signatures." << std::endl;

    // Initialize a vector to store matches
    std::vector<MatchResult> matches;

    // start wall
	std::cerr << "Starting timing and matching..." << std::endl;
    auto start_wall = std::chrono::high_resolution_clock::now();

    runMatcher(samples, sigs, matches);

    // end wall
	std::cerr << "runMatcher completed." << std::endl;
    auto end_wall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wall_elapsed_seconds = end_wall - start_wall;

    for (const auto& match : matches) {
        std::cout << match.sample_name << ": " << match.signature_name << " (" 
                  << std::fixed << std::setprecision(3) << match.match_score << ")\n";
    }

    std::cerr << "(FOR AUTOMATED CHECKING) Total runMatcher time:" << wall_elapsed_seconds.count() << "s" << std::endl;

    return 0;
}
