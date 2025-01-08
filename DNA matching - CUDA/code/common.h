#pragma once

struct MatchResult {
    std::string sample_name;
    std::string signature_name;
    double match_score;
};

void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& matches);


/*
struct KSeq {
    std::string name;     // Sequence identifier
    std::string comment; // Not used in this assignment
    std::string seq;    // The actual DNA sequence containing A/T/C/G/N
    std::string qual;  // (If it’s a sample) The Phred+33 encoded quality string
                      // e.g., where the character ‘5’ refers to a Phred score of 20.
};
*/