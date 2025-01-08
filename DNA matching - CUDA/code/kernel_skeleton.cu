#include "kseq/kseq.h"
#include "common.h"
// #include <fstream>
// #include <chrono>


// Utility function to log timing results to a file
// void logTiming(const std::string& message, float time) {
//     std::ofstream logFile("timing_log.txt", std::ios_base::app);
//     logFile << message << ": " << time << " ms" << std::endl;
//     logFile.close();
// }

__global__ void matchKernel(
    const char* d_signature_seq,
    const int* d_signature_seq_starting_pos,
    const int* d_signature_seq_sizes,
    const char* d_sample_seq,
    const char* d_sample_qual,
    const int* d_sample_seq_starting_pos,
    const int* d_sample_seq_sizes,
    double* d_match_score,
    const int num_samples,
    const int num_signatures,
    const int total_pairs) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs) return;

    // each thread takes one sample and one signature, scan whole sample for that signature
    int sample_idx = idx / num_signatures;
    int signature_idx = idx % num_signatures;

    int sample_start = d_sample_seq_starting_pos[sample_idx];
    int sample_length = d_sample_seq_sizes[sample_idx];
    const char* sample_seq = &d_sample_seq[sample_start];
    const char* sample_qual = &d_sample_qual[sample_start]; // same length for sample and qual string

    int signature_start = d_signature_seq_starting_pos[signature_idx];
    int signature_length = d_signature_seq_sizes[signature_idx];
    const char* signature_seq = &d_signature_seq[signature_start];

    double match_score = 0;
    int earliest_match = -1;

    // matching
    for (int i = 0; i <= sample_length - signature_length; ++i) {
        bool match = true;

        for (int j = 0; j < signature_length; ++j) {
            char sample_char = sample_seq[i + j];
            char signature_char = signature_seq[j];

            if (sample_char != 'N' && signature_char != 'N' && sample_char != signature_char) {
                match = false;
                break;
            }
        }
        
        if (match) {
            earliest_match = i;
            break;
        }
    }

    if (earliest_match != -1) {
        for (int i = 0; i < signature_length; ++i) {
            match_score += static_cast<int>(sample_qual[earliest_match + i]) - 33;
        }
        match_score = match_score / signature_length;
    }

    int output_idx = sample_idx * num_signatures + signature_idx;
    d_match_score[output_idx] = match_score;
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
    // ### for profiling
    // cudaEvent_t start, stop;
    // float elapsedTime;

    // Create CUDA events for timing
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    

    // cudaError_t rc;

    //### Data Preparation
    int num_signatures = signatures.size();
    int num_samples = samples.size();

    // flatten signature sequence, have a vector containing start indexes and sizes for each string
    std::string flat_signature_seq;
    std::vector<int> signature_seq_starting_pos;
    std::vector<int> signature_seq_sizes;
    int sig_starting_pos = 0;

    int total_signature_size = 0;
    for (const auto& signature : signatures) {
        total_signature_size += signature.seq.size();
    }
    flat_signature_seq.reserve(total_signature_size);


    for (const auto& signature : signatures) {
        signature_seq_starting_pos.push_back(sig_starting_pos);
        flat_signature_seq += signature.seq;
        signature_seq_sizes.push_back(signature.seq.size());
        sig_starting_pos += signature.seq.size();
    }

    // flatten sample sequence, have vector containing start indexes and sizes for each string
    std::string flat_sample_seq;
    std::string flat_sample_qual;
    std::vector<int> sample_seq_starting_pos;
    std::vector<int> sample_seq_sizes;
    int sample_starting_pos = 0;

    int total_sample_size = 0;
    for (const auto& sample : samples) {
        total_sample_size += sample.seq.size();
    }
    flat_sample_seq.reserve(total_sample_size);
    flat_sample_qual.reserve(total_sample_size);

    for (const auto& sample : samples) {
        sample_seq_starting_pos.push_back(sample_starting_pos);
        flat_sample_seq += sample.seq;
        flat_sample_qual += sample.qual;
        sample_seq_sizes.push_back(sample.seq.size());
        sample_starting_pos += sample.seq.size();
    }

	int total_pairs = num_samples * num_signatures;
	int total_signature_seq_length = flat_signature_seq.size();
	int total_sample_seq_length = flat_sample_seq.size();

    // cudaEventRecord(start); // profiling

	//### memory allocation and data transfer to device

	// for signature flattened sequence
	char* d_signature_seq;
	cudaMalloc(&d_signature_seq, total_signature_seq_length * sizeof(char));
	cudaMemcpy(d_signature_seq, flat_signature_seq.data(), total_signature_seq_length * sizeof(char), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy flat_signature_seq to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

	// for signature starting positions
	int* d_signature_seq_starting_pos;
	cudaMalloc(&d_signature_seq_starting_pos, num_signatures * sizeof(int));
	cudaMemcpy(d_signature_seq_starting_pos, signature_seq_starting_pos.data(), num_signatures * sizeof(int), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy signature_seq_starting_pos to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

	// for signature indivudal sizes
	int* d_signature_seq_sizes;
	cudaMalloc(&d_signature_seq_sizes, num_signatures * sizeof(int));
	cudaMemcpy(d_signature_seq_sizes, signature_seq_sizes.data(), num_signatures * sizeof(int), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy signature_seq_sizes to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

	// for sample flattened sequence
	char* d_sample_seq;
    cudaMalloc(&d_sample_seq, total_sample_seq_length * sizeof(char));
    cudaMemcpy(d_sample_seq, flat_sample_seq.data(), total_sample_seq_length * sizeof(char), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy sample_seq to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

    // for sample flattened quality string
	char* d_sample_qual;
    cudaMalloc(&d_sample_qual, total_sample_seq_length * sizeof(char));
    cudaMemcpy(d_sample_qual, flat_sample_qual.data(), total_sample_seq_length * sizeof(char), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy sample_qual to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

	// for sample starting positions
	int* d_sample_seq_starting_pos;
	cudaMalloc(&d_sample_seq_starting_pos, num_samples * sizeof(int));
	cudaMemcpy(d_sample_seq_starting_pos, sample_seq_starting_pos.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy sample_seq_starting_pos to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

    // for sample indivudal sizes
	int* d_sample_seq_sizes;
	cudaMalloc(&d_sample_seq_sizes, num_samples * sizeof(int));
	cudaMemcpy(d_sample_seq_sizes, sample_seq_sizes.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
	// if (rc != cudaSuccess) {
    //     printf("Could not copy sample_seq_sizes to device. Reason: %s\n", cudaGetErrorString(rc));
    // }

    // for storing match score on device
    double* d_match_score;
    cudaMalloc(&d_match_score, total_pairs * sizeof(double));


    // for profiling
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // logTiming("Data Transfer to Device", elapsedTime);

    // cudaEventRecord(start);

	//### Kernel Execution
    int THREADS_PER_BLOCK = 256; // try 256,512,1024
    // const dim3 threadsPerBlock = { THREADS_PER_BLOCK, 1, 1 }; 
    const int blocks_needed = (total_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // const dim3 numBlocks = { blocks_needed, 1, 1 };

    matchKernel<<<blocks_needed, THREADS_PER_BLOCK>>>(
        d_signature_seq,
        d_signature_seq_starting_pos,
        d_signature_seq_sizes,
        d_sample_seq,
        d_sample_qual,
        d_sample_seq_starting_pos,
        d_sample_seq_sizes,
        d_match_score,
        num_samples,
        num_signatures,
        total_pairs);

    cudaDeviceSynchronize();

    // for profiling
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // logTiming("Kernel Execution", elapsedTime);

    // cudaEventRecord(start);

    // ### Result Collection

    std::vector<double> h_match_score(total_pairs);
    cudaMemcpy(h_match_score.data(), d_match_score, total_pairs * sizeof(double), cudaMemcpyDeviceToHost);

    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        for (int signature_idx = 0; signature_idx < num_signatures; ++signature_idx) {
            int idx = sample_idx * num_signatures + signature_idx;
            if (h_match_score[idx] != 0) {
                MatchResult result = {
                    samples[sample_idx].name,
                    signatures[signature_idx].name,
                    h_match_score[idx]
                };
                matches.push_back(result);
            }
        }
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // logTiming("Result Collection", elapsedTime);

    // ### Cleanup
    cudaFree(d_signature_seq);
    cudaFree(d_signature_seq_starting_pos);
    cudaFree(d_signature_seq_sizes);
    cudaFree(d_sample_seq);
    cudaFree(d_sample_qual);
    cudaFree(d_sample_seq_starting_pos);
    cudaFree(d_sample_seq_sizes);
    cudaFree(d_match_score);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
}
