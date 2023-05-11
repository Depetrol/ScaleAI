#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 32

int blk;

// =================
// Helper Functions
// =================

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

// Particle Initialization
void init(double** output, double** weights, double** biases, int layer_size, int part_seed, int nsteps) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    std::uniform_real_distribution<double> rand_real(-1000.0, 1000.0);

    for (int i = 0; i < layer_size; ++i) {
        output[0][i] = rand_real(gen);
    }

    for (int step = 0; step < nsteps; ++step) {
        for (int i = 0; i < layer_size; ++i) {
            biases[step][i] = rand_real(gen);
            for (int j = 0; j < layer_size; ++j) {
                weights[step][i * layer_size + j] = rand_real(gen);
            }
        }
    }

    blk = (layer_size + NUM_THREADS - 1) / NUM_THREADS;
}

__global__ void calc_output_gpu(double* input, double* weights, double* biases, double* output, int layer_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= layer_size)
        return;
    output[tid] = biases[tid];
    for (int j = 0; j < layer_size; j++) {
        output[tid] += input[j] * weights[tid * layer_size + j];
    }
    output[tid] = 1 / (1 + exp(-output[tid]));
}

void forward_propagation(double* input, double* weights, double* biases, double* output, int layer_size) {
    // input, weights, biases, output, midresult live in gpu memory
    calc_output_gpu<<<blk, NUM_THREADS>>>(input, weights, biases, output, layer_size);
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of layers" << std::endl;
        std::cout << "-w <int>: set network width" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Initialize Particles
    int nsteps = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    int layer_size = find_int_arg(argc, argv, "-w", 1024);

    // double* input = new double[layer_size];
    double* output[nsteps + 1];
    double* weights[nsteps];
    double* biases[nsteps];

    for (int step = 0; step < nsteps; ++step) {
        output[step] = new double[layer_size];
        weights[step] = new double[layer_size * layer_size];
        biases[step] = new double[layer_size];
    }
    output[nsteps] = new double[layer_size];

    init(output, weights, biases, layer_size, part_seed, nsteps);

    // double* input_gpu;
    double* output_gpu[nsteps + 1];
    double* weights_gpu[nsteps];
    double* biases_gpu[nsteps];

    for (int step = 0; step < nsteps; ++step) {
        cudaMalloc((void**)&output_gpu[step], layer_size * sizeof(double));
        cudaMalloc((void**)&weights_gpu[step], layer_size * layer_size * sizeof(double));
        cudaMalloc((void**)&biases_gpu[step], layer_size * sizeof(double));
        cudaMemcpy(output_gpu[step], output[step], layer_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(weights_gpu[step], weights[step], layer_size * layer_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(biases_gpu[step], biases[step], layer_size * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&output_gpu[nsteps], layer_size * sizeof(double));
    cudaMemcpy(output_gpu[nsteps], output[nsteps], layer_size * sizeof(double), cudaMemcpyHostToDevice);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    for (int step = 0; step < nsteps; ++step) {
        forward_propagation(output_gpu[step], weights_gpu[step], biases_gpu[step], output_gpu[step + 1], layer_size);
        cudaDeviceSynchronize();
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << seconds << "\n";
    
    for (int step = 0; step < nsteps; ++step) {
        cudaFree(output_gpu[step]);
        cudaFree(weights_gpu[step]);
        cudaFree(biases_gpu[step]);
        delete[] output[step];
        delete[] weights[step];
        delete[] biases[step];
    }
    cudaFree(output_gpu[nsteps]);
    delete[] output[nsteps];
    
    return 0;
}
