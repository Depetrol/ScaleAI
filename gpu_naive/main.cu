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
void init(double *input, double *weights, double *biases, int layer_size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    std::uniform_real_distribution<double> rand_real(-1000.0, 1000.0);

    for (int i = 0; i < layer_size; ++i) {
        biases[i] = rand_real(gen);
        input[i] = rand_real(gen);
        for (int j = 0; j < layer_size; ++j) {
            weights[i * layer_size + j] = rand_real(gen);
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
        std::cout << "-n <int>: set number of steps" << std::endl;
        std::cout << "-l <int>: set layer size" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Initialize Particles
    int nsteps = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    int layer_size = find_int_arg(argc, argv, "-l", 1024);

    double* input = new double[layer_size];
    double* output = new double[layer_size];
    double* weights = new double[layer_size * layer_size];
    double* biases = new double[layer_size];

    init(input, weights, biases, layer_size, part_seed);

    double* input_gpu;
    double* output_gpu;
    double* weights_gpu;
    double* biases_gpu;

    cudaMalloc((void**)&input_gpu, layer_size * sizeof(double));
    cudaMalloc((void**)&output_gpu, layer_size * sizeof(double));
    cudaMalloc((void**)&weights_gpu, layer_size * layer_size * sizeof(double));
    cudaMalloc((void**)&biases_gpu, layer_size * sizeof(double));

    cudaMemcpy(input_gpu, input, layer_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(output_gpu, output, layer_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_gpu, weights, layer_size * layer_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(biases_gpu, biases, layer_size * sizeof(double), cudaMemcpyHostToDevice);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    for (int step = 0; step < nsteps; ++step) {
        forward_propagation(input_gpu, weights_gpu, biases_gpu, output_gpu, layer_size);
        cudaDeviceSynchronize();
        forward_propagation(output_gpu, weights_gpu, biases_gpu, input_gpu, layer_size);
        cudaDeviceSynchronize();
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << seconds << "\n";
    cudaFree(input_gpu);
    cudaFree(output_gpu);
    cudaFree(weights_gpu);
    cudaFree(biases_gpu);
    delete[] input;
    delete[] output;
    delete[] weights;
    delete[] biases;
    return 0;
}
