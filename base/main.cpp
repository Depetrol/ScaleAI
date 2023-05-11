#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

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
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

void forward_propagation(const double *input, const double *weights, const double *biases, double *output, int layer_size) {
    for (int i = 0; i < layer_size; i++) {
        double weighted_sum = biases[i];
        for (int j = 0; j < layer_size; j++) {
            weighted_sum += input[j] * weights[i * layer_size + j];
        }
        output[i] = sigmoid(weighted_sum);
    }
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

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    for (int step = 0; step < nsteps; ++step) {
        forward_propagation(output[step], weights[step], biases[step], output[step + 1], layer_size);
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << seconds << "\n";
    for (int step = 0; step < nsteps; ++step) {
        delete[] output[step];
        delete[] weights[step];
        delete[] biases[step];
    }
    delete[] output[nsteps];
}
