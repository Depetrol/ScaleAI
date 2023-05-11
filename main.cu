#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#define NUM_THREADS 32

// =================
// Helper Functions
// =================

struct div_op : public thrust::unary_function<int, int> {
    int layer_size;

    div_op(int layer_size) : layer_size(layer_size) {}

    __host__ __device__ int operator()(int x) const {
        return x / layer_size;
    }
};

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

__global__ void weighing_gpu(double* input, double* weights, double* midresult, int layer_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= layer_size * layer_size)
        return;
    int j = tid % layer_size;
    midresult[tid] = weights[tid] * input[j];
}

__global__ void activation_gpu(double* biases, double* output, int layer_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= layer_size)
        return;
    output[tid] = 1 / (1 + exp(-(output[tid] + biases[tid])));
}

void forward_propagation(double* input, double* weights, double* biases, double* output, int layer_size, double* midresult) {
    // input, weights, biases, output, midresult live in gpu memory

    weighing_gpu<<<(layer_size * layer_size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(input, weights, midresult, layer_size);

    // TODO: thrust for_each and thrust_reduce
    thrust::reduce_by_key(
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), div_op(layer_size)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(layer_size * layer_size), div_op(layer_size)),
        thrust::device_pointer_cast(midresult),
        thrust::discard_iterator<>(),
        thrust::device_pointer_cast(output),
        thrust::equal_to<int>(),
        thrust::plus<double>()
    );

    activation_gpu<<<(layer_size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(biases, output, layer_size);
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
    double* midresult;

    cudaMalloc((void**)&midresult, layer_size * layer_size * sizeof(double));
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
        forward_propagation(output_gpu[step], weights_gpu[step], biases_gpu[step], output_gpu[step + 1], layer_size, midresult);
        cudaDeviceSynchronize();
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << seconds << "\n";
    
    cudaFree(midresult);
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
