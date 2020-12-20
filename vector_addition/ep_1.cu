#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// CUDA kernel for vector addition
__global__ void vector_add(int *a, int *b, int *c, int n) {
    // Calculate global thread ID
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Vector boundary guard
    if (thread_id < n) {
        // Each thread adds a single element
        c[thread_id] = a[thread_id] + b[thread_id];
    }
}


// Initialize vector of size n to int between 0-99
void matrix_init(int *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % 100;
    }
}


// Check vector add result
void error_check(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; ++i) {
        assert(c[i] == a[i] + b[i]);
    }
}


int main() {
    // Initial values
    int n = 1 << 16;                 // Vector size of 2^16 (65536 elements)
    int *h_a, *h_b, *h_c;            // Host vector pointers
    int *d_a, *d_b, *d_c;            // Device vector pointers
    size_t bytes = sizeof(int) * n;  // Allocation size for all vectors

    // Allocate host memory
    h_a = (int *)malloc(bytes);
    h_b = (int *)malloc(bytes);
    h_c = (int *)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Setting up GPU threads
    int NUM_THREADS = 256;  // Threadblock size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // Launch kernel on default stream w/o shmem
    vector_add<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    error_check(h_a, h_b, h_c, n);
}