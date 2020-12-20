#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void vector_add(int *a, int *b, int *c, int n) {
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thread_id < n) {
        c[thread_id] = a[thread_id] + b[thread_id];
    }
}


void init_vector(int *a, int *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}


void check_answer(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; ++i) {
        assert(c[i] == a[i] + b[i]);
    }
}


int main() {
    // Initial values
    int id = cudaGetDevice(&id);     // Get the device ID for other CUDA calls
    int n = 1 << 16;                 // Number of elements per array
    size_t bytes = sizeof(int) * n;  // Size of each arrays in bytes
    int *a, *b, *c;                  // Unified memory pointers

    // Allocate host memory
    a = (int *)malloc(bytes);
    b = (int *)malloc(bytes);
    c = (int *)malloc(bytes);

    // Allocate memory for these pointers
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize vectors
    init_vector(a, b, n);

    // Set up threads
    int BLOCK_SIZE = 256;                       // Set threadblock size
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);  // Set grid size

    // Call CUDA kernel
    // Uncomment these for pre-fetching 'a' and 'b' vectors to device
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    vector_add<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, n);

    // Wait for all previous operations before using values
    cudaDeviceSynchronize();

    // Uncoment this for pre-fetching 'c' to the host
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // Check result
    check_answer(a, b, c, n);
}