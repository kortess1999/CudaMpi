
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

extern "C" void addWithCuda(int* data, const int arraySize, double* output, const int outputSize, const int blockSize);

__global__ void sumKernel(int* data, const int step, const int blockSize, const int dataSize)
{
    int i = threadIdx.x;
    int blockId = blockIdx.x;

    const int index_first = (2 << step) * i;
    const int index_second = index_first + (1 << step);

    if (index_second < blockSize && blockSize * blockId + index_second < dataSize) {
        data[blockSize * blockId + index_first] = data[blockSize * blockId + index_first] + data[blockSize * blockId + index_second];
    }
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" void addWithCuda(int* data, const int arraySize, double* output, const int outputSize, const int blockSize)
{
    int* d_data = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_data, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_data, data, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    for (int k = 0; k < ceil(log2((double)blockSize)); k++) {
        const int threadCount = ceil(blockSize / pow(2, k));
        sumKernel << <outputSize, threadCount >> > (d_data, k, blockSize, arraySize);

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(data, d_data, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(data, d_data, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    for (int i = 0; i < outputSize; i++) {
        if (i != outputSize - 1) {
            output[i] = (double)data[blockSize * i] / (double)blockSize;
        }
        else {
            output[i] = (double)data[blockSize * i] / (double)(arraySize - i * blockSize - 1);
        }
    }

Error:
    cudaFree(d_data);
}
