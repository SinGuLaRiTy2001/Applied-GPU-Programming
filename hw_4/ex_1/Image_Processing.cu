#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// CUDA kernel to double the values of each pixel
__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m) {
    // Calculate the row # of the d_Pin and d_Pout element to process
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_Pin and d_Pout element to process
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread computes one element of d_Pout if in range
    if ((Row < m) && (Col < n)) {
        d_Pout[Row * n + Col] = 2 * d_Pin[Row * n + Col];
    }
}

int main() {
    int n = 800;
    int m = 600;

    size_t size = n * m * sizeof(float);

    float* h_Pin = new float[n*m];
    float* h_Pout = new float[n*m];

    for (int i = 0; i < n * m; ++i) {
        h_Pin[i] = rand() / (float) (RAND_MAX + 1.0);
    }

    float* d_Pin;
    float* d_Pout;
    cudaMalloc((void**)&d_Pin, size);
    cudaMalloc((void**)&d_Pout, size);

    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    PictureKernel<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, n, m);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    delete[] h_Pin;
    delete[] h_Pout;
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    return 0;
}
