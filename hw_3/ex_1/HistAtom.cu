#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

const int ThreadNum = 128;

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
//@@ Insert code below to compute histogram of input using shared memory and atomics
    __shared__ unsigned int hist_temp[NUM_BINS];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned int i = threadIdx.x; i < NUM_BINS; i+=blockDim.x ) {
        hist_temp[i] = 0;
    }
    __syncthreads();
    if (index < num_elements) {
        atomicAdd(&(hist_temp[input[index]]), 1);
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < NUM_BINS; i+=blockDim.x ) {
        atomicAdd(&(bins[i]), hist_temp[i]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
//@@ Insert code below to clean up bins that saturate at 127
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < NUM_BINS) {
        bins[index] = bins[index]>127 ? 127 : bins[index];
    }
}

double getTime() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);
    double RunTime = (double)time_now.tv_sec + (double)time_now.tv_nsec / 1.e9;
    return RunTime;
}

int main(int argc, char **argv) {

    long long int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %lld\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    srand(time(NULL));
    for (unsigned int i = 0; i < inputLength; i++) {
        hostInput[i] = rand() % NUM_BINS;
    }

    //@@ Insert code below to create reference result in CPU
    resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
//    for (unsigned int i = 0; i < inputLength; i++) {
//        resultRef[hostInput[i]] = resultRef[hostInput[i]]<127 ? resultRef[hostInput[i]]+1 : resultRef[hostInput[i]];
//    }
    memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));
    for (unsigned int i = 0; i < inputLength; i++) {
        int binIndex = hostInput[i] < NUM_BINS ? hostInput[i] : NUM_BINS-1;
        if (++resultRef[binIndex] > 127) {
            resultRef[binIndex] = 127;
        }
    }

    printf("Reference result:\n");
    for (unsigned int i = 0; i < NUM_BINS; i++) {
        if (resultRef[i] != 0)
            printf("%d ", resultRef[i]);
    }
    printf("\n");

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    double start_time = getTime();
    int block_num1 = (inputLength + ThreadNum - 1) / ThreadNum;

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<block_num1, ThreadNum>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    cudaDeviceSynchronize();

    //@@ Initialize the second grid and block dimensions here
    int block_num2 = (NUM_BINS + ThreadNum - 1) / ThreadNum;

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<block_num2, ThreadNum>>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

    double end_time = getTime();
    printf("Kernel runtime: %.6lf ms\n", (end_time - start_time) * 1000);
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("GPU result:\n");
    for (unsigned int i = 0; i < NUM_BINS; i++) {
        if (hostBins[i] != 0)
            printf("%d ", hostBins[i]);
    }
    printf("\n");

    //@@ Insert code below to compare the output with the reference
    int diff_cnt = 0;
    for (unsigned int i = 0; i < NUM_BINS; i++) {
        if (resultRef[i] != hostBins[i]) {
            printf("%d, %d\n", resultRef[i], hostBins[i]);
            diff_cnt ++;
        }
    }
    if (diff_cnt > 0) {
        printf("Mismatch detected: %d\n", diff_cnt);
    }
    else {
        printf("Calculation match!\n");
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}

