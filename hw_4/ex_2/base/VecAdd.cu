#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

static int ThreadNum = 256;

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    //@@ Insert code to implement vector addition here
    const int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

//@@ Insert code to implement timer
double getTime() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);
    double RunTime = (double)time_now.tv_sec + (double)time_now.tv_nsec / 1.e9;
    return RunTime;
}


int main(int argc, char **argv) {

    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    double startTime, endTime;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType *) malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType *) malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType *) malloc(inputLength * sizeof(DataType));
    resultRef = (DataType *) malloc(inputLength * sizeof(DataType));

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType) (RAND_MAX + 1.0);
        hostInput2[i] = rand() / (DataType) (RAND_MAX + 1.0);
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    startTime = getTime();
    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //@@ Initialize the 1D grid and block dimensions here
    int Block_dim = ThreadNum;
    int Grid_dim = inputLength / ThreadNum + 1;

    //@@ Launch the GPU Kernel here
    vecAdd<<<Grid_dim, Block_dim>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    endTime = getTime();
    printf("Total synchronous operation time: %.6lf ms.\n", (endTime - startTime) * 1000);

    //@@ Insert code below to compare the output with the reference
    bool flag = true;
    for (unsigned int i = 0; i < inputLength; i++) {
        if (hostOutput[i] - resultRef[i] < 1e-10)
            continue;
        else {
            flag = false;
        }

    }
    if (flag)
        printf("Result match reference.");
    else
        printf("Result is different from reference.");

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}