#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

static int ThreadNum = 256;
static const int numStreams = 4; // Number of CUDA streams

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

double getTime() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);
    double RunTime = (double)time_now.tv_sec + (double)time_now.tv_nsec / 1.e9;
    return RunTime;
}

int main(int argc, char **argv) {
    int inputLength;
    int S_seg;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    double startTime, endTime;

    inputLength = atoi(argv[1]);
    S_seg = atoi(argv[2]); // Segment size
    printf("The input length is %d\n", inputLength);
    printf("Segment size: %d\n", S_seg);

    hostInput1 = (DataType *) malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType *) malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType *) malloc(inputLength * sizeof(DataType));
    resultRef = (DataType *) malloc(inputLength * sizeof(DataType));

    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType) (RAND_MAX + 1.0);
        hostInput2[i] = rand() / (DataType) (RAND_MAX + 1.0);
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    startTime = getTime();
    for (int i = 0; i < inputLength; i += S_seg) {
        int segSize = min(S_seg, inputLength - i);
        int streamIdx = i / S_seg % numStreams;

        cudaMemcpyAsync(deviceInput1 + i, hostInput1 + i, segSize * sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIdx]);
        cudaMemcpyAsync(deviceInput2 + i, hostInput2 + i, segSize * sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIdx]);

        int Block_dim = ThreadNum;
        int Grid_dim = segSize / ThreadNum + (segSize % ThreadNum != 0);
        vecAdd<<<Grid_dim, Block_dim, 0, streams[streamIdx]>>>(deviceInput1 + i, deviceInput2 + i, deviceOutput + i, segSize);

        cudaMemcpyAsync(hostOutput + i, deviceOutput + i, segSize * sizeof(DataType), cudaMemcpyDeviceToHost, streams[streamIdx]);
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    endTime = getTime();
    printf("Total asynchronous operation time: %.6lf ms.\n", (endTime - startTime) * 1000);

    bool flag = true;
    for (unsigned int i = 0; i < inputLength; i++) {
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-10) {
            flag = false;
            break;
        }
    }

    if (flag)
        printf("Result match reference.\n");
    else
        printf("Result is different from reference.\n");

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
