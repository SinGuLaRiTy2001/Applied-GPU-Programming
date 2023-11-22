
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define DataType float


double getTime() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);
    double RunTime = (double)time_now.tv_sec + (double)time_now.tv_nsec / 1.e9;
    return RunTime;
}

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    DataType ans = 0.0;

      if( rowIdx < numARows && colIdx < numBColumns  )
      {
        for(int m = 0; m < numAColumns; m++) 
        {
            ans += A[rowIdx * numAColumns + m] * B[m * numBColumns + colIdx];
        }
        C[rowIdx * numBColumns + colIdx] = ans;
      }

}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *refResult; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;
  int DBX = 32;
  int DBY = 32;
  char NOT_EQUAL[] = "result is false which is not equal to the reference result!";
  char EQUAL[] = "result is correct!";

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*) malloc(numARows * numAColumns * sizeof(DataType)); 
  hostB = (DataType*) malloc(numBRows * numBColumns * sizeof(DataType)); 
  hostC = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  refResult = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
           DataType randA = rand() / (DataType) RAND_MAX;
           hostA[i*numAColumns + j] = randA;
        }
  }
  for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
           DataType randB = rand() / (DataType) RAND_MAX;
           hostB[i*numBColumns + j] = randB;
        }
  }
  
  for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            refResult[i * numBColumns + j] = 0.0;
          for (int k = 0; k < numAColumns; k++) {
              refResult[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
          }        
        }
  }
   
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  double startTime = getTime();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double host2device = getTime() - startTime;
  printf("The time of data copy from host to device is: %f\n", host2device);

  //@@ Initialize the grid and block dimensions here

  int dimBlock[] = {DBX, DBY};
  int dimGrid[] {(numCColumns + dimBlock[0] - 1) / dimBlock[0], (numCRows + dimBlock[1] - 1) / dimBlock[1] };
 
  //@@ Launch the GPU Kernel here
  startTime = getTime();
  gemm<<<dim3(dimGrid[0], dimGrid[1], 1), dim3(dimBlock[0], dimBlock[1], 1)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double CudaKernel = getTime() - startTime;
  printf("The time of the CUDA Kernel is: %f\n", CudaKernel);  

  

  //@@ Copy the GPU memory back to the CPU here
  startTime = getTime();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  double device2host = getTime()-startTime;
  printf("The time of data copy from host to device : %f\n", device2host);

  //@@ Insert code below to compare the output with the reference
  bool isEqual = true;
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      if (fabs(hostC[i*numCColumns + j] - refResult[i * numCColumns + j]) > 1e-4)
        isEqual = false;
        break;
    }       
  }
  if(!isEqual)
      printf("%s\n", NOT_EQUAL);
  else
      printf("%s\n", EQUAL);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(refResult);

  return 0;
}
