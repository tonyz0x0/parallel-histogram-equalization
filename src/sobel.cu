#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TILE_SIZE 16

#define TIMER_CREATE(t)           \
  cudaEvent_t t##_start, t##_end; \
  cudaEventCreate(&t##_start);    \
  cudaEventCreate(&t##_end);

#define TIMER_START(t)        \
  cudaEventRecord(t##_start); \
  cudaEventSynchronize(t##_start);

#define TIMER_END(t)                            \
  cudaEventRecord(t##_end);                     \
  cudaEventSynchronize(t##_end);                \
  cudaEventElapsedTime(&t, t##_start, t##_end); \
  cudaEventDestroy(t##_start);                  \
  cudaEventDestroy(t##_end);

unsigned char *input_gpu;
unsigned char *output_gpu;

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    exit(-1);
  }
#endif
  return result;
}

__global__ void sobel(unsigned char *input,
                      unsigned char *output,
                      unsigned int height,
                      unsigned int width)
{

  int x = blockIdx.x * TILE_SIZE + threadIdx.x; // column
  int y = blockIdx.y * TILE_SIZE + threadIdx.y; // row
  int Gx = 0;
  int Gy = 0;

  if (x >= 1 && x < (width - 1) && y >= 1 && y < (height - 1))
  {
    Gx = input[(y - 1) * width + x + 1] +
         2 * input[y * width + x + 1] +
         input[(y + 1) * width + x + 1] -
         (input[(y - 1) * width + x - 1] +
          2 * input[y * width + x - 1] +
          input[(y + 1) * width + x - 1]);

    Gy = input[(y - 1) * width + x - 1] +
         2 * input[(y - 1) * width + x] +
         input[(y - 1) * width + x + 1] -
         (input[(y + 1) * width + x - 1] +
          2 * input[(y + 1) * width + x] +
          input[(y + 1) * width + x + 1]);

    output[y * width + x] = (abs(Gx) + abs(Gy)) / 2;
  }
}

void gpu_function(unsigned char *input,
                  unsigned char *output,
                  unsigned int height,
                  unsigned int width)
{

  int gridXSize = 1 + ((width - 1) / TILE_SIZE);
  int gridYSize = 1 + ((height - 1) / TILE_SIZE);

  // Both are the same size (CPU/GPU).
  int size = height * width;

  // Allocate arrays in GPU memory
  checkCuda(cudaMalloc((void **)&input_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMalloc((void **)&output_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMemset(output_gpu, 0, size * sizeof(unsigned char)));

  // Copy data to GPU
  checkCuda(cudaMemcpy(input_gpu,
                       input,
                       size * sizeof(unsigned char),
                       cudaMemcpyHostToDevice));

  checkCuda(cudaDeviceSynchronize());

  // Execute algorithm

  dim3 dimGrid(gridXSize, gridYSize);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);

// Kernel Call
#if defined(CUDA_TIMING)
  float Ktime;
  TIMER_CREATE(Ktime);
  TIMER_START(Ktime);
#endif

  sobel<<<dimGrid, dimBlock>>>(input_gpu,
                               output_gpu,
                               height,
                               width);

  checkCuda(cudaPeekAtLastError());
  checkCuda(cudaDeviceSynchronize());

#if defined(CUDA_TIMING)
  TIMER_END(Ktime);
  printf("Kernel Execution Time: %f ms\n", Ktime);
#endif

  // Retrieve results from the GPU
  checkCuda(cudaMemcpy(output,
                       output_gpu,
                       size * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost));

  // Free resources and end the program
  checkCuda(cudaFree(output_gpu));
  checkCuda(cudaFree(input_gpu));
}