#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TILE_SIZE 8

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);


#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \


#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);

float *input_gpu;
float *output_gpu;

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

__device__ int getid(int x,
          int y,
          int z,
          unsigned int height,
          unsigned int width,
          unsigned int length)
{
  return z*width*height + y*width + x;
}

__global__ void kernel_non_tiled(float *input,
                       float *output,
                       unsigned int height,
                       unsigned int width,
                       unsigned int length){

    int dx[6]={0,0,0,0,-1,1};
    int dy[6]={0,0,-1,1,0,0};
    int dz[6]={-1,1,0,0,0,0};

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    int z = blockIdx.z*TILE_SIZE+threadIdx.z;

    if((0 < x) && (x < width-1) &&
       (0 < y) && (y < height-1) &&
       (0 < z) && (z < length-1))
    {
            float res=0;
            for(int i=0;i<6;i++)
            {
              res += input[getid(x+dx[i],y+dy[i],z+dz[i],height,width,length)];
            }
            output[getid(x,y,z,height,width,length)] = 0.8 * res;
    }
}

__global__ void kernel_non_tiled_warmup(float *input,
                       float *output,
                       unsigned int height,
                       unsigned int width,
                       unsigned int length){

    int dx[6]={0,0,0,0,-1,1};
    int dy[6]={0,0,-1,1,0,0};
    int dz[6]={-1,1,0,0,0,0};

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    int z = blockIdx.z*TILE_SIZE+threadIdx.z;

    if((0 < x) && (x < width-1) &&
       (0 < y) && (y < height-1) &&
       (0 < z) && (z < length-1))
    {
            float res=0;
            for(int i=0;i<6;i++)
            {
              res += input[getid(x+dx[i],y+dy[i],z+dz[i],height,width,length)];
            }
            output[getid(x,y,z,height,width,length)] = 0.8 * res;
    }
}

__global__ void kernel_tiled(float *input,
                       float *output,
                       unsigned int height,
                       unsigned int width,
                       unsigned int length){

    __shared__ float cache[10][10][10];

    int nx = threadIdx.x,
        ny = threadIdx.y,
        nz = threadIdx.z;
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    int z = blockIdx.z*TILE_SIZE+threadIdx.z;

    cache[nx+1][ny+1][nz+1]=input[getid(x,y,z,height,width,length)];

    if(nx == 0 && x != 0){
      cache[0][ny+1][nz+1]=input[getid(x-1,y,z,height,width,length)];
    }
    if(nx == TILE_SIZE - 1 && x != height - 1) {
      cache[TILE_SIZE+1][ny+1][nz+1] = input[getid(x+1,y,z,height,width,length)];
    }
    if(ny == 0 && y != 0) {
      cache[nx+1][0][nz+1]=input[getid(x,y-1,z,height,width,length)];
    }
    if(ny == TILE_SIZE - 1 && y != width - 1) {
      cache[nx+1][TILE_SIZE+1][nz+1] = input[getid(x,y+1,z,height,width,length)];
    }
    if(nz == 0 && z != 0){
      cache[nx+1][ny+1][0]=input[getid(x,y,z-1,height,width,length)];
    }
    if(nz == TILE_SIZE - 1 && z != length - 1) {
      cache[nx+1][ny+1][TILE_SIZE+1] = input[getid(x,y,z+1,height,width,length)];
    }

    __syncthreads();

    int dx[6]={0,0,0,0,-1,1};
    int dy[6]={0,0,-1,1,0,0};
    int dz[6]={-1,1,0,0,0,0};

    if((0 < x) && (x < width-1) &&
       (0 < y) && (y < height-1) &&
       (0 < z) && (z < length-1))
    {
            float res=0;
            for(int i=0;i<6;i++)
            {
              res += cache[nx+dx[i]+1][ny+dy[i]+1][nz+dz[i]+1];
            }
            __syncthreads();
            output[getid(x,y,z,height,width,length)] = 0.8 * res;
    }
}

void gpu_non_tiled_func (float *input,
                   float *output,
                   unsigned int height,
                   unsigned int width,
                   unsigned int length){


    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    int gridZSize = 1 + ((length - 1) / TILE_SIZE);


    // Both are the same size (CPU/GPU).
    int size = height*width*length;

    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(float)));
    checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(float)));

    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(float)));

    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu,
        input,
        size*sizeof(float),
        cudaMemcpyHostToDevice));

    checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize, gridZSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);

    // Kernel Call
    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif

        kernel_non_tiled<<<dimGrid, dimBlock>>>(input_gpu,
                                      output_gpu,
                                      height,
                                      width,
                                      length);

        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(output,
            output_gpu,
            size*sizeof(float),
            cudaMemcpyDeviceToHost));

    // Free resources and end the program
    checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
}

void gpu_tiled_func (float *input,
                   float *output,
                   unsigned int height,
                   unsigned int width,
                   unsigned int length){


    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    int gridZSize = 1 + ((length - 1) / TILE_SIZE);


    // Both are the same size (CPU/GPU).
    int size = height*width*length;

    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(float)));
    checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(float)));

    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(float)));

    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu,
        input,
        size*sizeof(float),
        cudaMemcpyHostToDevice));

    checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize, gridZSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);

    // Kernel Call
    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif

        kernel_tiled<<<dimGrid, dimBlock>>>(input_gpu,
                                      output_gpu,
                                      height,
                                      width,
                                      length);

        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(output,
            output_gpu,
            size*sizeof(float),
            cudaMemcpyDeviceToHost));

    // Free resources and end the program
    checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
}

void gpu_warmup_func (float *input,
                   float *output,
                   unsigned int height,
                   unsigned int width,
                   unsigned int length){


    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    int gridZSize = 1 + ((length - 1) / TILE_SIZE);


    // Both are the same size (CPU/GPU).
    int size = height*width*length;

    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(float)));
    checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(float)));

    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(float)));

    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu,
        input,
        size*sizeof(float),
        cudaMemcpyHostToDevice));

    checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize, gridZSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);

    // Kernel Call
    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif

        kernel_non_tiled_warmup<<<dimGrid, dimBlock>>>(input_gpu,
                                      output_gpu,
                                      height,
                                      width,
                                      length);

        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(output,
            output_gpu,
            size*sizeof(float),
            cudaMemcpyDeviceToHost));

    // Free resources and end the program
    checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
}
