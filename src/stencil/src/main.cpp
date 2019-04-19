#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cuda.h>
#include <math.h>

using namespace std;

#define N 4
const float eps = 1e-3;
float cpu[N][N][N],
      b[N][N][N],
      gpu_non_tiled[N][N][N],
      gpu_tiled[N][N][N];

// Timer function
double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

void show(float a[N][N][N])
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            for(int k=0;k<N;k++)
                printf("%3f ",a[i][j][k]);
            printf("\n");
        }
        printf("\n");
    }
}

extern void gpu_non_tiled_func (float *input,
                          float *output,
                          unsigned int height,
                          unsigned int width,
                          unsigned int length);

extern void gpu_tiled_func (float *input,
                           float *output,
                           unsigned int height,
                           unsigned int width,
                           unsigned int length);

extern void gpu_warmup_func (float *input,
                            float *output,
                            unsigned int height,
                            unsigned int width,
                            unsigned int length);

int check(float a,float b)
{
    float res=a-b;
    if(res<0) res=-res;
    return res<eps;
}

int main( int argc, const char** argv ) {
    double start_gpu, finish_gpu;
    unsigned int length = N;
    unsigned int height = N;
    unsigned int  width = N;
    int dx[6]={0,0,0,0,-1,1};
    int dy[6]={0,0,-1,1,0,0};
    int dz[6]={-1,1,0,0,0,0};

    // Matrix Initialization
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<N;k++) {
                b[i][j][k] = 1000.0*(float)rand()/RAND_MAX;
            }
        }
    }

    ///////////////////////
    // START CPU Processing
    ///////////////////////

    for(int i=1;i<N-1;i++) {
        for(int j=1;j<N-1;j++) {
            for(int k=1;k<N-1;k++) {
                cpu[i][j][k]=0;
                for(int m=0;m<6;m++)
                {
                    cpu[i][j][k]+=b[i+dx[m]][j+dy[m]][k+dz[m]];
                }
                cpu[i][j][k]*=0.8;
            }
        }
    }

    ///////////////////////
    // START GPU Warmup
    ///////////////////////

    gpu_warmup_func((float *) b, // Input
                 (float *) gpu_non_tiled, // Output
                 height,
                 width,
                 length);

    ///////////////////////
    // START GPU Non-tile Processing
    ///////////////////////

    start_gpu = CLOCK();

    gpu_non_tiled_func((float *) b, // Input
                 (float *) gpu_non_tiled, // Output
                 height,
                 width,
                 length);

    finish_gpu = CLOCK();

    cout << "GPU execution time in Non-Tiled version: " << finish_gpu - start_gpu << " ms" << endl;

    ///////////////////////
    // START GPU Tile Warmup
    ///////////////////////

    start_gpu = CLOCK();

    gpu_tiled_func((float *) b, // Input
                 (float *) gpu_tiled, // Output
                 height,
                 width,
                 length);

    finish_gpu = CLOCK();

    cout << "GPU execution time in Tiled version: " << finish_gpu - start_gpu << " ms" << endl;

    // Check
    int rc = 1;
    for(int i=0;i<N;i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if (!check(cpu[i][j][k], gpu_non_tiled[i][j][k]) &&
                    !check(cpu[i][j][k], gpu_tiled[i][j][k])) {
                    rc = 0;
                }
            }
        }
    }
    show(cpu);
    cout << "---" << endl;
    show(gpu_non_tiled);
    cout << "---" << endl;
    show(gpu_tiled);

    if(rc == 0) {
        cout << "Check...[Failed]" << endl;
    } else {
        cout << "Check...[Success]" << endl;
    }
    return 0;

}
