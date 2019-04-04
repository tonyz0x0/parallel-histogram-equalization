# Parallel Histogram Equalization

A program that using CUDA and C++ to generate Histogram Equalization picture in parallel way.

> Update: Add Sobel Detection CUDA program

## Goal
***

* Understand how Histogram Equalization is applied to images.
* Write an optimized GPU code in CUDA that provides the same functionality of the histogram
equalization from OpenCV but can perform the algorithm faster.

## Instruction
***

The basic algorithm for Histagram Equalization can be divided into four steps:
 1. Calculate the histogram of the image.
    1. Considering to split one big image into multi small images and parallelly caluculate that. (not used this)
    2. Can comprise the image from the CPU, so less GPU memory will be malloced by calling cudamalloc(), which will save more time. (Not used this, this technology is called RLC, Run Length Coding)
    3. Atomicadd method is initially considered to use, but it will reduce the performance. But we have no choice, it should definitely be used for histogram.
    4. Better way to do is do per-thread histogrms parallelly, sort each gray value and reduce by key, then reduce all histograms. This is a per-block or per-thread histogram generation algorithm.
       
       This method can work well on GTX 1060(Pascal Architecture), not well on NEU Discovery cluster, so it is not used.
    5. Used down-sampling method to reduce the whole work of calculating histogram

 2. Calculate the cumulative distribution function(CDF). Using prefix sum to parallely calculate. The algorithm is called Hillis and Steele scan algorithm
 
 3. Calculate the cdfmin, maybe using the reduction tree method
 
 4. Calculate the histogram equalization value with the given formula

 5. Put the calculated value back to generate new image data, and transfer it back to host memory

## Resources
***

1. [An optimized approach to histogram computation on GPU](https://www.researchgate.net/publication/256674650_An_optimized_approach_to_histogram_computation_on_GPU)

2. [Histogram optimization with CUDA](https://www.researchgate.net/publication/311911798_Histogram_optimization_with_CUDA)

3. [Array Reduction](https://gist.github.com/jatesy/9920023)

4. [https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

5. [Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization)

6. [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)