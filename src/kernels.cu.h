/*-----------------------------------------------------------------------------------------------------
 * If problems occur - line 79 of generated .cu.obj.cmake file: delete all flags but -m**
 * --------------------------------------------------------------------------------------------------*/

#include <helper_cuda.h>    // includes for helper CUDA functions
#include <cuda_runtime_api.h>
#include <cuda.h>

/*-------------------------------------------------------------
    d_result    - output 2D data from interpolation, same size and d_xout and d_yout
    d_data      - input 2D data evaluated at index coordinates (as 1D array),
                    values are accessed at indexes equal to y * N1 + x;
    d_xout      - x indexes to evaluate input data at
    d_yout      - y indexes to evaluate input data at
    M1          - d_data width
    M2          - d_data height
    N1          - d_xout and d_yout width
    N2          - d_xout and d_yout height

    Note that ROI and input data are aligned from top left, indexed from 0

    Any samples that fall outside d_data region are set to 0
-------------------------------------------------------------*/
void bilinear_interpolation(float * d_result, const float * d_data,
                            const float * d_xout, const float * d_yout,
                            const int M1, const int M2, const int N1, const int N2,
                            dim3 blocks, dim3 threads);

/*---------------------------------------------------------------------
 * Creates a 2D mesh grid, same as using [x, y] = meshgrid(0:colums-1, 0:rows-1) in MATLAB
 * x and y values are stored in 1D array with indexes equal to y*columns+x
 * x and y must be large enough to store rows * columns elements
 * columns same as width
 * rows same as height
 * ------------------------------------------------------------------*/
void CreateGrid2D(float * d_x, float * d_y, const int columns, const int rows, dim3 blocks, dim3 threads);

/*---------------------------------------------------------------------
 * Calculates NCC given parameters
 * d_prod_mean - mean of products
 * d_mean1 - mean of input data 1
 * d_mean2 - mean of input data 2
 * d_std1 - std of input data 1
 * d_std2 - std of input data 2
 * stdthresh1 - std threshold of input data 1
 * stdthresh2 - std threshold of input data 2
 * width - width of data
 * height - height of data
 *
 * It is assumed that sizes of all inputs are the same and equal to given
 * width and height.
 *
 * Formula used:
 * NCC = (mean of producs - product of means) / (std1 * std2)
 *
 * If either std is below given threshold, indicating a homogenous region
 * in image processing, NCC is set to 0. This way avoids division by 0.
 * ------------------------------------------------------------------*/
void calcNCC(float * __restrict__ d_ncc, float * __restrict d_prod_mean,
             float * __restrict__ d_mean1, float * __restrict__ d_mean2,
             float * __restrict__ d_std1, float * __restrict__ d_std2,
             const float stdthresh1, const float stdthresh2,
             const int width, const int height,
             dim3 blocks, dim3 threads);

