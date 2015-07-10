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
    M1          - d_data pitch
    M2          - d_data height
    N1          - d_xout and d_yout pitch
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


