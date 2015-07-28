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
 * Calculates affine transformation x and y indexes, indexes returned
 * are 0 based indexed but calculation performed is based on 1 based indexing, i.e.
 *     [x' + 1]   [h11 h12 h13]   [x + 1]
 * w * [y' + 1] = [h21 h22 h23] * [y + 1]
 *     [   1  ]   [h31 h32 h33]   [  1  ]
 * d_x - output x indexes, 0 based indexing
 * d_y - output y indexes, 0 based indexing
 * width and height - dimensions of arrays
 * ------------------------------------------------------------------*/
void transform_indexes(float * d_x, float * d_y,
                       const float h11, const float h12, const float h13,
                       const float h21, const float h22, const float h23,
                       const float h31, const float h32, const float h33,
                       const int width, const int height, dim3 blocks, dim3 threads);

/*---------------------------------------------------------------------
 * Calculates NCC given parameters
 * d_prod_mean - mean of products
 * d_mean1 - mean of input data 1
 * d_mean2 - mean of input data 2
 * d_std1 - std of input data 1
 * d_std2 - std of input data 2
 * stdthresh1 - std threshold of input data 1
 * stdthresh2 - std threshold of input data 2
 * width and height - dimensions of arrays
 *
 * Formula used:
 * NCC = (mean of producs - product of means) / (std1 * std2)
 *
 * If either std is below given threshold, indicating a homogenous region
 * in image processing, NCC is set to 0. This way avoids division by 0.
 * ------------------------------------------------------------------*/
void calcNCC(float * d_ncc, const float * d_prod_mean,
             const float * d_mean1, const float * d_mean2,
             const float * d_std1, const float * d_std2,
             const float stdthresh1, const float stdthresh2,
             const int width, const int height,
             dim3 blocks, dim3 threads);

/*------------------------------------------------------------------------
 * Updates depthmap and best NCC arrays if newly calculated NCC is greater
 * d_depthmap - output updated depthmap
 * d_bestncc - output updated best NCC
 * d_currentncc - input newly calculated NCC
 * d_currentdepth - depth at which new NCC was calculated
 * width and height - dimensions of all arrays
 * ---------------------------------------------------------------------*/
void update_arrays(float * d_depthmap, float * d_bestncc,
                   const float * d_currentncc, const float current_depth,
                   const int width, const int height,
                   dim3 blocks, dim3 threads);

/*------------------------------------------------------------------------
 * Adds d_depthmap values to d_depthmap_out and increases count (used later
 * averaging) if corresponding d_ncc value is greater than nccthreshold
 * d_depthmap_out - output sum of depthmaps
 * d_count - output count (how many times ncc was greater than threshold)
 * d_depthmap - input calculated depthmap
 * d_ncc - input best ncc
 * nccthreshold - NCC threshold
 * width and height - dimensions of arrays
 *----------------------------------------------------------------------*/
void sum_depthmap_NCC(float * d_depthmap_out, float * d_count,
                      const float * d_depthmap, const float * d_ncc,
                      const float nccthreshold,
                      const int width, const int height,
                      dim3 blocks, dim3 threads);
/*------------------------------------------------------------------------
 * Calculates STD given mean and mean of squares of data
 * d_std - output STD values
 * d_mean - input mean
 * d_mean_of_squares - input mean of squares
 * widht and height - dimensions of arrays
 * ---------------------------------------------------------------------*/
void calculate_STD(float * d_std, const float * d_mean,
                   const float * d_mean_of_squares,
                   const int width, const int height,
                   dim3 blocks, dim3 threads);

/*------------------------------------------------------------------------
 * Sets all values in d_output to 'value'
 * ---------------------------------------------------------------------*/
void set_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads);

/*-----------------------------------------------------------------------
 * Element wise multiplication
 * --------------------------------------------------------------------*/
void element_multiply(float * d_output, const float * d_input1,
                      const float * d_input2,
                      const int width, const int height,
                      dim3 blocks, dim3 threads);

/*-----------------------------------------------------------------------
 * Element wise divisiont d_input1 ./ d_input2
 * --------------------------------------------------------------------*/
void element_rdivide(float * d_output, const float * d_input1,
                     const float * d_input2,
                     const int width, const int height,
                     dim3 blocks, dim3 threads);

// DOES NOT WORK YET
/*-----------------------------------------------------------------------
 * float to uchar conversion, using bounds
 * all values below min are set to 0, all above - UCHAR_MAX, rest
 * are set to UCHAR_MAX * (d_input - min) / (max - min)
 * --------------------------------------------------------------------*/
void convert_float_to_uchar(unsigned char *d_output, const float * d_input,
                            const float min, const float max,
                            const int width, const int height,
                            dim3 blocks, dim3 threads);


/*-----------------------------------------------------------------------
 * Calculates windowed mean row wise
 * d_output - output means
 * d_input - input array
 * winsize - size of the window
 * squared - if true, will calculate mean of squares
 * width and height - dimensions of arrays
 * --------------------------------------------------------------------*/
void windowed_mean_row(float * d_output, const float * d_input,
                       const unsigned int winsize, const bool squared,
                       const int width, const int height, dim3 blocks, dim3 threads);

/*-----------------------------------------------------------------------
 * Calculates windowed mean column wise
 * d_output - output means
 * d_input - input array
 * winsize - size of the window
 * squared - if true, will calculate mean of squares
 * width and height - dimensions of arrays
 * --------------------------------------------------------------------*/
void windowed_mean_column(float * d_output, const float * d_input,
                          const unsigned int winsize, const bool squared,
                          const int width, const int height, dim3 blocks, dim3 threads);

// Converts unsigned char image to float DOES NOT WORK
void convert_uchar_to_float(float * d_output, const unsigned char * d_input,
                            const int width, const int height,
                            dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates x and y of P matrix (as used in OpenCV denoising_TVL1)
 * -------------------------------------------------------------------*/
void denoising_TVL1_calculateP(float * d_Px, float * d_Py,
                               const float * d_input,
                               const float sigma,
                               const int width, const int height,
                               dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Scale each element in d_output by scale
 * -------------------------------------------------------------------*/
void element_scale(float * d_output, const float scale, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Add value to each element in d_output
 * -------------------------------------------------------------------*/
void element_add(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Set QNANs to value
 * -------------------------------------------------------------------*/
void set_QNAN_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Updates d_output given the parameters tau, theta, lambda, sigma and
 * based on values in d_R, d_Px, d_Py and d_origin (based on OpenCV
 * implementation of TVl1 denoising)
 * -------------------------------------------------------------------*/
void denoising_TVL1_update(float * d_output, float * d_R,
                           const float * d_Px, const float * d_Py, const float * d_origin,
                           const float tau, const float theta, const float lambda, const float sigma,
                           const int width, const int height, dim3 blocks, dim3 threads);

/*========================================================================================================*\
 * Total generalised varation multistereo view functions
 *
 * page 32 of http://gpu4vision.icg.tugraz.at/papers/2012/graber_master.pdf#pub68
 * notation used is the same, prefix "d_" means that the variable resides on device memory
 *
 * At each point of the image variables will have:
 * p - 2 components, initialized to 0
 * q - 4 components, initialized to 0
 * r - 1 component for each source view, initialized to 0
 * u - 1 component, initialized to some guess
 * u1 - 2 components, initialized to 0
/*========================================================================================================*\

/*----------------------------------------------------------------------
 * Updates both p components depending on u and u1
 * -------------------------------------------------------------------*/
void TGV2_updateP(float * d_Px, float * d_Py, const float * d_u, const float * d_u1x, const float * d_u1y,
                  const float alpha1, const float sigma, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Updates all 4 q components depending on u1
 * -------------------------------------------------------------------*/
void TGV2_updateQ(float * d_Qx, float * d_Qy, float * d_Qz, float * d_Qw, const float * d_u1x, const float * d_u1y,
                  const float alpha0, const float sigma, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Updates r and sums r*Iu cumulatively to prodsum, used for update u
 * -------------------------------------------------------------------*/
void TGV2_updateR(float * d_r, float * d_prodsum, const float * d_u, const float * d_u0, const float * d_It, const float * d_Iu,
                  const float sigma, const float lambda, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Updates u, u1, ubar and u1bar given calculated images It and Iu
 * -------------------------------------------------------------------*/
void TGV2_updateU(float * d_u, float * d_u1x, float * d_u1y, float * d_ubar, float * d_u1xbar, float * d_u1ybar,
                  const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                  const float * d_Qz, const float * d_Qw, const float * d_prodsum, const float alpha0,
                  const float alpha1, const float tau, const float lambda,
                  const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates x and y pixel coordinates in the source view of each pixel
 * from the reference view given camera calibration matrix K,
 * relative rotation and translation between the views and real pixel depth.
 * Also outputs 3D X, Y and Z coordinates of the pixel
 * -------------------------------------------------------------------*/
void TGV2_transform_coordinates(float * d_x, float * d_y, float * d_X, float * d_Y, float * d_Z, const float * d_u,
                                const double K[3][3], const double Rrel[3][3], const double trel[3], const double invK[3][3],
                                const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates in1 - in2
 * -------------------------------------------------------------------*/
void subtract(float * d_out, const float * d_in1, const float * d_in2, const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates camera coordinate derivatives dX, dY and dZ given inverse of
 * camera calibration matrix invK and relative rotation Rrel
 * -------------------------------------------------------------------*/
void TGV2_calculate_coordinate_derivatives(float * d_dX, float * d_dY, float * d_dZ, const double invK[3][3], const double Rrel[3][3],
                                           const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates derivative of f(x,u) given X, Y and Z coordinates and their
 * derivatives dX, dY and dZ
 * -------------------------------------------------------------------*/
void TGV2_calculate_derivativeF(float * d_dfx, float * d_dfy, const float * d_X, const float * d_dX, const float * d_Y, const float * d_dY,
                                const float * d_Z, const float * d_dZ, const float fx, const float fy,
                                const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates image Iu given interpolated image I(f(x,u) and f(x,u) derivatives
 * dfx and dfy
 * -------------------------------------------------------------------*/
void TGV2_calculate_Iu(float * d_Iu, const float * d_I, const float * d_dfx, const float * d_dfy,
                       const int width, const int height, dim3 blocks, dim3 threads);

/*----------------------------------------------------------------------
 * Calculates anisotropic diffusion tensor according to formula
 * T = exp(-beta * |grad(Img)| ^ gamma) * n * n' + m * m'
 * where n = grad(Img) / |grad(Img)| and m is normal to n
 * -------------------------------------------------------------------*/
void Anisotropic_diffusion_tensor(float * d_T11, float * d_T12, float * d_T21, float * d_T22, const float * d_Img,
                                  const float beta, const float gamma, const int width, const int height,
                                  dim3 blocks, dim3 threads);

void TGV2_updateU_tensor_weighed(float * d_u, float * d_u1x, float * d_u1y, const float * d_T11, const float * d_T12,
                                 const float * d_T21, const float * d_T22, float * d_ubar, float * d_u1xbar, float * d_u1ybar,
                                 const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                                 const float * d_Qz, const float * d_Qw, const float * d_prodsum,
                                 const float alpha0, const float alpha1, const float tau, const float lambda,
                                 const int width, const int height, dim3 blocks, dim3 threads);

void TGV2_updateP_tensor_weighed(float * d_Px, float * d_Py, const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                 const float * d_u, const float * d_u1x, const float * d_u1y, const float alpha1,
                                 const float sigma, const int width, const int height, dim3 blocks, dim3 threads);
