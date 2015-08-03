 /**
 *  \file kernels.cu.h
 *  \brief Header file containing kernel invocation functions for planesweep and total generalized variation multiview stereo algorithms
 *
 * If there is an error generating cuda file, leave only <em>-m**</em> flag in line \b 79 of <em>*.cu.obj.cmake</em> file that is giving errors
 */
 
#ifndef KERNELS_CU_H
#define KERNELS_CU_H

#include <helper_cuda.h>    // includes for helper CUDA functions
#include <cuda_runtime_api.h>
#include <cuda.h>

/** \addtogroup general  General
* \brief General CUDA kernel functions, mostly for float type data
* @{
*/

/**
 *  \brief Perform bilinear interpolation on data
 *  
 *  \param d_result pointer to output 2D data from interpolation, same size and \a d_xout and \a d_yout
 *  \param d_data   pointer to input 2D data evaluated at index coordinates
 *  \param d_xout   pointer to \a x indexes to evaluate input data at
 *  \param d_yout   pointer to \a y indexes to evaluate input data at
 *  \param M1       \a d_data width
 *  \param M2       \a d_data height
 *  \param N1       \a d_xout and \a d_yout width
 *  \param N2       \a d_xout and \a d_yout height
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details Any samples that fall outside \a d_data region are set to 0
 */
void bilinear_interpolation(float * d_result, const float * d_data,
                            const float * d_xout, const float * d_yout,
                            const int M1, const int M2, const int N1, const int N2,
                            dim3 blocks, dim3 threads);

 /**
 *  \brief Calculate normalized cross correlation (NCC) for each element
 *  
 *  \param d_ncc       pointer to output NCC values
 *  \param d_prod_mean pointer to input mean of products
 *  \param d_mean1     pointer to input mean of data 1
 *  \param d_mean2     pointer to input mean of data 2
 *  \param d_std1      pointer to input standard deviation of data 1
 *  \param d_std2      pointer to input standard deviation of data 2
 *  \param stdthresh1  standard deviation threshold for data 1
 *  \param stdthresh2  standard deviation threshold for data 2
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details NCC is calculated as <em>(mean of products - product of means) / (std1 * std2)</em>
 *
 If either STD is below given threshold, indicating a homogeneous region
 in an image, NCC is set to 0. This way avoids division by 0.
 */
void calcNCC(float * d_ncc, const float * d_prod_mean,
             const float * d_mean1, const float * d_mean2,
             const float * d_std1, const float * d_std2,
             const float stdthresh1, const float stdthresh2,
             const int width, const int height,
             dim3 blocks, dim3 threads);

 /**
 *  \brief Perform standard deviation calculation
 *  
 *  \param d_std               pointer to output standard deviation values
 *  \param d_mean              pointer to input means
 *  \param d_mean_of_squares   pointer to input means of squares
 *  \param width               width of given arrays
 *  \param height              height of given arrays
 *  \param blocks              kernel grid dimensions
 *  \param threads             single block dimensions
 *  \return No return value
 *  
 *  \details STD is given by <em>mean of squares - square of mean</em>
 */
void calculate_STD(float * d_std, const float * d_mean,
                   const float * d_mean_of_squares,
                   const int width, const int height,
                   dim3 blocks, dim3 threads);

 /**
 *  \brief Set to constant value
 *  
 *  \param d_output    pointer to data to set values
 *  \param value       value to set to
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void set_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Element wise mutiplication
 *  
 *  \param d_output    pointer to output data
 *  \param d_input1    pointer to input data 1
 *  \param d_input2    pointer to input data 2
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details Equivalent to <em>input1 .* input2</em> in \a \b MATLAB
 */
void element_multiply(float * d_output, const float * d_input1,
                      const float * d_input2,
                      const int width, const int height,
                      dim3 blocks, dim3 threads);

 /**
 *  \brief Element wise division
 *  
 *  \param d_output    pointer to output data
 *  \param d_input1    pointer to input data 1
 *  \param d_input2    pointer to input data 2
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details Equivalent to <em>input1 ./ input2</em> in \a \b MATLAB
 */
void element_rdivide(float * d_output, const float * d_input1,
                     const float * d_input2,
                     const int width, const int height,
                     dim3 blocks, dim3 threads);

 /**
 *  \brief Conversion from float to unsigned char image
 *  
 *  \param d_output    pointer to output unsigned char data
 *  \param d_input     pointer to input float data
 *  \param min         minimum values of input array to set to 0
 *  \param max         maximum values of input array to set to 255
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details <b>DOES NOT WORK, OUTPUTS WRONG VALUES</b>
 * All values below \a min are set to 0, above \a max - to 255
 */
void convert_float_to_uchar(unsigned char *d_output, const float * d_input,
                            const float min, const float max,
                            const int width, const int height,
                            dim3 blocks, dim3 threads);

 /**
 *  \brief Row wise mean calculation
 *  
 *  \param d_output    pointer to output means
 *  \param d_input     pointer to input data
 *  \param winsize     size of window to calculate means in
 *  \param squared     calculate mean of squares?
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details 
 */
void windowed_mean_row(float * d_output, const float * d_input,
                       const unsigned int winsize, const bool squared,
                       const int width, const int height, dim3 blocks, dim3 threads);

/**
 *  \brief Column wise mean calculation
 *  
 *  \param d_output    pointer to output means
 *  \param d_input     pointer to input data
 *  \param winsize     size of window to calculate means in
 *  \param squared     calculate mean of squares?
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details 
 */
void windowed_mean_column(float * d_output, const float * d_input,
                          const unsigned int winsize, const bool squared,
                          const int width, const int height, dim3 blocks, dim3 threads);

/**
 *  \brief Conversion from unsigned char to float array
 *  
 *  \param d_output    pointer to output float data
 *  \param d_input     pointer to input unsigned char data
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details <b>DOES NOT WORK, OUTPUTS WRONG VALUES</b>
 */
void convert_uchar_to_float(float * d_output, const unsigned char * d_input,
                            const int width, const int height,
                            dim3 blocks, dim3 threads);

 /**
 *  \brief Scale elements of given array
 *  
 *  \param d_output    pointer to data values to scale
 *  \param scale       scaling value
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void element_scale(float * d_output, const float scale, const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Add constant to given array elements
 *  
 *  \param d_output    pointer to data values to modify
 *  \param value       constant to add to array values
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void element_add(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Find and replace all \a QNANs with scalar value
 *  
 *  \param d_output    pointer to data to replace \a QNANs in
 *  \param value       \a QNAN replacement value
 *  \param width       width of given arrays
 *  \param height      height of given arrays
 *  \param blocks      kernel grid dimensions
 *  \param threads     single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void set_QNAN_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads);

/**
*  \brief Calculate 3D world positions of pixels in an image
*
*  \param d_x     pointer to output x coordinates of a pixel
*  \param d_y     pointer to output y coordinates of a pixel
*  \param d_z     pointer to output z coordinates of a pixel, must contain pixel depth as input
*  \param Rrel    rotation matrix from camera to world coordinates
*  \param trel    translation vector from camera to world position
*  \param invK    inverse camera calibration matrix \f$K^{-1}\f$
*  \param width   width of given arrays
*  \param height  height of given arrays
*  \param blocks  kernel grid dimensions
*  \param threads single block dimensions
*  \return No return value
*
*  \details
*/
void compute3D(float * d_x, float * d_y, float * d_z, const double Rrel[3][3], const double trel[3],
              const double invK[3][3], const int width, const int height, dim3 blocks, dim3 threads);

/**
*  \brief Subtract <em>in1 - in2</em>
*
*  \param d_out    pointer to output array
*  \param d_in1    pointer to input array to subtract from
*  \param d_in2    pointer to input array to subtract
*  \param width    width of given arrays
*  \param height   height of given arrays
*  \param blocks   kernel grid dimensions
*  \param threads  single block dimensions
*  \return No return value
*
*  \details
*/
void subtract(float * d_out, const float * d_in1, const float * d_in2, const int width, const int height, dim3 blocks, dim3 threads);
 /** @} */ // group general

/** \addtogroup planesweep  Planesweep
* \brief Planesweep algorithm CUDA functions
* @{
*/

/**
*  \brief Apply homography (3x3 matrix) transformation to image pixel coordinates
that fall between [0,0] and (height, width)
*
*  \param d_x     pointer to output \a x indexes
*  \param d_y     pointer to output \a y indexes
*  \param h11     matrix element at \b (1,1)
*  \param h12     matrix element at \b (1,2)
*  \param h13     matrix element at \b (1,3)
*  \param h21     matrix element at \b (2,1)
*  \param h22     matrix element at \b (2,2)
*  \param h23     matrix element at \b (2,3)
*  \param h31     matrix element at \b (3,1)
*  \param h32     matrix element at \b (3,2)
*  \param h33     matrix element at \b (3,3)
*  \param width   width of given arrays
*  \param height  height of given arrays
*  \param blocks  kernel grid dimensions
*  \param threads single block dimensions
*  \return No return value
*
*  \details Transformation is applied to 1 based index coordinates
*/
void transform_indexes(float * d_x, float * d_y,
                      const float h11, const float h12, const float h13,
                      const float h21, const float h22, const float h23,
                      const float h31, const float h32, const float h33,
                      const int width, const int height, dim3 blocks, dim3 threads);

/**
*  \brief Depthmap update function
*
*  \param d_depthmap      pointer to depthmap to be updated
*  \param d_bestncc       pointer to best NCC values to be updated
*  \param d_currentncc    pointer to input NCC values calculated at \a current_depth
*  \param current_depth   current depth of planesweep algorithm
*  \param width           width of given arrays
*  \param height          height of given arrays
*  \param blocks          kernel grid dimensions
*  \param threads         single block dimensions
*  \return No return value
*
*  \details If current NCC value is greater than best value, best value is changed to current
and depthmap value is changed to current depth
*/
void update_arrays(float * d_depthmap, float * d_bestncc,
                  const float * d_currentncc, const float current_depth,
                  const int width, const int height,
                  dim3 blocks, dim3 threads);

/**
*  \brief Sum depthmaps and increases summation count if corresponding NCC value is greater than threshold
*
*  \param d_depthmap_out  pointer to summed depthmap to be updated
*  \param d_count         pointer to summation count to be updated
*  \param d_depthmap      pointer to input depthmap to be added
*  \param d_ncc           pointer to input NCC values of depthmap
*  \param nccthreshold    NCC threshold
*  \param width           width of given arrays
*  \param height          height of given arrays
*  \param blocks          kernel grid dimensions
*  \param threads         single block dimensions
*  \return No return value
*
*  \details Keeping count is required for averaging in later step
*/
void sum_depthmap_NCC(float * d_depthmap_out, float * d_count,
                     const float * d_depthmap, const float * d_ncc,
                     const float nccthreshold,
                     const int width, const int height,
                     dim3 blocks, dim3 threads);

/** @} */ // group planesweep

/** \addtogroup TVL1  TVL1 denoising
* \brief TVL1 single image input denoising functions.
*
* For more info see http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
*
* Notation used is the same, prefix "d_" means that the variable resides on device memory
*
* At each point of the image variables will have:
* * \a \f$p \in \mathbb{R}^2\f$ - initialized to 0
* * \a \f$r \in \mathbb{R}^1\f$ - initialized to 0
* * \a \f$u \in \mathbb{R}^1\f$ - initialized to normalized input image
* @{
*/

 /**
 *  \brief Update dual variavle \f$r\f$ and primal variable \f$u\f$ values
 *  
 *  \param d_output pointer to primal variable \f$u\f$ values to be updated
 *  \param d_R      pointer to dual variable \f$r\f$ values to be updated
 *  \param d_Px     pointer to input component \a x of dual variable \f$p\f$
 *  \param d_Py     pointer to input component \a y of dual variable \f$p\f$
 *  \param d_origin pointer to input original input image normalized and scaled by \f$-\sigma\f$
 *  \param tau      TVL1 parameter \f$\tau\f$
 *  \param theta    TVL1 parameter \f$\theta\f$
 *  \param lambda   TVL1 parameter \f$\lambda\f$
 *  \param sigma    TVL1 parameter \f$\sigma\f$
 *  \param width    width of given arrays
 *  \param height   height of given arrays
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details Same as in \a denoise_TVL1 in \a OpenCV
 */
void denoising_TVL1_update(float * d_output, float * d_R,
                           const float * d_Px, const float * d_Py, const float * d_origin,
                           const float tau, const float theta, const float lambda, const float sigma,
                           const int width, const int height, dim3 blocks, dim3 threads);

/**
 *  \brief Update dual variable \f$r\f$ and primal variable \f$u\f$ values by weighing with 2 by 2 tensor \f$T\f$
 *  
 *  \param d_output pointer to primal variable \f$u\f$ values to be updated
 *  \param d_R      pointer to dual variable \f$r\f$ values to be updated
 *  \param d_Px     pointer to input component \a x of dual variable \f$p\f$
 *  \param d_Py     pointer to input component \a y of dual variable \f$p\f$
 *  \param d_origin pointer to input original input image normalized and scaled by \f$-\sigma\f$
 *  \param d_T11    pointer to input values of \f$T(1,1)\f$
 *  \param d_T12    pointer to input values of \f$T(1,2)\f$
 *  \param d_T21    pointer to input values of \f$T(2,1)\f$
 *  \param d_T22    pointer to input values of \f$T(2,2)\f$
 *  \param tau      TVL1 parameter \f$\tau\f$
 *  \param theta    TVL1 parameter \f$\theta\f$
 *  \param lambda   TVL1 parameter \f$\lambda\f$
 *  \param sigma    TVL1 parameter \f$\sigma\f$
 *  \param width    width of given arrays
 *  \param height   height of given arrays
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void denoising_TVL1_update_tensor_weighed(float * d_output, float * d_R,
                                          const float * d_Px, const float * d_Py, const float * d_origin,
                                          const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                          const float tau, const float theta, const float lambda, const float sigma,
                                          const int width, const int height, dim3 blocks, dim3 threads);

/**
*  \brief Update dual variable \f$p\f$ values in TVL1 algorithm
*
*  \param d_Px        pointer to values of \a x component of \f$p\f$ to be updated
*  \param d_Py        pointer to values of \a y component of \f$p\f$ to be updated
*  \param d_input     pointer to input primal variable \f$u\f$ values
*  \param sigma       TVL1 parameter \f$\sigma\f$
*  \param width       width of given arrays
*  \param height      height of given arrays
*  \param blocks      kernel grid dimensions
*  \param threads     single block dimensions
*  \return No return value
*
*  \details Same as \a denoise_TVL1 in \a OpenCV
*/
void denoising_TVL1_calculateP(float * d_Px, float * d_Py,
                              const float * d_input,
                              const float sigma,
                              const int width, const int height,
                              dim3 blocks, dim3 threads);

/**
*  \brief Update dual variable \f$p\f$ values in TVL1 algorithm by weighing with 2 by 2 tensor \f$T\f$
*
*  \param d_Px        pointer to values of \a x component of \f$p\f$ to be updated
*  \param d_Py        pointer to values of \a y component of \f$p\f$ to be updated
*  \param d_T11       pointer to input values of \f$T(1,1)\f$
*  \param d_T12       pointer to input values of \f$T(1,2)\f$
*  \param d_T21       pointer to input values of \f$T(2,1)\f$
*  \param d_T22       pointer to input values of \f$T(2,2)\f$
*  \param d_input     pointer to input primal variable \f$u\f$ values
*  \param sigma       TVL1 parameter \f$\sigma\f$
*  \param width       width of given arrays
*  \param height      height of given arrays
*  \param blocks      kernel grid dimensions
*  \param threads     single block dimensions
*  \return No return value
*
*  \details
*/
void denoising_TVL1_calculateP_tensor_weighed(float * d_Px, float * d_Py,
                                             const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                             const float * d_input, const float sigma,
                                             const int width, const int height,
                                             dim3 blocks, dim3 threads);
 /** @} */ // group TVL1

 /** \addtogroup TGV2  TGV2 Multiview Stereo
 * \brief Total generalised varation multistereo view functions.
 *
 * For more info see page 32 of http://gpu4vision.icg.tugraz.at/papers/2012/graber_master.pdf#pub68
 *
 * Notation used is the same, prefix "d_" means that the variable resides on device memory
 *
 * At each point of the image variables will have:
 * * \a \f$p \in \mathbb{R}^2\f$ - initialized to 0
 * * \a \f$q \in \mathbb{R}^4\f$ - initialized to 0
 * * \a \f$r \in \mathbb{R}^1\f$ - one for each source view, initialized to 0
 * * \a \f$u \in \mathbb{R}^1\f$ - initialized to some guess
 * * \a \f$\overline{u} \in \mathbb{R}^1\f$ - initialized to \f$u\f$
 * * \a \f$u_1 \in \mathbb{R}^2\f$ - initialized to 0
 * * \a \f$\overline{u}_1 \in \mathbb{R}^2\f$ - initialized to 0
 * @{
 */

 /**
 *  \brief Update dual variable \f$p\f$ values using TGV2 algorithm
 *  
 *  \param d_Px    pointer to x component of dual variable \f$p\f$ to be updated
 *  \param d_Py    pointer to y component of dual variable \f$p\f$ to be updated
 *  \param d_u     pointer to primal variable \f$\overline{u}\f$ values
 *  \param d_u1x   pointer to x component of primal variable \f$\overline{u}_1\f$
 *  \param d_u1y   pointer to y component of primal variable \f$\overline{u}_1\f$
 *  \param alpha1  TGV2 weight parameter \f$\alpha_1\f$
 *  \param sigma   TGV2 parameter \f$\sigma\f$
 *  \param width   width of given arrays
 *  \param height  height of given arrays
 *  \param blocks  kernel grid dimensions
 *  \param threads single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_updateP(float * d_Px, float * d_Py, const float * d_u, const float * d_u1x, const float * d_u1y,
                  const float alpha1, const float sigma, const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Update dual variable \f$q\f$ using TGV2 algorithm
 *  
 *  \param d_Qx    pointer to x component of dual variable \f$q\f$ to be updated
 *  \param d_Qy    pointer to y component of dual variable \f$q\f$ to be updated
 *  \param d_Qz    pointer to z component of dual variable \f$q\f$ to be updated
 *  \param d_Qw    pointer to w component of dual variable \f$q\f$ to be updated
 *  \param d_u1x   pointer to x component of primal variable \f$\overline{u}_1\f$
 *  \param d_u1y   pointer to y component of primal variable \f$\overline{u}_1\f$
 *  \param alpha0  TGV2 weight parameter \f$\alpha_0\f$
 *  \param sigma   TGV2 parameter \f$\sigma\f$
 *  \param width   width of given arrays
 *  \param height  height of given arrays
 *  \param blocks  kernel grid dimensions
 *  \param threads single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_updateQ(float * d_Qx, float * d_Qy, float * d_Qz, float * d_Qw, const float * d_u1x, const float * d_u1y,
                  const float alpha0, const float sigma, const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Update dual variable \f$r\f$ using TGV2 algorithm and cumulatively sum \f$r\f$ and \f$I_u\f$ product
 *  
 *  \param d_r       		pointer to dual variable \f$r\f$ to be updated
 *  \param d_prodsum            pointer to cumulative sum of \f$r\f$ and \f$I_u\f$ to be udpated
 *  \param d_u       		pointer to primal variable \f$u\f$
 *  \param d_u0      		pointer to primal variable \f$u\f$ initialization value
 *  \param d_It                 pointer to difference image \f$I_t\f$ of interpolated at \f$u_0\f$ and reference
 *  \param d_Iu                 pointer to derivative image \f$I_u\f$
 *  \param sigma                TGV2 parameter \f$\sigma\f$
 *  \param lambda               TGV2 parameter \f$\lambda\f$
 *  \param width    		width of given arrays
 *  \param height   		height of given arrays
 *  \param blocks  		kernel grid dimensions
 *  \param threads  		single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_updateR(float * d_r, float * d_prodsum, const float * d_u, const float * d_u0, const float * d_It, const float * d_Iu,
                  const float sigma, const float lambda, const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Update primal variables \f$u\f$, \f$\overline{u}\f$, \f$u_1\f$ and \f$\overline{u}_1\f$ using TGV2 algorithm
 *  
 *  \param d_u          pointer to primal variable \f$u\f$ to be updated
 *  \param d_u1x        pointer to component \a x of primal variable \f$u_1\f$ to be updated
 *  \param d_u1y        pointer to component \a x of primal variable \f$u_1\f$ to be updated
 *  \param d_ubar       pointer to primal variable \f$\overline{u}\f$ to be updated
 *  \param d_u1xbar     pointer to component \a x of primal variable \f$\overline{u}_1\f$ to be updated
 *  \param d_u1ybar     pointer to component \a y of primal variable \f$\overline{u}_1\f$ to be updated
 *  \param d_Px         pointer to component \a x of dual variable \f$p\f$
 *  \param d_Py         pointer to component \a y of dual variable \f$p\f$
 *  \param d_Qx         pointer to component \a x of dual variable \f$q\f$
 *  \param d_Qy         pointer to component \a y of dual variable \f$q\f$
 *  \param d_Qz         pointer to component \a z of dual variable \f$q\f$
 *  \param d_Qw         pointer to component \a w of dual variable \f$q\f$
 *  \param d_prodsum    pointer to sum of \f$I^i_u r^i\f$
 *  \param alpha0       TGV2 weight parameter \f$\alpha_0\f$
 *  \param alpha1       TGV2 weight parameter \f$\alpha_1\f$
 *  \param tau          TGV2 paramater \f$\tau\f$
 *  \param lambda       TGV2 parameter \f$\lambda\f$
 *  \param width        width of given arrays
 *  \param height       height of given arrays
 *  \param blocks       kernel grid dimensions
 *  \param threads      single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_updateU(float * d_u, float * d_u1x, float * d_u1y, float * d_ubar, float * d_u1xbar, float * d_u1ybar,
                  const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                  const float * d_Qz, const float * d_Qw, const float * d_prodsum, const float alpha0,
                  const float alpha1, const float tau, const float lambda,
                  const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Calculates transformed pixel coordinates given depthmap and position and camera matrices
 *  
 *  \param d_x          pointer to output transformed pixel \a x coordinates
 *  \param d_y          pointer to output transformed pixel \a y coordinates
 *  \param d_X          pointer to output transformed world \a x coordinates in camera reference frame
 *  \param d_Y          pointer to output transformed world \a y coordinates in camera reference frame
 *  \param d_Z          pointer to output transformed world \a z coordinates in camera reference frame
 *  \param d_u          pointer to input depthmap
 *  \param K            camera calibration matrix \f$K\f$
 *  \param Rrel         rotation matrix from reference to sensor view
 *  \param trel         translation vector from reference to sensor view
 *  \param invK         inverse camera calibration matrix \f$K^{-1}\f$
 *  \param width        width of given arrays
 *  \param height       height of given arrays
 *  \param blocks       kernel grid dimensions
 *  \param threads      single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_transform_coordinates(float * d_x, float * d_y, float * d_X, float * d_Y, float * d_Z, const float * d_u,
                                const double K[3][3], const double Rrel[3][3], const double trel[3], const double invK[3][3],
                                const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Calculate derivatives of coordinates in camera reference frame
 *  
 *  \param d_dX         pointer to output derivative of \a x coordinate
 *  \param d_dY         pointer to output derivative of \a y coordinate
 *  \param d_dZ         pointer to output derivative of \a z coordinate
 *  \param invK         inverse camera calibration matrix \f$K^{-1}\f$
 *  \param Rrel         rotation matrix from reference to sensor view
 *  \param width        width of given arrays
 *  \param height       height of given arrays
 *  \param blocks       kernel grid dimensions
 *  \param threads      single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_calculate_coordinate_derivatives(float * d_dX, float * d_dY, float * d_dZ, const double invK[3][3], const double Rrel[3][3],
                                           const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Calculate gradient of \f$f(x, u)\f$
 *  
 *  \param d_dfx    pointer to output \a x component of \f$\nabla{f(x,u)}\f$
 *  \param d_dfy    pointer to output \a y component of \f$\nabla{f(x,u)}\f$
 *  \param d_X      pointer to input \a x coordinate in sensor view camera reference frame
 *  \param d_dX     pointer to input derivative of \a x coordinate in sensor view camera reference frame
 *  \param d_Y      pointer to input \a y coordinate in sensor view camera reference frame
 *  \param d_dY     pointer to input derivative of \a y coordinate in sensor view camera reference frame
 *  \param d_Z      pointer to input \a z coordinate in sensor view camera reference frame
 *  \param d_dZ     pointer to input derivative of \a z coordinate in sensor view camera reference frame
 *  \param fx       camera \a x focal length \f$f_x\f$
 *  \param fy       camera \a y focal length \f$f_y\f$
 *  \param width    width of given arrays
 *  \param height   height of given arrays
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_calculate_derivativeF(float * d_dfx, float * d_dfy, const float * d_X, const float * d_dX, const float * d_Y, const float * d_dY,
                                const float * d_Z, const float * d_dZ, const float fx, const float fy,
                                const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Calculate derivative image
 *  
 *  \param d_Iu     pointer to output derivative image \f$I_u\f$
 *  \param d_I      pointer to input intensity image \f$I\f$
 *  \param d_dfx    pointer to input \a x component of \f$\nabla{f(x, u_0)}\f$
 *  \param d_dfy    pointer to input \a y component of \f$\nabla{f(x, u_0)}\f$
 *  \param width    width of given arrays
 *  \param height   height of given arrays
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_calculate_Iu(float * d_Iu, const float * d_I, const float * d_dfx, const float * d_dfy,
                       const int width, const int height, dim3 blocks, dim3 threads);

 /**
 *  \brief Calculate anisotropic diffusion tensor \f$T\f$
 *  
 *  \param d_T11    pointer to output tensor values at \f$T(1,1)\f$
 *  \param d_T12    pointer to output tensor values at \f$T(1,2)\f$
 *  \param d_T21    pointer to output tensor values at \f$T(2,1)\f$
 *  \param d_T22    pointer to output tensor values at \f$T(2,2)\f$
 *  \param d_Img    pointer to input intensity image \f$I\f$
 *  \param beta     parameter \f$\beta\f$
 *  \param gamma    parameter \f$\gamma\f$
 *  \param width    width of given arrays
 *  \param height   height of given arrays
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details Tensor is calculated by \f$T = exp(-\beta |\nabla{I}|^{\gamma}) nn^T + n_{\perp}n^T_{\perp}\f$,
 * where \f$n = \left(\frac{\nabla{I}}{|\nabla{I}|} \right)\f$ and \f$n_{\perp} \perp n\f$
 */
void Anisotropic_diffusion_tensor(float * d_T11, float * d_T12, float * d_T21, float * d_T22, const float * d_Img,
                                  const float beta, const float gamma, const int width, const int height,
                                  dim3 blocks, dim3 threads);
							  
/**
 *  \brief Update primal variables \f$u\f$, \f$\overline{u}\f$, \f$u_1\f$ and \f$\overline{u}_1\f$ weighed by tensor \f$T\f$ using TGV2 algorithm
 *  
 *  \param d_u          pointer to primal variable \f$u\f$ to be updated
 *  \param d_u1x        pointer to component \a x of primal variable \f$u_1\f$ to be udpated
 *  \param d_u1y        pointer to component \a y of primal variable \f$u_1\f$ to be updated
 *  \param d_T11        pointer to input tensor values at \f$T(1,1)\f$
 *  \param d_T12        pointer to input tensor values at \f$T(1,2)\f$
 *  \param d_T21        pointer to input tensor values at \f$T(2,1)\f$
 *  \param d_T22        pointer to input tensor values at \f$T(2,2)\f$
 *  \param d_ubar       pointer to primal variable \f$\overline{u}\f$ to be updated
 *  \param d_u1xbar     pointer to component \a x of primal variable \f$\overline{u}_1\f$ to be updated
 *  \param d_u1ybar     pointer to component \a y of primal variable \f$\overline{u}_1\f$ to be updated
 *  \param d_Px         pointer to input component \a x of dual variable \f$p\f$
 *  \param d_Py         pointer to input component \a y of dual variable \f$p\f$
 *  \param d_Qx         pointer to input component \a x of dual variable \f$q\f$
 *  \param d_Qy         pointer to input component \a y of dual variable \f$q\f$
 *  \param d_Qz         pointer to input component \a z of dual variable \f$q\f$
 *  \param d_Qw         pointer to input component \a w of dual variable \f$q\f$
 *  \param d_prodsum    pointer to input sum of \f$I^i_u r^i\f$
 *  \param alpha0       TGV2 weight parameter \f$\alpha_0\f$
 *  \param alpha1       TGV2 weight parameter \f$\alpha_1\f$
 *  \param tau          TGV2 parameter \f$\tau\f$
 *  \param lambda       TGV2 parameter \f$\lambda\f$
 *  \param width        width of given arrays
 *  \param height       height of given arrays
 *  \param blocks       kernel grid dimensions
 *  \param threads      single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_updateU_tensor_weighed(float * d_u, float * d_u1x, float * d_u1y, const float * d_T11, const float * d_T12,
                                 const float * d_T21, const float * d_T22, float * d_ubar, float * d_u1xbar, float * d_u1ybar,
                                 const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                                 const float * d_Qz, const float * d_Qw, const float * d_prodsum,
                                 const float alpha0, const float alpha1, const float tau, const float lambda,
                                 const int width, const int height, dim3 blocks, dim3 threads);

/**
 *  \brief Update dual variable \f$p\f$ values weighed by tensor \f$T\f$ using TGV2 algorithm
 *  
 *  \param d_Px     pointer to component \a x of dual variable \f$p\f$ to be updated
 *  \param d_Py     pointer to component \a y of dual variable \f$p\f$ to be updated
 *  \param d_T11    pointer to input tensor values at \f$T(1,1)\f$
 *  \param d_T12    pointer to input tensor values at \f$T(1,2)\f$
 *  \param d_T21    pointer to input tensor values at \f$T(2,1)\f$
 *  \param d_T22    pointer to input tensor values at \f$T(2,2)\f$
 *  \param d_u      pointer to input primal variable \f$\overline{u}\f$
 *  \param d_u1x    pointer to input component \a x of primal variable \f$\overline{u}_1\f$
 *  \param d_u1y    pointer to input component \a y of primal variable \f$\overline{u}_1\f$
 *  \param alpha1   TGV2 weight parameter \f$\alpha_1\f$
 *  \param sigma    TGV2 parameter \f$\sigma\f$
 *  \param width    width of given arrays
 *  \param height   height of given arrays
 *  \param blocks   kernel grid dimensions
 *  \param threads  single block dimensions
 *  \return No return value
 *  
 *  \details
 */
void TGV2_updateP_tensor_weighed(float * d_Px, float * d_Py, const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                 const float * d_u, const float * d_u1x, const float * d_u1y, const float alpha1,
                                 const float sigma, const int width, const int height, dim3 blocks, dim3 threads);

 /** @} */ // group TGV2

#endif // KERNELS_CU_H
