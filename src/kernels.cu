#include <kernels.cu.h>

__global__ void bilinear_interpolation_kernel_GPU(float * __restrict__ d_result, const float * __restrict__ d_data,
                                                  const float * __restrict__ d_xout, const float * __restrict__ d_yout,
                                                  const int M1, const int M2, const int N1, const int N2)
{
   const int l = threadIdx.x + blockDim.x * blockIdx.x;
   const int k = threadIdx.y + blockDim.y * blockIdx.y;

   if ((l<N1)&&(k<N2)) {

       float result_temp1, result_temp2;

       const int    ind_x = floor(d_xout[k*N1+l]);
       const float  a     = d_xout[k*N1+l]-ind_x;

       const int    ind_y = floor(d_yout[k*N1+l]);
       const float  b     = d_yout[k*N1+l]-ind_y;

       float d00, d01, d10, d11;
       if ((ind_x < 0) || (ind_y < 0)) { d_result[k*N1+l] = 0.f; return; }
       if (((ind_x)   < M1)&&((ind_y)   < M2))  d00 = d_data[ind_y*M1+ind_x];       else    { d_result[k*N1+l] = 0.f; return; }
       if (((ind_x+1) < M1)&&((ind_y)   < M2))  d10 = d_data[ind_y*M1+ind_x+1];     else    { d_result[k*N1+l] = 0.f; return; }
       if (((ind_x)   < M1)&&((ind_y+1) < M2))  d01 = d_data[(ind_y+1)*M1+ind_x];   else    { d_result[k*N1+l] = 0.f; return; }
       if (((ind_x+1) < M1)&&((ind_y+1) < M2))  d11 = d_data[(ind_y+1)*M1+ind_x+1]; else    { d_result[k*N1+l] = 0.f; return; }

       result_temp1 = a * d10 + (-d00 * a + d00);

       result_temp2 = a * d11 + (-d01 * a + d01);

       d_result[k*N1+l] = b * result_temp2 + (-result_temp1 * b + result_temp1);

   }
}

__global__ void affine_transform_indexes_kernel(float * __restrict__ d_x, float * __restrict__ d_y,
                                                const float h11, const float h12, const float h13,
                                                const float h21, const float h22, const float h23,
                                                const int width, const int height)
{
    const int l = threadIdx.x + blockDim.x * blockIdx.x;
    const int k = threadIdx.y + blockDim.y * blockIdx.y;

    if ((l < width) && (k < height)) {
        d_x[width * k + l] = h11 * (l + 1) + h12 * (k + 1) + h13 - 1;
        d_y[height * k + l] = h21 * (l + 1) + h22 * (k + 1) + h23 - 1;
    }
}

__global__ void calcNCC_kernel(float * __restrict__ d_ncc, const float * __restrict d_prod_mean,
                               const float * __restrict__ d_mean1, const float * __restrict__ d_mean2,
                               const float * __restrict__ d_std1, const float * __restrict__ d_std2,
                               const float stdthresh1, const float stdthresh2,
                               const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        if ((d_std1[ind] < stdthresh1) || (d_std2[ind] < stdthresh2)) d_ncc[ind] = 0.f;
        else {
            d_ncc[ind] = (d_prod_mean[ind] - d_mean1[ind] * d_mean2[ind]) / (d_std1[ind] * d_std2[ind]);
        }
    }
}

__global__ void update_arrays_kernel(float * __restrict__ d_depthmap, float * __restrict__ d_bestncc,
                                     const float * __restrict__ d_currentncc, const float current_depth,
                                     const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        if (d_currentncc[ind] > d_bestncc[ind]){
            d_bestncc[ind] = d_currentncc[ind];
            d_depthmap[ind] = current_depth;
        }
    }
}

__global__ void sum_depthmap_NCC_kernel(float * __restrict__ d_depthmap_out, float * __restrict__ d_count,
                                        const float * __restrict__ d_depthmap, const float * __restrict__ d_ncc,
                                        const float nccthreshold,
                                        const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        if (d_ncc[ind] > nccthreshold){
            d_depthmap_out[ind] += d_depthmap[ind];
            d_count[ind]++;
        }
    }
}

__global__ void calculate_STD_kernel(float * __restrict__ d_std, const float * __restrict__ d_mean,
                                     const float * __restrict__ d_mean_of_squares,
                                     const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        float var = d_mean_of_squares[ind] - d_mean[ind] * d_mean[ind];

        if (var > 0) d_std[ind] = sqrt(var);
        else d_std[ind] = 0.f;
    }
}

__global__ void set_value_kernel(float * __restrict__ d_output, const float value, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        d_output[ind] = value;
    }
}

__global__ void element_multiply_kernel(float * __restrict__ d_output, const float * __restrict__ d_input1,
                                   const float * __restrict__ d_input2,
                                   const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        d_output[ind] = d_input1[ind] * d_input2[ind];
    }
}

__global__ void element_rdivide_kernel(float * __restrict__ d_output, const float * __restrict__ d_input1,
                                  const float * __restrict__ d_input2,
                                  const int width, const int height, const float QNaN)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        if (d_input2[ind] != 0) d_output[ind] = d_input1[ind] / d_input2[ind];
        else d_output[ind] = QNaN;
    }
}

__global__ void convert_float_to_uchar_kernel(unsigned char * __restrict__ d_output, const float * __restrict__ d_input,
                                         const float min, const float max,
                                         const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        if (max == min) d_output[ind] = (unsigned char)(UCHAR_MAX / 2);
        else {
            if (min > max){
                if (d_input[ind] > min) d_output[ind] = UCHAR_MAX;
                else if (d_input[ind] < max) d_output[ind] = NULL;
                else if (d_input[ind] == d_input[ind]) d_output[ind] = (unsigned char)(UCHAR_MAX * (d_input[ind] - max) / (min - max));
                else d_output[ind] = UCHAR_MAX;
            }
            else {
                if (d_input[ind] > max) d_output[ind] = UCHAR_MAX;
                else if (d_input[ind] < min) d_output[ind] = NULL;
                else if (d_input[ind] == d_input[ind]) d_output[ind] = (unsigned char)(UCHAR_MAX * (d_input[ind] - min) / (max - min));
                else d_output[ind] = UCHAR_MAX;
            }
        }
    }
}

void affine_transform_indexes(float * d_x, float *  d_y,
                              const float h11, const float h12, const float h13,
                              const float h21, const float h22, const float h23,
                              const int width, const int height, dim3 blocks, dim3 threads)
{
    affine_transform_indexes_kernel<<<blocks, threads>>>(d_x, d_y,
                                                         h11, h12, h13,
                                                         h21, h22, h23,
                                                         width, height);
    checkCudaErrors(cudaPeekAtLastError() );
}

void bilinear_interpolation(float * d_result, const float * d_data,
                            const float * d_xout, const float * d_yout,
                            const int M1, const int M2, const int N1, const int N2,
                            dim3 blocks, dim3 threads)
{
    bilinear_interpolation_kernel_GPU<<<blocks, threads>>>(d_result, d_data, d_xout, d_yout, M1, M2, N1, N2);
    checkCudaErrors(cudaPeekAtLastError() );
}

void calcNCC(float * __restrict__ d_ncc, const float * __restrict__ d_prod_mean,
             const float * __restrict__ d_mean1, const float * __restrict__ d_mean2,
             const float * __restrict__ d_std1, const float * __restrict__ d_std2,
             const float stdthresh1, const float stdthresh2,
             const int width, const int height,
             dim3 blocks, dim3 threads)
{
    calcNCC_kernel<<<blocks, threads>>>(d_ncc, d_prod_mean,
                                        d_mean1, d_mean2,
                                        d_std1, d_std2,
                                        stdthresh1, stdthresh2,
                                        width, height);
    checkCudaErrors(cudaPeekAtLastError() );
}

void update_arrays(float * d_depthmap, float * d_bestncc,
                   const float * d_currentncc, const float current_depth,
                   const int width, const int height,
                   dim3 blocks, dim3 threads)
{
    update_arrays_kernel<<<blocks, threads>>>(d_depthmap, d_bestncc,
                                              d_currentncc, current_depth,
                                              width, height);
    checkCudaErrors(cudaPeekAtLastError() );
}

void sum_depthmap_NCC(float * d_depthmap_out, float * d_count,
                      const float * d_depthmap, const float * d_ncc,
                      const float nccthreshold,
                      const int width, const int height,
                      dim3 blocks, dim3 threads)
{
    sum_depthmap_NCC_kernel<<<blocks, threads>>>(d_depthmap_out, d_count,
                                                 d_depthmap, d_ncc,
                                                 nccthreshold,
                                                 width, height);
    checkCudaErrors(cudaPeekAtLastError() );
}

void calculate_STD(float * d_std, const float * d_mean,
                   const float * d_mean_of_squares,
                   const int width, const int height,
                   dim3 blocks, dim3 threads)
{
    calculate_STD_kernel<<<blocks, threads>>>(d_std, d_mean,
                                              d_mean_of_squares,
                                              width, height);
    checkCudaErrors(cudaPeekAtLastError() );
}

void set_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads)
{
    set_value_kernel<<<blocks, threads>>>(d_output, value, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void element_multiply(float * d_output, const float * d_input1,
                      const float * d_input2,
                      const int width, const int height,
                      dim3 blocks, dim3 threads)
{
    element_multiply_kernel<<<blocks, threads>>>(d_output, d_input1, d_input2,
                                                 width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void element_rdivide(float * d_output, const float * d_input1,
                     const float * d_input2,
                     const int width, const int height,
                     dim3 blocks, dim3 threads)
{
    const float QNan = std::numeric_limits<float>::quiet_NaN();
    element_rdivide_kernel<<<blocks, threads>>>(d_output, d_input1, d_input2, width, height, QNan);
    checkCudaErrors(cudaPeekAtLastError());
}

void convert_float_to_uchar(unsigned char * d_output, const float * d_input,
                            const float min, const float max,
                            const int width, const int height,
                            dim3 blocks, dim3 threads)
{
    convert_float_to_uchar_kernel<<<blocks, threads>>>(d_output, d_input, min, max, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}
