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
       if (((ind_x)   < M1)&&((ind_y)   < M2))  d00 = d_data[ind_y*M1+ind_x];       else    d00 = 0.f;
       if (((ind_x+1) < M1)&&((ind_y)   < M2))  d10 = d_data[ind_y*M1+ind_x+1];     else    d10 = 0.f;
       if (((ind_x)   < M1)&&((ind_y+1) < M2))  d01 = d_data[(ind_y+1)*M1+ind_x];   else    d01 = 0.f;
       if (((ind_x+1) < M1)&&((ind_y+1) < M2))  d11 = d_data[(ind_y+1)*M1+ind_x+1]; else    d11 = 0.f;

       result_temp1 = a * d10 + (-d00 * a + d00);

       result_temp2 = a * d11 + (-d01 * a + d01);

       d_result[k*N1+l] = b * result_temp2 + (-result_temp1 * b + result_temp1);

   }
}

__global__ void CreateGrid2D_kernel(float * __restrict__ d_x, float * __restrict__ d_y, const int rows, const int columns)
{
    const int l = threadIdx.x + blockDim.x * blockIdx.x;
    const int k = threadIdx.y + blockDim.y * blockIdx.y;

    if ((l<columns)&&(k<rows)) {
        d_x[columns * k + l] = float(l);
        d_y[columns * k + 1] = float(k);
    }
}

__global__ void calcNCC_kernel(float * __restrict__ d_ncc, float * __restrict d_prod_mean,
                               float * __restrict__ d_mean1, float * __restrict__ d_mean2,
                               float * __restrict__ d_std1, float * __restrict__ d_std2,
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

void CreateGrid2D(float * d_x, float *  d_y, const int columns, const int rows, dim3 blocks, dim3 threads)
{
    CreateGrid2D_kernel<<<blocks, threads>>>(d_x, d_y, rows, columns);
}

void bilinear_interpolation(float * d_result, const float * d_data,
                            const float * d_xout, const float * d_yout,
                            const int M1, const int M2, const int N1, const int N2,
                            dim3 blocks, dim3 threads)
{
    bilinear_interpolation_kernel_GPU<<<blocks, threads>>>(d_result, d_data, d_xout, d_yout, M1, M2, N1, N2);
}

void calcNCC(float * __restrict__ d_ncc, float * __restrict__ d_prod_mean,
             float * __restrict__ d_mean1, float * __restrict__ d_mean2,
             float * __restrict__ d_std1, float * __restrict__ d_std2,
             const float stdthresh1, const float stdthresh2,
             const int width, const int height,
             dim3 blocks, dim3 threads)
{
    calcNCC_kernel<<<blocks, threads>>>(d_ncc, d_prod_mean,
                                        d_mean1, d_mean2,
                                        d_std1, d_std2,
                                        stdthresh1, stdthresh2,
                                        width, height);
}
