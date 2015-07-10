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
