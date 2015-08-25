#include <kernels.cu.h>
#include <helper_structs.h>

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

        if ((ind_x < 0) || (ind_y < 0) || (ind_y+1 > M2-1) || (ind_x+1 > M1-1)) { d_result[k*N1+l] = 0.f; return; }

        result_temp1 = a * d_data[ind_y*M1+ind_x+1] + (1 - a) * d_data[ind_y*M1+ind_x];

        result_temp2 = a * d_data[(ind_y+1)*M1+ind_x+1] + (1 - a) * d_data[(ind_y+1)*M1+ind_x];

        d_result[k*N1+l] = b * result_temp2 + (1 - b) * result_temp1;
    }
}

__global__ void transform_indexes_kernel(float * __restrict__ d_x, float * __restrict__ d_y,
                                         const Matrix3D h,
                                         const int width, const int height)
{
    const int l = threadIdx.x + blockDim.x * blockIdx.x;
    const int k = threadIdx.y + blockDim.y * blockIdx.y;

    if ((l < width) && (k < height)) {
        float3 x = h * make_float3(l+1, k+1, 1);
        x = x / x.z - 1;
        d_x[width * k + l] = x.x;
        d_y[width * k + l] = x.y;
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

        // If either STD is below threshold, set NCC to 0
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

        // Update if better correspondance was found
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

        // Sum if NCC is above threshold
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

        // variance (easy but numerically unstable method)
        float var = d_mean_of_squares[ind] - d_mean[ind] * d_mean[ind];

        // check for negative variance
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
        d_output[ind_x+ind_y*width] = d_input1[ind_x+ind_y*width] * d_input2[ind_x+ind_y*width];
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

__global__ void windowed_mean_row_kernel(float * __restrict__ d_output, const float * __restrict__ d_input,
                                         const unsigned int winsize, const bool squared,
                                         const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        float mean = 0.f;
        int n = winsize / 2;
        int k;

        for (int i = -n; i <= n; i++){
            k = ind_x + i;
            if (k < 0) k = -k;
            if (k > width - 1) k = 2 * (width - 1) - k;
            if (squared) mean += d_input[ind_y * width + k] * d_input[ind_y * width + k];
            else mean += d_input[ind_y * width + k];
        }
        d_output[ind] = mean / (float)winsize;
    }
}

__global__ void windowed_mean_column_kernel(float * __restrict__ d_output, const float * __restrict__ d_input,
                                            const unsigned int winsize, const bool squared,
                                            const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        float mean = 0.f;
        int n = winsize / 2;
        int k;

        for (int i = -n; i <= n; i++){
            k = ind_y + i;
            if (k < 0) k = -k;
            if (k > height - 1) k = 2 * (height - 1) - k;
            if (squared) mean += d_input[k * width + ind_x] * d_input[k * width + ind_x];
            else mean += d_input[k * width + ind_x];
        }
        d_output[ind] = mean / (float)winsize;
    }
}

__global__ void convert_uchar_to_float_kernel(float * __restrict__ d_output, const unsigned char * __restrict__ d_input,
                                              const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;

        d_output[ind] = d_input[ind];
    }
}

__global__ void denoising_TVL1_calculateP_kernel(float * __restrict__ d_Px, float * __restrict__ d_Py,
                                                 const float * d_input, const float sigma,
                                                 const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        int ny = ind_y + 1;
        if (ny > height - 1) ny = height - 1;

        double dx, dy, m;

        if (ind_x == width - 1){ // last column
            dy = (d_input[ny * width + ind_x] - d_input[ind]) * sigma + d_Py[ind];
            m = 1.f / dy;
            if (m < 0.f) m = -m;
            if (m > 1.f) m = 1.f;
            d_Px[ind] = 0.f;
            d_Py[ind] = dy * m;
        }
        else {
            dx = (d_input[ind + 1] - d_input[ind]) * sigma + d_Px[ind];
            dy = (d_input[ny * width + ind_x] - d_input[ind]) * sigma + d_Py[ind];
            m = 1.f / sqrt(dx * dx + dy * dy);
            if (m > 1.f) m = 1.f;
            d_Px[ind] = dx * m;
            d_Py[ind] = dy * m;
        }
    }
}

__global__ void denoising_TVL1_calculateP_tensor_weighed_kernel(float * __restrict__ d_Px, float * __restrict__ d_Py,
                                                                const float * __restrict__ d_T11, const float * __restrict__ d_T12,
                                                                const float * __restrict__ d_T21, const float * __restrict__ d_T22,
                                                                const float * d_input, const float sigma,
                                                                const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {

        const int i = ind_y * width + ind_x;

        int xn = fminf(ind_x + 1, width - 1);
        int yn = fminf(ind_y + 1, height - 1);

        double x = d_input[ind_y * width + xn] - d_input[i];
        double y = d_input[yn * width + ind_x] - d_input[i];
        double dx = d_Px[i] + sigma * (d_T11[i] * x + d_T12[i] * y);
        double dy = d_Py[i] + sigma * (d_T21[i] * x + d_T22[i] * y);
        double d = fmaxf(1.f, sqrt(dx * dx + dy * dy));
        d_Px[i] = dx / d;
        d_Py[i] = dy / d;
    }
}

__global__ void element_scale_kernel(float * __restrict__ d_output, const float scale, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        d_output[ind] = d_output[ind] * scale;
    }
}

__global__ void element_add_kernel(float * __restrict__ d_output, const float value, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        d_output[ind] += value;
    }
}

__global__ void set_QNAN_value_kernel(float * __restrict__ d_output, const float value, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        if (d_output[ind] != d_output[ind]) d_output[ind] = value;
    }
}

__global__ void denoising_TVL1_update_kernel(float * __restrict__ d_output, float * __restrict__ d_R,
                                             const float * d_Px, const float * d_Py, const float * __restrict__ d_origin,
                                             const float tau, const float theta, const float lambda, const float sigma,
                                             const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int ind = ind_y * width + ind_x;
        double x_new;
        int yp = ind_y - 1;
        if (yp < 0) yp = 0;

        d_R[ind] += d_origin[ind];
        d_R[ind] += sigma * d_output[ind];
        if (d_R[ind] > lambda) d_R[ind] = lambda;
        if (d_R[ind] < -lambda) d_R[ind] = -lambda;

        if (ind_x == 0){
            x_new = d_output[ind] + tau*(d_Py[ind] - d_Py[yp * width + ind_x]) - tau * d_R[ind];
            d_output[ind] = x_new + theta*(x_new - d_output[ind]);
        }
        else {
            x_new = d_output[ind] + tau*(d_Px[ind] - d_Px[ind - 1] + d_Py[ind] - d_Py[yp * width + ind_x]) - tau * d_R[ind];
            d_output[ind] = x_new + theta*(x_new - d_output[ind]);
        }
    }
}

__global__ void denoising_TVL1_update_tensor_weighed_kernel(float * __restrict__ d_output, float * __restrict__ d_R,
                                                            const float * d_Px, const float * d_Py, const float * __restrict__ d_origin,
                                                            const float * __restrict__ d_T11, const float * __restrict__ d_T12,
                                                            const float * __restrict__ d_T21, const float * __restrict__ d_T22,
                                                            const float tau, const float theta, const float lambda, const float sigma,
                                                            const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {

        const int i = ind_y * width + ind_x;

        int xp = fmaxf(ind_x - 1, 0.f);
        int yp = fmaxf(ind_y - 1, 0.f);

        d_R[i] += d_origin[i];
        d_R[i] += sigma * d_output[i];
        if (d_R[i] > lambda) d_R[i] = lambda;
        if (d_R[i] < -lambda) d_R[i] = -lambda;

        float   c_px = d_Px[i], c_py = d_Py[i],
                xp_px = d_Px[ind_y*width+xp], yp_px = d_Px[yp*width+ind_x],
                xp_py = d_Py[ind_y*width+xp], yp_py = d_Py[yp*width+ind_x];

        double x_new = d_output[i] + tau*((d_T11[i] * (c_px - xp_px) + d_T12[i] * (c_py - xp_py) +
                                           d_T21[i] * (c_px - yp_px) + d_T22[i] * (c_py - yp_py)) - d_R[i]);

        d_output[i] = x_new + theta*(x_new - d_output[i]);
    }
}

__global__ void compute3D_kernel(float * __restrict__ d_x, float * __restrict__ d_y, float * __restrict__ d_z,
                                 const Matrix3D Rrel,
                                 const Vector3D trel,
                                 const Matrix3D invK,
                                 const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int x = ind_x + 1, y = ind_y + 1;
        float z = d_z[i];

        if (z < 5.5)
        {
            float3 x1 = Rrel * (z * invK * make_float3(x, y, 1)) + trel;
            d_x[i] = x1.x;
            d_y[i] = x1.y;
            d_z[i] = x1.z;
        }
        else d_z[i] = -9.f;
    }
}

void transform_indexes(float * d_x, float *  d_y,
                       const Matrix3D h,
                       const int width, const int height, dim3 blocks, dim3 threads)
{
    transform_indexes_kernel<<<blocks, threads>>>(d_x, d_y,
                                                  h,
                                                  width, height);
}

void bilinear_interpolation(float * d_result, const float * d_data,
                            const float * d_xout, const float * d_yout,
                            const int M1, const int M2, const int N1, const int N2,
                            dim3 blocks, dim3 threads)
{
    bilinear_interpolation_kernel_GPU<<<blocks, threads>>>(d_result, d_data, d_xout, d_yout, M1, M2, N1, N2);
}

void calcNCC(float * d_ncc, const float * d_prod_mean,
             const float * d_mean1, const float * d_mean2,
             const float * d_std1, const float * d_std2,
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

void update_arrays(float * d_depthmap, float * d_bestncc,
                   const float * d_currentncc, const float current_depth,
                   const int width, const int height,
                   dim3 blocks, dim3 threads)
{
    update_arrays_kernel<<<blocks, threads>>>(d_depthmap, d_bestncc,
                                              d_currentncc, current_depth,
                                              width, height);
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
}

void calculate_STD(float * d_std, const float * d_mean,
                   const float * d_mean_of_squares,
                   const int width, const int height,
                   dim3 blocks, dim3 threads)
{
    calculate_STD_kernel<<<blocks, threads>>>(d_std, d_mean,
                                              d_mean_of_squares,
                                              width, height);
}

void set_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads)
{
    set_value_kernel<<<blocks, threads>>>(d_output, value, width, height);
}

void element_multiply(float * d_output, const float * d_input1,
                      const float * d_input2,
                      const int width, const int height,
                      dim3 blocks, dim3 threads)
{
    element_multiply_kernel<<<blocks, threads>>>(d_output, d_input1, d_input2,
                                                 width, height);
}

void element_rdivide(float * d_output, const float * d_input1,
                     const float * d_input2,
                     const int width, const int height,
                     dim3 blocks, dim3 threads)
{
    const float QNan = std::numeric_limits<float>::quiet_NaN();
    element_rdivide_kernel<<<blocks, threads>>>(d_output, d_input1, d_input2, width, height, QNan);
}

void convert_float_to_uchar(unsigned char * d_output, const float * d_input,
                            const float min, const float max,
                            const int width, const int height,
                            dim3 blocks, dim3 threads)
{
    convert_float_to_uchar_kernel<<<blocks, threads>>>(d_output, d_input, min, max, width, height);
}

void windowed_mean_row(float * d_output, const float * d_input,
                       const unsigned int winsize, const bool squared,
                       const int width, const int height, dim3 blocks, dim3 threads)
{
    windowed_mean_row_kernel<<<blocks, threads>>>(d_output, d_input, winsize, squared,
                                                  width, height);
}

void windowed_mean_column(float * d_output, const float * d_input,
                          const unsigned int winsize, const bool squared,
                          const int width, const int height, dim3 blocks, dim3 threads)
{
    windowed_mean_column_kernel<<<blocks, threads>>>(d_output, d_input, winsize, squared,
                                                     width, height);
}

void convert_uchar_to_float(float * d_output, const unsigned char * d_input,
                            const int width, const int height, dim3 blocks, dim3 threads)
{
    convert_uchar_to_float_kernel<<<blocks, threads>>>(d_output, d_input, width, height);
}

void denoising_TVL1_calculateP(float * d_Px, float * d_Py,
                               const float * d_input, const float sigma,
                               const int width, const int height,
                               dim3 blocks, dim3 threads)
{
    denoising_TVL1_calculateP_kernel<<<blocks, threads>>>(d_Px, d_Py, d_input, sigma, width, height);
}

void denoising_TVL1_calculateP_tensor_weighed(float * d_Px, float * d_Py,
                                              const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                              const float * d_input, const float sigma,
                                              const int width, const int height,
                                              dim3 blocks, dim3 threads)
{
    denoising_TVL1_calculateP_tensor_weighed_kernel<<<blocks, threads>>>(d_Px, d_Py, d_T11, d_T12, d_T21, d_T22,
                                                                         d_input, sigma, width, height);
}

void element_scale(float * d_output, const float scale, const int width, const int height, dim3 blocks, dim3 threads)
{
    element_scale_kernel<<<blocks, threads>>>(d_output, scale, width, height);
}

void element_add(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads)
{
    element_add_kernel<<<blocks, threads>>>(d_output, value, width, height);
}

void set_QNAN_value(float * d_output, const float value, const int width, const int height, dim3 blocks, dim3 threads)
{
    set_QNAN_value_kernel<<<blocks, threads>>>(d_output, value, width, height);
}

void denoising_TVL1_update(float * d_output, float * d_R,
                           const float * d_Px, const float * d_Py, const float * d_origin,
                           const float tau, const float theta, const float lambda, const float sigma,
                           const int width, const int height, dim3 blocks, dim3 threads)
{
    denoising_TVL1_update_kernel<<<blocks, threads>>>(d_output, d_R, d_Px, d_Py, d_origin,
                                                      tau, theta, lambda, sigma, width, height);
}

void denoising_TVL1_update_tensor_weighed(float * d_output, float * d_R,
                                          const float * d_Px, const float * d_Py, const float * d_origin,
                                          const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                          const float tau, const float theta, const float lambda, const float sigma,
                                          const int width, const int height, dim3 blocks, dim3 threads)
{
    denoising_TVL1_update_tensor_weighed_kernel<<<blocks, threads>>>(d_output, d_R, d_Px, d_Py, d_origin, d_T11, d_T12, d_T21, d_T22,
                                                                     tau, theta, lambda, sigma, width, height);
}

void compute3D(float * d_x, float * d_y, float * d_z, const Matrix3D Rrel, const Vector3D trel,
               const Matrix3D invK, const int width, const int height, dim3 blocks, dim3 threads)
{
    compute3D_kernel<<<blocks, threads>>>(d_x, d_y, d_z,
                                          Rrel, trel, invK, width, height);
}
