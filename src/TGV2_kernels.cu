/* More info http://gpu4vision.icg.tugraz.at/papers/2012/graber_master.pdf#pub68
 * section 2.3.3 "Total Generalised Varation Multiview Stereo"
 *
 * Kernels list:
 * update p - TGV2_updateP_kernel()
 * update q - TGV2_updateQ_kernel()
 * update all u - TGV2_updateU_kernel()
 * update r - TGV2_updateR_kernel()
 * calculate It:
 *      transform coordinates at u0- TGV2_transform_coordinates_kernel()
 *      interpolate - bilinear_interpolation_kernel() in kernels.cu
 *      subtract - subtract_kernel()
 * calculate Iu:
 *      transform coordinates at u - TGV2_transform_coordinates_kernel()
 *      interpolate - bilinear_interpolation_kernel() in kernels.cu
 *      calculate coordinate derivatives at u0 - TGV2_calculate_coordinate_derivatives_kernel() (only needed once for each source view)
 *      calculate f(x,u) derivative - TGV2_calculate_derivativeF_kernel()
 *      calculate Iu - TGV2_calculate_Iu_kernel()
 *
 * p, q and u1 are initialised to 0
 * u requires an initial solution
 *
 * p and u1 are rank 2
 * q is rank 4
 * u and r are rank 1
 */

#include <kernels.cu.h>

__global__ void TGV2_updateP_kernel(float * __restrict__ d_Px, float * __restrict__ d_Py,
                                    const float * d_u, const float * __restrict__ d_u1x, const float * __restrict__ d_u1y,
                                    const float alpha1, const float sigma, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int xn = fminf(ind_x + 1, width - 1);
        int yn = fminf(ind_y + 1, height - 1);

        // p(n+1) = project(p(n) + sigma*alpha1*(grad(ubar(n)) - u1bar(n)))
        // where project(x) = x / max(1, |x|) and x is a vector
        double dx = d_Px[i] + alpha1 * sigma * (d_u[ind_y * width + xn] - d_u[i] - d_u1x[i]);
        double dy = d_Py[i] + alpha1 * sigma * (d_u[yn * width + ind_x] - d_u[i] - d_u1y[i]);
        double d = fmaxf(1.f, sqrt(dx * dx + dy * dy));
        d_Px[i] = dx / d;
        d_Py[i] = dy / d;
    }
}

__global__ void TGV2_updateQ_kernel(float * __restrict__ d_Qx, float * __restrict__ d_Qy,
                                    float * __restrict__ d_Qz, float * __restrict__ d_Qw,
                                    const float * d_u1x, const float * d_u1y,
                                    const float alpha0, const float sigma,
                                    const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int xn = fminf(ind_x + 1, width - 1);
        int yn = fminf(ind_y + 1, height - 1);

        // q(n+1) = project(q(n) + alpha0*sigma*grad(u1bar(n)))
        // where project(x) = x / max(1, |x|) and x is a vector
        float dx_u1x = d_u1x[ind_y * width + xn] - d_u1x[i];
        float dy_u1x = d_u1x[yn * width + ind_x] - d_u1x[i];
        float dx_u1y = d_u1y[ind_y * width + xn] - d_u1y[i];
        float dy_u1y = d_u1y[yn * width + ind_x] - d_u1y[i];
        double dx = d_Qx[i] + alpha0 * sigma * dx_u1x;
        double dy = d_Qy[i] + alpha0 * sigma * dy_u1y;
        double dz = d_Qz[i] + alpha0 * sigma * (dy_u1x + dx_u1y)/2.0f;
        double dw = d_Qw[i] + alpha0 * sigma * (dy_u1x + dx_u1y)/2.0f;
        double d = fmaxf(1.f, sqrt(dx * dx + dy * dy + dz * dz + dw * dw));
        d_Qx[i] = dx / d;
        d_Qy[i] = dy / d;
        d_Qz[i] = dz / d;
        d_Qw[i] = dw / d;
    }
}

__global__ void TGV2_updateR_kernel(float * __restrict__ d_r, float * __restrict__ d_prodsum,
                                    const float * __restrict__ d_u, const float * __restrict__ d_u0,
                                    const float * __restrict__ d_It, const float * __restrict__ d_Iu,
                                    const float sigma, const float lambda, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        // r(n+1) = project(r(n) + sigma*lambda*(It + (u-u0)*Iu))
        // where project(x) = x / max(1, |x|) and x is a vector
        d_r[i] = d_r[i] + sigma * lambda * (d_It[i] + (d_u[i] - d_u0[i]) * d_Iu[i]);
        d_r[i] = d_r[i] / fmaxf(1.f, fabs(d_r[i]));

        d_prodsum[i] += d_r[i] * d_Iu[i];
    }
}

__global__ void TGV2_updateU_kernel(float * __restrict__ d_u, float * __restrict__ d_u1x, float * __restrict__ d_u1y,
                                    float * __restrict__ d_ubar, float * __restrict__ d_u1xbar, float * __restrict__ d_u1ybar,
                                    const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                                    const float * d_Qz, const float * d_Qw,
                                    const float * __restrict__ d_prodsum,
                                    const float alpha0, const float alpha1, const float tau, const float lambda,
                                    const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int xp = fmaxf(ind_x - 1, 0.f);
        int yp = fmaxf(ind_y - 1, 0.f);;

        float uprev = d_u[i], u1xprev = d_u1x[i], u1yprev = d_u1y[i];

        // u(n+1) = u(n) - tau*(-alpha1*div(p(n+1)) + lambda*sum_over_i(Iui*ri(n+1)))
        // u1(n+1)= u1(n)- tau*(-alpha1*p(n+1) - alpha0*div(q(n+1)))
        d_u[i] = d_u[i] - tau*(-alpha1 * (d_Px[i] - d_Px[ind_y*width + xp] + d_Py[i] - d_Py[yp*width + ind_x]) + lambda*d_prodsum[i]);
        d_u1x[i] = d_u1x[i] - tau*(-alpha1*d_Px[i] - alpha0*(d_Qx[i] - d_Qx[ind_y*width + xp] + d_Qz[i] - d_Qz[yp*width + ind_x]));
        d_u1y[i] = d_u1y[i] - tau*(-alpha1*d_Py[i] - alpha0*(d_Qz[i] - d_Qz[ind_y*width + xp] + d_Qy[i] - d_Qy[yp*width + ind_x]));

        // ubar(n+1) = 2 * u(n+1) - u(n)
        // u1bar(n+1)= 2 * u1(n+1)- u1(n)
        d_ubar[i] = 2 * d_u[i] - uprev;
        d_u1xbar[i] = 2 * d_u1x[i] - u1xprev;
        d_u1ybar[i] = 2 * d_u1y[i] - u1yprev;
    }
}

// Kernel did not like arrays of known dimensions at compile time, thus each element had to be passed separately...
__global__ void TGV2_transform_coordinates_kernel(float * __restrict__ d_x, float * __restrict__ d_y,
                                                  float * __restrict__ d_X, float * __restrict__ d_Y, float * __restrict__ d_Z,
                                                  const float * __restrict__ d_u,
                                                  const double K11, const double K12, const double K13,
                                                  const double K21, const double K22, const double K23,
                                                  const double K31, const double K32, const double K33,
                                                  const double Rrel11, const double Rrel12, const double Rrel13,
                                                  const double Rrel21, const double Rrel22, const double Rrel23,
                                                  const double Rrel31, const double Rrel32, const double Rrel33,
                                                  const double trel1, const double trel2, const double trel3,
                                                  const double invK11, const double invK12, const double invK13,
                                                  const double invK21, const double invK22, const double invK23,
                                                  const double invK31, const double invK32, const double invK33,
                                                  const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        // 1 based indexing:
        int x = ind_x + 1;
        int y = ind_y + 1;

        // Calculate x1 = u * K^(-1) * x
        double x1 = d_u[i] * (invK11 * x + invK12 * y + invK13);
        double y1 = d_u[i] * (invK21 * x + invK22 * y + invK23);
        double z1 = d_u[i] * (invK31 * x + invK32 * y + invK33);

        // Calculate x2 = [R | t] * x1 = R * x1 + t
        double x2 = (Rrel11 * x1 + Rrel12 * y1 + Rrel13 * z1) + trel1;
        double y2 = (Rrel21 * x1 + Rrel22 * y1 + Rrel23 * z1) + trel2;
        double z2 = (Rrel31 * x1 + Rrel32 * y1 + Rrel33 * z1) + trel3;

        // Store 3D coordinates in the coordinate frame of the 2nd view
        d_X[i] = x2;
        d_Y[i] = y2;
        d_Z[i] = z2;

        // Calculate x1 = K * x2
        x1 = (K11 * x2 + K12 * y2 + K13 * z2);
        y1 = (K21 * x2 + K22 * y2 + K23 * z2);
        z1 = (K31 * x2 + K32 * y2 + K33 * z2);

        // Normalize z and revert to 0 based indexing
        d_x[i] = x1 / z1 - 1;
        d_y[i] = y1 / z1 - 1;
    }
}

__global__ void subtract_kernel(float * __restrict__ d_out, const float * __restrict__ d_in1, const float * __restrict__ d_in2,
                                const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        d_out[i] = d_in1[i] - d_in2[i];
    }
}

__global__ void TGV2_calculate_coordinate_derivatives_kernel(float * __restrict__ d_dX, float * __restrict__ d_dY,
                                                             float * __restrict__ d_dZ,
                                                             const double invK11, const double invK12, const double invK13,
                                                             const double invK21, const double invK22, const double invK23,
                                                             const double invK31, const double invK32, const double invK33,
                                                             const double Rrel11, const double Rrel12, const double Rrel13,
                                                             const double Rrel21, const double Rrel22, const double Rrel23,
                                                             const double Rrel31, const double Rrel32, const double Rrel33,
                                                             const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        // 1 based indexing:
        int x = ind_x + 1;
        int y = ind_y + 1;

        // Derivatives are given by grad(X) = Rrel * K^(-1) * x
        // Calculate x1 = K^(-1) * x
        double x1 = (invK11 * x + invK12 * y + invK13);
        double y1 = (invK21 * x + invK22 * y + invK23);
        double z1 = (invK31 * x + invK32 * y + invK33);

        // Calculate derivatives
        d_dX[i] = (Rrel11 * x1 + Rrel12 * y1 + Rrel13 * z1);
        d_dY[i] = (Rrel21 * x1 + Rrel22 * y1 + Rrel23 * z1);
        d_dZ[i] = (Rrel31 * x1 + Rrel32 * y1 + Rrel33 * z1);
    }
}

__global__ void TGV2_calculate_derivativeF_kernel(float * __restrict__ d_dfx, float * __restrict__ d_dfy,
                                                  const float * __restrict__ d_X, const float * __restrict__ d_dX,
                                                  const float * __restrict__ d_Y, const float * __restrict__ d_dY,
                                                  const float * __restrict__ d_Z, const float * __restrict__ d_dZ,
                                                  const float fx, const float fy, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        d_dfx[i] = fx * (d_dX[i] * d_Z[i] - d_X[i] * d_dZ[i]) / (d_Z[i] * d_Z[i]);
        d_dfy[i] = fy * (d_dY[i] * d_Z[i] - d_Y[i] * d_dZ[i]) / (d_Z[i] * d_Z[i]);
    }
}

__global__ void TGV2_calculate_Iu_kernel(float * __restrict__ d_Iu, const float * d_I,
                                         const float * __restrict__ d_dfx, const float * __restrict__ d_dfy,
                                         const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int xn = fminf(ind_x + 1, width -1);
        int yn = fminf(ind_y + 1, height - 1);

        double dx = d_I[ind_y*width + xn] - d_I[i];
        double dy = d_I[yn*width + ind_x] - d_I[i];
        d_Iu[i] = dx * d_dfx[i] + dy * d_dfy[i];
    }
}

__global__ void Anisotropic_diffusion_tensor_kernel(float * __restrict__ d_T11, float * __restrict__ d_T12,
                                                    float * __restrict__ d_T21, float * __restrict__ d_T22,
                                                    const float * __restrict__ d_Img,
                                                    const float beta, const float gamma, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;
        int xn = fminf(ind_x + 1, width -1);
        int yn = fminf(ind_y + 1, height - 1);

        // Calculate image gradient:
        float x = d_Img[ind_y*width+xn] - d_Img[i];
        float y = d_Img[yn*width+ind_x] - d_Img[i];

        // normalize
        float d = sqrt(x * x + y * y);
        float k;

        // Calculate tensor = exp(-beta*|grad(Img)|^gamma)n*trans(n) + m*trans(m),
        // where n is normalized image gradient vector and m is normal to n
        // n = [x y]' and m = [-y x]'
        if (d > 0.f) { // check for division by 0, this avoids QNAN values in tensor
            x = x / d;
            y = y / d;
            k = expf(- beta * powf(d, gamma));
            d_T11[i] = k * x * x + y * y;
            d_T12[i] = (k - 1) * x * y;
            d_T21[i] = (k - 1) * x * y;
            d_T22[i] = k * y * y + x * x;
        }
        else { // set to identity matrix
            d_T11[i] = 1.f;
            d_T12[i] = 0.f;
            d_T21[i] = 0.f;
            d_T22[i] = 1.f;
        }

    }
}

__global__ void TGV2_updateU_tensor_weighed_kernel(float * __restrict__ d_u, float * __restrict__ d_u1x, float * __restrict__ d_u1y,
                                                   const float * __restrict__ d_T11, const float * __restrict__ d_T12,
                                                   const float * __restrict__ d_T21, const float * __restrict__ d_T22,
                                                   float * __restrict__ d_ubar, float * __restrict__ d_u1xbar, float * __restrict__ d_u1ybar,
                                                   const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                                                   const float * d_Qz, const float * d_Qw,
                                                   const float * __restrict__ d_prodsum,
                                                   const float alpha0, const float alpha1, const float tau, const float lambda,
                                                   const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int xp = fmaxf(ind_x - 1, 0.f);
        int yp = fmaxf(ind_y - 1, 0.f);;

        float uprev = d_u[i], u1xprev = d_u1x[i], u1yprev = d_u1y[i];

        // u(n+1) = u(n) - tau*(-alpha1*div(tensor*p(n+1)) + lambda*sum_over_i(Iui*ri(n+1)))
        // u1(n+1)= u1(n)- tau*(-alpha1*tensor*p(n+1) - alpha0*div(q(n+1)))
        float c_px = d_Px[i], c_py = d_Py[i],
                xp_px = d_Px[ind_y*width+xp], yp_px = d_Px[yp*width+ind_x],
                xp_py = d_Py[ind_y*width+xp], yp_py = d_Py[yp*width+ind_x];

        d_u[i] = d_u[i] - tau*(-alpha1 * (d_T11[i] * (c_px - xp_px) + d_T12[i] * (c_py - xp_py) +
                                          d_T21[i] * (c_px - yp_px) + d_T22[i] * (c_py - yp_py)) + lambda*d_prodsum[i]);
        d_u1x[i] = d_u1x[i] - tau*(-alpha1*(d_T11[i]*c_px+d_T12[i]*c_py) - alpha0*(d_Qx[i] - d_Qx[ind_y*width + xp] + d_Qz[i] - d_Qz[yp*width + ind_x]));
        d_u1y[i] = d_u1y[i] - tau*(-alpha1*(d_T21[i]*c_px+d_T22[i]*c_py) - alpha0*(d_Qz[i] - d_Qz[ind_y*width + xp] + d_Qy[i] - d_Qy[yp*width + ind_x]));

        // ubar(n+1) = 2 * u(n+1) - u(n)
        // u1bar(n+1)= 2 * u1(n+1)- u1(n)
        d_ubar[i] = 2 * d_u[i] - uprev;
        d_u1xbar[i] = 2 * d_u1x[i] - u1xprev;
        d_u1ybar[i] = 2 * d_u1y[i] - u1yprev;
    }
}

__global__ void TGV2_updateP_tensor_weighed_kernel(float * __restrict__ d_Px, float * __restrict__ d_Py,
                                                   const float * __restrict__ d_T11, const float * __restrict__ d_T12,
                                                   const float * __restrict__ d_T21, const float * __restrict__ d_T22,
                                                   const float * d_u, const float * __restrict__ d_u1x, const float * __restrict__ d_u1y,
                                                   const float alpha1, const float sigma, const int width, const int height)
{
    const int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((ind_x < width) && (ind_y < height)) {
        const int i = ind_y * width + ind_x;

        int xn = fminf(ind_x + 1, width - 1);
        int yn = fminf(ind_y + 1, height - 1);

        // p(n+1) = project(p(n) + sigma*alpha1*(grad(ubar(n)) - u1bar(n)))
        // where project(x) = x / max(1, |x|) and x is a vector
        double x = d_u[ind_y * width + xn] - d_u[i] - d_u1x[i];
        double y = d_u[yn * width + ind_x] - d_u[i] - d_u1y[i];
        double dx = d_Px[i] + alpha1 * sigma * (d_T11[i] * x + d_T12[i] * y);
        double dy = d_Py[i] + alpha1 * sigma * (d_T21[i] * x + d_T22[i] * y);
        double d = fmaxf(1.f, sqrt(dx * dx + dy * dy));
        d_Px[i] = dx / d;
        d_Py[i] = dy / d;
    }
}

void TGV2_updateP(float * d_Px, float * d_Py, const float * d_u, const float * d_u1x, const float * d_u1y,
                  const float alpha1, const float sigma, const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_updateP_kernel<<<blocks, threads>>>(d_Px, d_Py, d_u, d_u1x, d_u1y, alpha1, sigma, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_updateQ(float * d_Qx, float * d_Qy, float * d_Qz, float * d_Qw, const float * d_u1x, const float * d_u1y,
                  const float alpha0, const float sigma, const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_updateQ_kernel<<<blocks, threads>>>(d_Qx, d_Qy, d_Qz, d_Qw, d_u1x, d_u1y, alpha0, sigma, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_updateR(float * d_r, float * d_prodsum, const float * d_u, const float * d_u0, const float * d_It, const float * d_Iu,
                  const float sigma, const float lambda, const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_updateR_kernel<<<blocks, threads>>>(d_r, d_prodsum, d_u, d_u0, d_It, d_Iu, sigma, lambda, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_updateU(float * d_u, float * d_u1x, float * d_u1y, float * d_ubar, float * d_u1xbar, float * d_u1ybar,
                  const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                  const float * d_Qz, const float * d_Qw, const float * d_prodsum, const float alpha0,
                  const float alpha1, const float tau, const float lambda,
                  const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_updateU_kernel<<<blocks, threads>>>(d_u, d_u1x, d_u1y, d_ubar, d_u1xbar, d_u1ybar, d_Px, d_Py, d_Qx, d_Qy,
                                             d_Qz, d_Qw, d_prodsum, alpha0, alpha1, tau, lambda, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_transform_coordinates(float * d_x, float * d_y, float * d_X, float * d_Y, float * d_Z, const float * d_u,
                                const double K[3][3], const double Rrel[3][3], const double trel[3], const double invK[3][3],
const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_transform_coordinates_kernel<<<blocks, threads>>>(d_x, d_y, d_X, d_Y, d_Z, d_u,
                                                           K[0][0], K[0][1], K[0][2], K[1][0], K[1][1], K[1][2], K[2][0], K[2][1], K[2][2],
            Rrel[0][0], Rrel[0][1], Rrel[0][2], Rrel[1][0], Rrel[1][1], Rrel[1][2], Rrel[2][0], Rrel[2][1], Rrel[2][2],
            trel[0], trel[1], trel[2],
            invK[0][0], invK[0][1], invK[0][2], invK[1][0], invK[1][1], invK[1][2], invK[2][0], invK[2][1], invK[2][2],
            width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void subtract(float * d_out, const float * d_in1, const float * d_in2, const int width, const int height, dim3 blocks, dim3 threads)
{
    subtract_kernel<<<blocks, threads>>>(d_out, d_in1, d_in2, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_calculate_coordinate_derivatives(float * d_dX, float * d_dY, float * d_dZ, const double invK[3][3], const double Rrel[3][3],
const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_calculate_coordinate_derivatives_kernel<<<blocks, threads>>>(d_dX, d_dY, d_dZ,
                                                                      invK[0][0], invK[0][1], invK[0][2], invK[1][0], invK[1][1], invK[1][2], invK[2][0], invK[2][1], invK[2][2],
            Rrel[0][0], Rrel[0][1], Rrel[0][2], Rrel[1][0], Rrel[1][1], Rrel[1][2], Rrel[2][0], Rrel[2][1], Rrel[2][2],
            width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_calculate_derivativeF(float * d_dfx, float * d_dfy, const float * d_X, const float * d_dX, const float * d_Y, const float * d_dY,
                                const float * d_Z, const float * d_dZ, const float fx, const float fy,
                                const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_calculate_derivativeF_kernel<<<blocks, threads>>>(d_dfx, d_dfy, d_X, d_dX, d_Y, d_dY, d_Z, d_dZ, fx, fy, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_calculate_Iu(float * d_Iu, const float * d_I, const float * d_dfx, const float * d_dfy,
                       const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_calculate_Iu_kernel<<<blocks, threads>>>(d_Iu, d_I, d_dfx, d_dfy, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void Anisotropic_diffusion_tensor(float * d_T11, float * d_T12, float * d_T21, float * d_T22, const float * d_Img,
                                  const float beta, const float gamma, const int width, const int height,
                                  dim3 blocks, dim3 threads)
{
    Anisotropic_diffusion_tensor_kernel<<<blocks, threads>>>(d_T11, d_T12, d_T21, d_T22, d_Img, beta, gamma, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_updateU_tensor_weighed(float * d_u, float * d_u1x, float * d_u1y, const float * d_T11, const float * d_T12,
                                 const float * d_T21, const float * d_T22, float * d_ubar, float * d_u1xbar, float * d_u1ybar,
                                 const float * d_Px, const float * d_Py, const float * d_Qx, const float * d_Qy,
                                 const float * d_Qz, const float * d_Qw, const float * d_prodsum,
                                 const float alpha0, const float alpha1, const float tau, const float lambda,
                                 const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_updateU_tensor_weighed_kernel<<<blocks, threads>>>(d_u, d_u1x, d_u1y, d_T11, d_T12, d_T21, d_T22, d_ubar, d_u1xbar, d_u1ybar,
                                                            d_Px, d_Py, d_Qx, d_Qy, d_Qz, d_Qw, d_prodsum, alpha0, alpha1, tau, lambda,
                                                            width, height);
    checkCudaErrors(cudaPeekAtLastError());
}

void TGV2_updateP_tensor_weighed(float * d_Px, float * d_Py, const float * d_T11, const float * d_T12, const float * d_T21, const float * d_T22,
                                 const float * d_u, const float * d_u1x, const float * d_u1y, const float alpha1,
                                 const float sigma, const int width, const int height, dim3 blocks, dim3 threads)
{
    TGV2_updateP_tensor_weighed_kernel<<<blocks, threads>>>(d_Px, d_Py, d_T11, d_T12, d_T21, d_T22, d_u, d_u1x, d_u1y, alpha1,
                                                            sigma, width, height);
    checkCudaErrors(cudaPeekAtLastError());
}
