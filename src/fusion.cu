// Kernels for depthmap fusion (WIP):
#include "fusion.cu.h"

template<unsigned char _bins>
__global__ void FusionUpdateHistogram_kernel(fusionData<_bins> * f, const float * __restrict__ depthmap, const Matrix3D & K,
                                             const Matrix3D & R, const Vector3D & t, const float threshold, const int width, const int height)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < f->width()) && (y < f->height()) && (z < f->depth()))
    {
        // Get world coordinates of the voxel
        float3 c = f->worldCoords(x, y, z);

        // Transform world coordinates to camera coordinates
        c = R * c + t;

        // Transform camera coordinates to homogeneous pixel coordinates
        c = K * c; // c.z - voxel depth in camera coordinates
        float2 px = make_float2(c / c.z);

        // Check if pixel coordinates fall inside image range
        if ((px.x < 0) || (px.x > width-1) || (px.y < 0) || (px.y > height-1)) return;

        // Get int pixel coords
        int2 pxc = make_int2(maxf(floorf(px), 0));
        int2 pxc1 = make_int2(minf(pxc.x+1, width-1), minf(pxc.y+1, height-1));

        // Get fractions
        float2 frac = fracf(px);

        // Read image values for bilinterp
        float2 y0 = make_float2(depthmap[pxc.x+pxc.y*width], depthmap[pxc1.x+pxc.y*width]); // values at (x,y) and (x+1,y)
        float2 y1 = make_float2(depthmap[pxc.x+pxc1.y*width], depthmap[pxc1.x+pxc1.y*width]); // values at (x,y+1) and (x+1,y+1)

        // Interpolate voxel depth
        float depth = bilinterp(y0, y1, frac);

        // Update histogram
        f->updateHist(x, y, z, c.z, depth, threshold);
    }
}

template<unsigned char _bins>
__global__ void FusionUpdateU_kernel(fusionData<_bins> * f, const double tau, const double lambda)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
	
    if ((x < f->width()) && (y < f->height()) && (z < f->depth()))
    {
        const double un = f->u(x, y, z);
        const double u = un - tau * (- f->divPBwd(x, y, z));
        f->u(x, y, z) = f->proxHist(u, x, y, z, tau, lambda);
        f->v(x, y, z) = 2 * f->u(x, y, z) - un;
    }
}

template<unsigned char _bins>
__global__ void FusionUpdateP_kernel(fusionData<_bins> * f, const double sigma)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
	
    if ((x < f->width()) && (y < f->height()) && (z < f->depth()))
    {
        f->p(x, y, z) = f->projectUnitBall(f->p(x, y, z) + sigma * f->gradVFwd(x, y, z));
    }
}

template<unsigned char _bins>
void FusionUpdateU(fusionData<_bins> * f, const double tau, const double lambda, dim3 blocks, dim3 threads)
{
    FusionUpdateU_kernel<_bins><<<blocks, threads>>>(f, tau, lambda);
}

template<unsigned char _bins>
void FusionUpdateP(fusionData<_bins> * f, const double sigma, dim3 blocks, dim3 threads)
{
    FusionUpdateP_kernel<_bins><<<blocks, threads>>>(f, sigma);
}

template<unsigned char _bins>
void FusionUpdateHistogram(fusionData<_bins> * f, const float * depthmap, const Matrix3D & K, const Matrix3D & R,
                           const Vector3D &t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads)
{
    FusionUpdateHistogram_kernel<_bins><<<blocks, threads>>>(f, depthmap, K, R, t, threshold, width, height);
}

template<unsigned char _bins> inline
void FusionUpdateIteration(fusionData<_bins, Device> *f, const float * depthmap, const Matrix3D & K, const Matrix3D & R, const Vector3D & t,
                           const float threshold, const double tau, const double lambda, const double sigma,
                           const int width, const int height, dim3 blocks, dim3 threads)
{
    FusionUpdateHistogram_kernel<_bins><<<blocks, threads>>>(f, depthmap, K, R, t, threshold, width, height);
    FusionUpdateU_kernel<_bins><<<blocks, threads>>>(f, tau, lambda);
    FusionUpdateP_kernel<_bins><<<blocks, threads>>>(f, sigma);
}
