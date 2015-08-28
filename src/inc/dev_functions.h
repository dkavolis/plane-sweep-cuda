#ifndef DEVICE_FUNCTIONS_H
#define DEVICE_FUNCTIONS_H

#include "structs.h"
#include "image.h"
#include <device_launch_parameters.h>

//////////////////////////////////////////////////////////////
// General device functions
//////////////////////////////////////////////////////////////
/** \addtogroup general
* @{
*/

__device__ inline
int getGlobalIdx(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

/**
 *  \brief Bilinear interpolation function
 *
 *  \param y0   values at (x,y) and (x+1,y) respectively
 *  \param y1   values at (x,y+1) and (x+1,y+1) respectively
 *  \param frac fractions in x and y directions respectively in range [0,1]
 *  \return Bilinear interpolation result
 */
__host__ __device__ inline
float bilinterp(float2 y0, float2 y1, float2 frac)
{
    float2 x = lerp(y0, y1, frac.y);
    return lerp(x.x, x.y, frac.x);
}

/**
 *  \name Quaternion to rotation
 *  \brief Quaternion Q to rotation matrix transformation
 *  \return 3x3 Rotation matrix
 *  @{
 *  \param qx   Q x component
 *  \param qy   Q y component
 *  \param qz   Q z component
 *  \param qw   Q w component
 */
__host__ __device__ inline
Matrix3D quat2rot(float qx, float qy, float qz, float qw)
{
    float n = qw * qw + qx * qx + qy * qy + qz * qz;
    float s;
    if (n == 0.f)
        s = 0.f;
    else
        s = 2.f / n;

    float wx = s * qw * qx;
    float wy = s * qw * qy;
    float wz = s * qw * qz;
    float xx = s * qx * qx;
    float xy = s * qx * qy;
    float xz = s * qx * qz;
    float yy = s * qy * qy;
    float yz = s * qy * qz;
    float zz = s * qz * qz;

    return Matrix3D(1 - (yy + zz), xy - wz, xz + wy,
                    xy + wz, 1 - (xx + zz), yz - wx,
                    xz - wy, yz + wx, 1 - (xx + yy));
}

/**
 *  \brief Overload of quat2rot(float qx, float qy, float qz, float qw)
 */
__host__ __device__ inline
Matrix3D quat2rot(float4 q)
{
    return quat2rot(q.x, q.y, q.z, q.w);
}

/** @} */ // \name Quaternion to rotation

template <typename T>
__host__ __device__ inline
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

__host__ __device__ inline
float2 gradFwd(const Image<float>& img, float val, int x, int y)
{

    float2 d = make_float2(0,0);
    if(x < img.width() - 1) d.x = img(x + 1, y) - val;
    if(y < img.height() - 1) d.y = img(x, y + 1) - val;
    return d;
}

__host__ __device__ inline
float divBwd(const Image<float2>& img, float2 val, int x, int y)
{
    float div = val.x + val.y;
    if(x > 0) div -= img(x - 1, y).x;
    if(y > 0) div -= img(x , y - 1).y;
    return div;
}

 __host__ __device__ inline
float2 divBwd(const Image<float4>& img, float4 val, int x, int y)
{
    float2 div = make_float2(val.x + val.z, val.z + val.y);

    if (0 < x){
        div.x -= img(x - 1, y).x;
        div.y -= img(x - 1, y).z;
    }

    if (0 < y){
        div.x -= img(x, y - 1).z;
        div.y -= img(x, y - 1).y;
    }

    return div;
}

__host__ __device__ inline
float4 epsilon(const Image<float2>& img, float2 val, int x, int y)
{
    float4 d = make_float4(0);

    if (x < img.width() - 1) {
        const float2 px = img(x+1,y);
        d.x = px.x - val.x;
        d.z = px.y - val.y;
    }

    if (y < img.height() - 1) {
        const float2 py = img(x,y+1);
        d.w = py.x - val.x;
        d.y = py.y - val.y;
    }

    return make_float4(d.x, d.y, (d.z+d.w)/2.0f, (d.z+d.w)/2.0f );
}

inline __host__ __device__
float project(float val)
{
    return val / fmaxf(1.0f, fabs(val));
}

inline __host__ __device__
float2 project(float2 val)
{
    return val / fmaxf(1.0f, length(val));
}

inline __host__ __device__
float3 project(float3 val)
{
    return val / fmaxf(1.0f, length(val));
}

inline __host__ __device__
float4 project(float4 val)
{
    return val / fmaxf(1.0f, length(val));
}

/** @} */ // group general

#endif // DEVICE_FUNCTIONS_H
