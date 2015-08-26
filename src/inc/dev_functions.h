#ifndef DEVICE_FUNCTIONS_H
#define DEVICE_FUNCTIONS_H

#include "structs.h"
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

template<typename Tfrom, typename Tto>
__host__ __device__ inline
Tto convertDatatype(Tfrom d)
{
    return (Tto)d;
}

template<>
__host__ __device__ inline
uchar4 convertDatatype(unsigned char p)
{
    return make_uchar4(p,p,p,255);
}

template<>
__host__ __device__ inline
uchar3 convertDatatype(unsigned char p)
{
    return make_uchar3(p, p, p);
}

template<>
__host__ __device__ inline
unsigned char convertDatatype(uchar3 p)
{
    const unsigned sum = p.x + p.y + p.z;
    return sum / 3;
}

template<>
__host__ __device__ inline
unsigned char convertDatatype(uchar4 p)
{
    const unsigned sum = p.x + p.y + p.z + p.w;
    return sum / 4;
}

template<>
__host__ __device__ inline
uchar4 convertDatatype(uchar3 p)
{
    return make_uchar4(p.x,p.y,p.z,255);
}

template<>
__host__ __device__ inline
uchar3 convertDatatype(uint3 p)
{
    return make_uchar3(
        (unsigned char)(p.x),
        (unsigned char)(p.y),
        (unsigned char)(p.z)
        );
}

template<>
__host__ __device__ inline
uint3 convertDatatype(uchar3 p)
{
    return make_uint3(
        (unsigned int)(p.x),
        (unsigned int)(p.y),
        (unsigned int)(p.z)
        );
}

template<>
__host__ __device__ inline
uchar4 convertDatatype(uint4 p)
{
    return make_uchar4(
        (unsigned char)(p.x),
        (unsigned char)(p.y),
        (unsigned char)(p.z),
        (unsigned char)(p.w)
        );
}

template<>
__host__ __device__ inline
uint4 convertDatatype(uchar4 p)
{
    return make_uint4(
        (unsigned int)(p.x),
        (unsigned int)(p.y),
        (unsigned int)(p.z),
        (unsigned int)(p.w)
        );
}

template<>
__host__ __device__ inline
uchar4 convertDatatype(float4 p)
{
    return make_uchar4(
        (unsigned char)(p.x*255.0f),
        (unsigned char)(p.y*255.0f),
        (unsigned char)(p.z*255.0f),
        (unsigned char)(p.w*255.0f)
        );
}

template<>
__host__ __device__ inline
uchar3 convertDatatype(uchar4 p)
{
    return make_uchar3(p.x,p.y,p.z);
}

template<>
__host__ __device__ inline
float4 convertDatatype(float p)
{
    return make_float4(p,p,p,1.0f);
}

template<>
__host__ __device__ inline
float3 convertDatatype(uchar3 p)
{
    return make_float3(p.x, p.y, p.z);
}

template<>
__host__ __device__ inline
float convertDatatype(uchar3 p)
{
    return (p.x+p.y+p.z) / (3.0f*255.0f);
}

template<>
__host__ __device__ inline
float4 convertDatatype(uchar4 p)
{
    return make_float4(p.x, p.y, p.z, p.w) / 255.f;
}

template<>
__host__ __device__ inline
float4 convertDatatype(uchar3 p)
{
    return make_float4(make_float3(p.x, p.y, p.z)/255.f,1.0);
}

template<>
__host__ __device__ inline
float3 convertDatatype(float p)
{
    return make_float3(p);
}

template<>
__host__ __device__ inline
float convertDatatype(float3 p)
{
    return (p.x + p.y + p.z) / 3.0f;
}

/** @} */ // group general

#endif // DEVICE_FUNCTIONS_H
