#ifndef TYPE_CONVERT_H
#define TYPE_CONVERT_H

#include <vector_functions.h>

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
    return make_float3(p,p,p);
}

template<>
__host__ __device__ inline
float convertDatatype(float3 p)
{
    return (p.x + p.y + p.z) / 3.0f;
}

#endif // TYPE_CONVERT_H
