#ifndef STRUCTS_H
#define STRUCTS_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_math.h>

typedef unsigned int uint;

// Simple structure to hold histogram data of each voxel
// bin index 0 refers to occluded voxel (signed distance -1)
// bin index _nBins-1 refers to empty voxel (signed distance 1)
// bin signed distance can be calculated as (2 * index / (_nBins - 1) - 1)
template<unsigned char _nBins>
struct histogram
{
    unsigned char bin[_nBins];

    __device__ __host__ inline
    histogram()
    {
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = 0;
    }

    __device__ __host__ inline
    histogram<_nBins>& operator=(histogram<_nBins> & hist)
    {
        if (this == &hist) return *this;
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = hist.bin[i];
        return *this;
    }

    __device__ __host__ inline
    unsigned char& first(){ return bin[0]; }

    __device__ __host__ inline
    unsigned char& last(){ return bin[_nBins - 1]; }

    __device__ __host__ inline
    unsigned char& operator()(unsigned char i){ return bin[i]; }
};

// Simple structure to hold all data of a single voxel
//required for depthmap fusion
template<unsigned char _nBins>
struct fusionvoxel
{
    float u;
    float v;
    float3 p;
    histogram<_nBins> h;

    __host__ __device__ inline
    fusionvoxel() :
        u(0), p(make_float3(0, 0, 0)), h(), v(0)
    {}

    __host__ __device__ inline
    fusionvoxel(const double u) :
        u(u), p(make_float3(0, 0, 0)), h(), v(0)
    {}

    __host__ __device__ inline
    fusionvoxel(const fusionvoxel<_nBins>& f) :
        u(f.u), p(f.p), h(f.h), v(f.v)
    {}

    __host__ __device__ inline
    fusionvoxel<_nBins>& operator=(fusionvoxel<_nBins> & vox)
    {
        if (this == &vox) return *this;
        u = vox.u;
        p = vox.p;
        h = vox.h;
        v = vox.v;
        return *this;
    }
};

// Structure for calculating the prox histogram
// constructor with array input assumes values are sorted from least to greatest
template<unsigned char _nBins>
struct sortedHist
{
    double element[2 * _nBins + 1];
    unsigned char elements;

    __host__ __device__ inline
    sortedHist() : elements(0)
    {}

    __host__ __device__ inline
    sortedHist(double bincenter[_nBins]) : elements(_nBins)
    {
        for (unsigned char i = 0; i < _nBins; i++) element[i] = bincenter[i];
    }

    __host__ __device__ inline
    void insert(double val)
    {
        unsigned char next;
        if (elements != 0)
        {
            for (char i = elements - 1; i >= 0; i--){
                next = fmaxf(i + 1, 2 * _nBins + 1);
                if (val < element[i]) element[next] = element[i];
                else {
                    element[next] = val;
                    break;
                }
            }
        }
        else element[0] = val;
        elements++;
    }

    __host__ __device__ inline
    double median(){ return element[_nBins + 1]; }
};

// Simple struct to hold coordinates of volume rectangle
struct Rectangle
{
    float3 a, b;

    __host__ __device__ inline
    Rectangle() :
        a(make_float3(0,0,0)), b(make_float3(0,0,0))
    {}

    __host__ __device__ inline
    Rectangle(float3 x, float3 y) :
        a(x), b(y)
    {}

    __host__ __device__ inline
    Rectangle(const Rectangle& r) :
        a(r.a), b(r.b)
    {}

    __host__ __device__ inline
    float3 size()
    {
        return (a - b);
    }

    __host__ __device__ inline
    Rectangle& operator=(Rectangle & r)
    {
        if (this == &r) return *this;
        a = r.a;
        b = r.b;
        return *this;
    }
};

// Rectangle operator overloads
__host__ __device__ inline
Rectangle operator*(Rectangle r, float b)
{
    return Rectangle(r.a * b, r.b * b);
}

__host__ __device__ inline
Rectangle operator*(float b, Rectangle r)
{
    return Rectangle(r.a * b, r.b * b);
}

__host__ __device__ inline
void operator*=(Rectangle &r, float b)
{
    r.a *= b; r.b *= b;
}

__host__ __device__ inline
Rectangle operator/(Rectangle r, float b)
{
    return Rectangle(r.a / b, r.b / b);
}

__host__ __device__ inline
Rectangle operator/(float b, Rectangle r)
{
    return Rectangle(b / r.a, b / r.b);
}

__host__ __device__ inline
void operator/=(Rectangle &r, float b)
{
    r.a /= b; r.b /= b;
}

#endif // STRUCTS_H
