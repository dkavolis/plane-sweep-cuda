#ifndef FUSION_H
#define FUSION_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "memory.h"
#include "structs.h"

template<unsigned char _histBins, bool _onDevice = true>
class fusionData : public MemoryManagement<fusionvoxel<_histBins>, _onDevice>
{
public:
    __device__ __host__ inline
    fusionData() :
        _w(0), _h(0), _d(0), _pitch(0), _spitch(0), _vol()
    {
        binParams();
    }

    __device__ __host__ inline
    fusionData(size_t w, size_t h, size_t d) :
        _w(w), _h(h), _d(d), _vol()
    {
        binParams();
        Malloc(_voxel, _w, _h, _d, _pitch, _spitch);
    }

    __device__ __host__ inline
    fusionData(size_t w, size_t h, size_t d, float3 x, float3 y) :
        _w(w), _h(h), _d(d), _vol(Rectangle(x,y))
    {
        binParams();
        Malloc(_voxel, _w, _h, _d, _pitch, _spitch);
    }

    __device__ __host__ inline
    fusionData(size_t w, size_t h, size_t d, Rectangle& vol) :
        _w(w), _h(h), _d(d), _vol(vol)
    {
        binParams();
        Malloc(_voxel, _w, _h, _d, _pitch, _spitch);
    }

    __device__ __host__ inline
    ~fusionData()
    {
        CleanUp(_voxel);
    }

    // Getters:
    __device__ __host__ inline
    size_t width(){ return _w; }

    __device__ __host__ inline
    size_t height(){ return _h; }

    __device__ __host__ inline
    size_t depth(){ return _d; }

    __device__ __host__ inline
    size_t pitch(){ return _pitch; }

    __device__ __host__ inline
    size_t slicePitch(){ return _spitch; }

    __device__ __host__ inline
    unsigned char bins(){ return _histBins; }

    __device__ __host__ inline
    Rectangle volume(){ return _vol; }

    __device__ __host__ inline
    size_t elements(){ return _w * _h * _d; }

    __device__ __host__ inline
    size_t sizeBytes(){ return _spitch * _d; }

    __device__ __host__ inline
    double sizeKBytes(){ return sizeBytes() / 1024.f; }

    __device__ __host__ inline
    double sizeMBytes(){ return sizeKBytes() / 1024.f; }

    __device__ __host__ inline
    double sizeGBytes(){ return sizeMBytes() / 1024.f; }

    __device__ __host__ inline
    float3 worldCoords(int x, int y, int z)
    {
        return _vol.a + _vol.size() * make_float3((x + .5) / _w, (y + .5) / _h, (z + .5) / _d);
    }

    // Setters:
    __device__ __host__ inline
    void volume(Rectangle &vol){ _vol = vol; }

    __device__ __host__ inline
    void volume(float3 x, float3 y){ _vol = Rectangle(x, y); }

    // Access to elements:
    __device__ __host__ inline
    float& u(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].u;
    }

    __device__ __host__ inline
    const float& u(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].u;
    }

    __device__ __host__ inline
    float& v(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].v;
    }

    __device__ __host__ inline
    const float& v(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].v;
    }

    __device__ __host__ inline
    float3& p(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].p;
    }

    __device__ __host__ inline
    const float3& p(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].p;
    }

    __device__ __host__ inline
    histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].h;
    }

    __device__ __host__ inline
    const histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].h;
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> * voxelPtr(int nx = 0, int ny = 0, int nz = 0)
    {
        return &_voxel[nx+ny*_w+nz*_w*_h];
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> * voxelPtr(int nx = 0, int ny = 0, int nz = 0) const
    {
        return &_voxel[nx+ny*_w+nz*_w*_h];
    }

    // Get bin parameters:
    __device__ __host__ inline
    double binCenter(unsigned char binindex)
    {
        if (binindex < _histBins) return _bincenters[binindex];
        else return 0.f;
    }

    __device__ __host__ inline
    double binStep(){ return _binstep; }

    // Difference functions:
    __host__ __device__ inline
    float3 gradUFwd(uint x, uint y, uint z){
        float u = this->u(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->u(x+1, y, z) - u;
        if (y < _h - 1) result.y = this->u(x, y+1, z) - u;
        if (z < _d - 1) result.z = this->u(x, y, z+1) - u;
        return result;
    }

    __host__ __device__ inline
    float3 gradVFwd(uint x, uint y, uint z){
        float v = this->v(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->v(x+1, y, z) - v;
        if (y < _h - 1) result.y = this->v(x, y+1, z) - v;
        if (z < _d - 1) result.z = this->v(x, y, z+1) - v;
        return result;
    }

    __host__ __device__ inline
    float divPBwd(int x, int y, int z)
    {
        float3 p = this->p(x, y, z);
        float result = p.x + p.y + p.z;
        if (x > 0) result -= this->p(x-1, y, z).x;
        if (y > 0) result -= this->p(x, y-1, z).y;
        if (z > 0) result -= this->p(x, y, z-1).z;
        return result;
    }

    // Prox Hist calculation functions:
    __host__ __device__ inline
    int Wi(unsigned char i, int x, int y, int z)
    {
        int r = 0;
        for (unsigned char j = 1; j <= i; j++) r -= this->h(x, y, z)(j);
        for (unsigned char j = i + 1; j <= _histBins; j++) r += this->h(x, y, z)(j);
        return r;
    }

    __host__ __device__ inline
    float pi(double u, unsigned char i, int x, int y, int z, double tau, double lambda)
    {
        return u + tau * lambda * Wi(i, x, y, z);
    }

    __host__ __device__ inline
    float proxHist(double u, int x, int y, int z, double tau, double lambda)
    {
        sortedHist<_histBins> prox(_bincenters);
        prox.insert(u); // insert p0
        for (unsigned char j = 1; j <= _histBins; j++) prox.insert(pi(u, j, x, y, z, tau, lambda));
        return prox.median();
    }

    // Prox function for p:
    __host__ __device__ inline
    float3 projectUnitBall(float3 x)
    {
        return x / fmaxf(1.f, sqrt(x.x * x.x + x.y * x.y + x.z * x.z));
    }

    // Histogram update function
    __host__ __device__ inline
    void updateHist(int x, int y, int z, float voxdepth, float depth, float threshold)
    {
        float sd = voxdepth - depth;
        if (_histBins == 2) threshold = 0.f;

        // check if empty
        if (sd >= threshold)
        {
            this->h(x, y, z).last()++;
            return;
        }

        // check if occluded
        if (sd <= - threshold)
        {
            this->h(x, y, z).first()++;
            return;
        }

        // close to surface
        this->h(x, y, z)(roundf((sd + threshold) / (2.f * threshold) * (_histBins - 3)))++;
    }

protected:
    fusionvoxel<_histBins> * _voxel;
    double _bincenters[_histBins], _binstep;
    size_t  _w, _h, _d, _pitch, _spitch;
    Rectangle _vol;

    __host__ __device__ inline
    void binParams()
    {
        // index = 0 bin is reserved for occluded voxel (signed distance < -1)
        // index = _histBins - 1 is reserced for empty voxel (signed distance > 1)
        // other bins store signed distance values in the range (-1; 1)
        _bincenters[0] = -1.f;
        _bincenters[_histBins - 1] = 1.f;
        for (unsigned char i = 1; i < _histBins - 1; i++) _bincenters[i] = 2.f * float(i - 1) / float(_histBins - 3) - 1.f;
        _binstep = 2 / float(_histBins - 3);
    }
};

// typedefs for fusionData with memory on GPU
typedef fusionData<2> dfusionData2;
typedef fusionData<3> dfusionData3;
typedef fusionData<4> dfusionData4;
typedef fusionData<5> dfusionData5;
typedef fusionData<6> dfusionData6;
typedef fusionData<7> dfusionData7;
typedef fusionData<8> dfusionData8;
typedef fusionData<9> dfusionData9;
typedef fusionData<10> dfusionData10;

// typedefs for fusionData with memory on CPU
typedef fusionData<2, false> fusionData2;
typedef fusionData<3, false> fusionData3;
typedef fusionData<4, false> fusionData4;
typedef fusionData<5, false> fusionData5;
typedef fusionData<6, false> fusionData6;
typedef fusionData<7, false> fusionData7;
typedef fusionData<8, false> fusionData8;
typedef fusionData<9, false> fusionData9;
typedef fusionData<10, false> fusionData10;

#endif // FUSION_H
