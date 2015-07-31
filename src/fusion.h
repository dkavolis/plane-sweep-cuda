#ifndef FUSION_H
#define FUSION_H

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

    __device__ __host__ histogram()
    {
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = 0;
    }

    __device__ __host__ histogram<_nBins>& operator=(histogram<_nBins> & hist)
    {
        if (this == &hist) return *this;
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = hist.bin[i];
        return *this;
    }

    __device__ __host__ unsigned char& first(){ return bin[0]; }

    __device__ __host__ unsigned char& last(){ return bin[_nBins - 1]; }

    __device__ __host__ unsigned char& operator()(unsigned char i){ return bin[i]; }
};

// Simple structure to hold all data of a single voxel 
//required for depthmap fusion
template<unsigned char _nBins>
struct fusionvoxel
{
    double u;
    double v;
    float3 p;
    histogram<_nBins> h;
    
    __host__ __device__ fusionvoxel() :
        u(0), p(make_float3(0, 0, 0)), h(), v(0)
    {}
        
    __host__ __device__ fusionvoxel(const double u) :
        u(u), p(make_float3(0, 0, 0)), h(), v(0)
    {}

    __host__ __device__ fusionvoxel(const fusionvoxel<_nBins>& f) :
        u(f.u), p(f.p), h(f.h), v(f.v)
    {}
        
    __host__ __device__ fusionvoxel<_nBins>& operator=(fusionvoxel<_nBins> & vox)
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

    __host__ __device__ sortedHist() : elements(0)
    {}

    __host__ __device__ sortedHist(double bincenter[_nBins]) : elements(_nBins)
    {
        for (unsigned char i = 0; i < _nBins; i++) element[i] = bincenter[i];
    }

    __host__ __device__ void insert(double val)
    {
        unsigned char next;
        if (elements != 0)
        {
            for (unsigned char i = elements - 1; i >= 0; i--){
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

    __host__ __device__ double median(){ return element[_nBins + 1]; }
};

// Simple struct to hold coordinates of volume rectangle
struct Rectangle
{
    float3 a, b;

    __host__ __device__ Rectangle() :
        a(make_float3(0,0,0)), b(make_float3(0,0,0))
    {}

    __host__ __device__ Rectangle(float3 x, float3 y) :
        a(x), b(y)
    {}

    __host__ __device__ Rectangle(const Rectangle& r) :
        a(r.a), b(r.b)
    {}

    __host__ __device__ float3 size()
    {
        return fabs(a - b);
    }

    __host__ __device__ Rectangle& operator=(Rectangle & r)
    {
        if (this == &r) return *this;
        a = r.a;
        b = r.b;
        return *this;
    }
};

// Rectangle operator overloads
__host__ __device__ Rectangle operator*(Rectangle r, float b)
{
    return Rectangle(r.a * b, r.b * b);
}

__host__ __device__ Rectangle operator*(float b, Rectangle r)
{
    return Rectangle(r.a * b, r.b * b);
}

__host__ __device__ void operator*=(Rectangle &r, float b)
{
    r.a *= b; r.b *= b;
}

__host__ __device__ Rectangle operator/(Rectangle r, float b)
{
    return Rectangle(r.a / b, r.b / b);
}

__host__ __device__ Rectangle operator/(float b, Rectangle r)
{
    return Rectangle(b / r.a, b / r.b);
}

__host__ __device__ void operator/=(Rectangle &r, float b)
{
    r.a /= b; r.b /= b;
}

template<unsigned char _histBins, bool _onDevice = true>
class fusionData
{
public:
    __device__ __host__ fusionData() :
        _w(0), _h(0), _d(0), _pitch(0), _spitch(0), _vol()
    {
        binParams();
    }

    __device__ __host__ fusionData(size_t w, size_t h, size_t d) :
        _w(w), _h(h), _d(d), _vol()
    {
        binParams();
        Malloc<fusionvoxel<_histBins>>(_voxel, _pitch, _spitch);
    }

    __device__ __host__ fusionData(size_t w, size_t h, size_t d, float3 x, float3 y) :
        _w(w), _h(h), _d(d), _vol(Rectangle(x,y))
    {
        binParams();
        Malloc<fusionvoxel<_histBins>>(_voxel, _pitch, _spitch);
    }

    __device__ __host__ fusionData(size_t w, size_t h, size_t d, Rectangle& vol) :
        _w(w), _h(h), _d(d), _vol(vol)
    {
        binParams();
        Malloc<fusionvoxel<_histBins>>(_voxel, _pitch, _spitch);
    }

    __device__ __host__ ~fusionData()
    {
        CleanUp<fusionvoxel<_histBins>>(_voxel);
    }

    // Getters:
    __device__ __host__ size_t width(){ return _w; }
    __device__ __host__ size_t height(){ return _h; }
    __device__ __host__ size_t depth(){ return _d; }
    __device__ __host__ size_t pitch(){ return _pitch; }
    __device__ __host__ size_t slicePitch(){ return _spitch; }
    __device__ __host__ unsigned char bins(){ return _histBins; }
    __device__ __host__ Rectangle volume(){ return _vol; }

    // Setters:
    __device__ __host__ void volume(Rectangle &vol){ _vol = vol; }
    __device__ __host__ void volume(float3 x, float3 y){ _vol = Rectangle(x, y); }

    // Access to elements:
    __device__ __host__ double& u(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].u;
    }

    __device__ __host__ const double& u(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].u;
    }

    __device__ __host__ double& v(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].v;
    }

    __device__ __host__ const double& v(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].v;
    }

    __device__ __host__ float3& p(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].p;
    }

    __device__ __host__ const float3& p(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].p;
    }

    __device__ __host__ histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0)
    {
        return _voxel[nx+ny*_w+nz*_w*_h].h;
    }

    __device__ __host__ const histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0) const
    {
        return _voxel[nx+ny*_w+nz*_w*_h].h;
    }

    __device__ __host__ fusionvoxel<_histBins> * voxelPtr(int nx = 0, int ny = 0, int nz = 0)
    {
        return &_voxel[nx+ny*_w+nz*_w*_h];
    }

    __device__ __host__ const fusionvoxel<_histBins> * voxelPtr(int nx = 0, int ny = 0, int nz = 0) const
    {
        return &_voxel[nx+ny*_w+nz*_w*_h];
    }

    // Get bin parameters:
    __device__ __host__ double binCenter(unsigned char binindex)
    {
        if (binindex < _histBins) return _bincenters[binindex];
        else return 0.f;
    }

    __device__ __host__ double binStep(){ return _binstep; }

    // Difference functions:
    __host__ __device__ float3 gradUFwd(uint x, uint y, uint z){
        float u = this->u(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->u(x+1, y, z) - u;
        if (y < _h - 1) result.y = this->u(x, y+1, z) - u;
        if (z < _d - 1) result.z = this->u(x, y, z+1) - u;
        return result;
    }

    __host__ __device__ float3 gradVFwd(uint x, uint y, uint z){
        float v = this->v(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->v(x+1, y, z) - v;
        if (y < _h - 1) result.y = this->v(x, y+1, z) - v;
        if (z < _d - 1) result.z = this->v(x, y, z+1) - v;
        return result;
    }

    __host__ __device__ float divPBwd(int x, int y, int z)
    {
        float3 p = this->p(x, y, z);
        float result = p.x + p.y + p.z;
        if (x > 0) result -= this->p(x-1, y, z).x;
        if (y > 0) result -= this->p(x, y-1, z).y;
        if (z > 0) result -= this->p(x, y, z-1).z;
        return result;
    }

    // Prox Hist calculation functions:
    __host__ __device__ int Wi(unsigned char i, int x, int y, int z)
    {
        int r = 0;
        for (unsigned char j = 1; j <= i; j++) r -= this->h(x, y, z)(j);
        for (unsigned char j = i + 1; j <= _histBins; j++) r += this->h(x, y, z)(j);
        return r;
    }

    __host__ __device__ double pi(double u, unsigned char i, int x, int y, int z, double tau, double lambda)
    {
        return u + tau * lambda * Wi(i, x, y, z);
    }

    __host__ __device__ double proxHist(double u, int x, int y, int z, double tau, double lambda)
    {
        sortedHist<_histBins> prox(_bincenters);
        prox.insert(u); // insert p0
        for (unsigned char j = 1; j <= _histBins; j++) prox.insert(pi(u, j, x, y, z, tau, lambda));
        return prox.median();
    }

    // Prox function for p:
    __host__ __device__ float3 projectUnitBall(float3 x)
    {
        return x / fmaxf(1.f, sqrt(x.x * x.x + x.y * x.y + x.z * x.z));
    }

    // Histogram update function
    __host__ __device__ void updateHist(int x, int y, int z, float voxdepth, float depth, float threshold)
    {
        float sd = voxdepth - depth;

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

    template<typename T>
    __host__ __device__ void CleanUp(T * ptr)
    { 
        if (_onDevice) cudaFree(ptr); 
        else cudaFreeHost(ptr);
    }

    template<typename T>
    __host__ __device__ void Malloc(T * &ptr, size_t &pitch, size_t &spitch)
    {
        if (_onDevice){
            cudaMallocPitch(&ptr, &pitch, _w * sizeof(T), _h * _d);
            spitch = _h * pitch;
        }
        else {
            pitch = _w * sizeof(T);
            spitch = _h * pitch;
            cudaMallocHost(&ptr, pitch * _h * _d);
        }
    }

    __host__ __device__ void binParams()
    {
        // index = 0 bin is reserved for occluded voxel (signed distance < -1)
        // index = _histBins - 1 is reserced for empty voxel (signed distance > 1)
        // other bins store signed distance values in the range (-1; 1)
        _bincenters[0] = -1.f;
        _bincenters[_histBins - 1] = 1.f;
        for (unsigned char i = 1; i < _histBins - 1; i++) _bincenters[i] = 2 * float(i - 1) / float(_histBins - 3) - 1.f;
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
