#ifndef IMAGE_H
#define IMAGE_H

#include "memory.h"
#include "type_convert.h"

template<typename T, MemoryKind memT = Device>
struct Image : MemoryManagement<T, memT>
{
public:
    inline __device__ __host__
    virtual ~Image()
    {
        free();
    }

    inline __host__ __device__
    Image( const Image<T,memT>& img )
        : pitch_(img.pitch_), ptr_(img.ptr_), w_(img.w_), h_(img.h_), managed_(false)
    {}

    inline __host__
    Image()
        : pitch_(0), ptr_(0), w_(0), h_(0), managed_(false)
    {}

    inline __host__
    Image(size_t w, size_t h)
        :w_(w), h_(h), managed_(true)
    {
       Malloc(ptr_, w_, h_, pitch_);
    }

    inline __device__ __host__
    Image(T* ptr)
        : pitch_(0), ptr_(ptr), w_(0), h_(0), managed_(false)
    {}

    inline __device__ __host__
    Image(T* ptr, size_t w)
        : pitch_(sizeof(T)*w), ptr_(ptr), w_(w), h_(0), managed_(false)
    {}

    inline __device__ __host__
    Image(T* ptr, size_t w, size_t h)
        : pitch_(sizeof(T)*w), ptr_(ptr), w_(w), h_(h), managed_(false)
    {}

    inline __device__ __host__
    Image(T* ptr, size_t w, size_t h, size_t pitch)
        : pitch_(pitch), ptr_(ptr), w_(w), h_(h), managed_(false)
    {}

    inline __device__ __host__
    size_t width() const
    {
        return w_;
    }

    inline __device__ __host__
    size_t height() const
    {
        return h_;
    }

    inline __device__ __host__
    size_t area() const
    {
        return w_*h_;
    }

    inline __device__ __host__
    size_t pitch() const
    {
        return pitch_;
    }

    inline __device__ __host__
    bool managed() const
    {
        return managed_;
    }

    inline __device__ __host__
    void setManaged(bool managed)
    {
        managed_ = managed;
    }

    template<MemoryKind memFrom>
    inline __host__
    void copyFrom(const Image<T,memFrom>& img)
    {
        ASSERT_AUTO(((w_ == img.width()) && (h_ == img.height())));
        bool dto = (memT == Device) || (memT == Managed);
        bool hto = !dto;
        bool dfrom = (memFrom == Device) || (memFrom == Managed);
        bool hfrom = !dfrom;
        if (dto && dfrom) { Device2DeviceCopy(ptr_, pitch_, img.data(), img.pitch(), w_, h_); return; }
        if (dto && hfrom) { Host2DeviceCopy(ptr_, pitch_, img.data(), img.pitch(), w_, h_); return; }
        if (hto && dfrom) { Device2HostCopy(ptr_, pitch_, img.data(), img.pitch(), w_, h_); return; }
        if (hto && hfrom) { Host2HostCopy(ptr_, pitch_, img.data(), img.pitch(), w_, h_); return; }
    }

    template<MemoryKind memTo>
    inline __host__
    void copyTo(Image<T,memTo> & img) const
    {
        ASSERT_AUTO(((w_ == img.width()) && (h_ == img.height())));
        bool dto = (memTo == Device) || (memTo == Managed);
        bool hto = !dto;
        bool dfrom = (memT == Device) || (memT == Managed);
        bool hfrom = !dfrom;
        if (dto && dfrom) { Device2DeviceCopy(img.data(), img.pitch(), ptr_, pitch_, w_, h_); return; }
        if (dto && hfrom) { Host2DeviceCopy(img.data(), img.pitch(), ptr_, pitch_, w_, h_); return; }
        if (hto && dfrom) { Device2HostCopy(img.data(), img.pitch(), ptr_, pitch_, w_, h_); return; }
        if (hto && hfrom) { Host2HostCopy(img.data(), img.pitch(), ptr_, pitch_, w_, h_); return; }
    }

    inline __host__
    void copyFrom(const T* ptr, size_t pitch)
    {
        if ((memT == Device) || (memT == Managed)) return Host2DeviceCopy(ptr_, pitch_, ptr, pitch, w_, h_);
        else return Host2HostCopy(ptr_, pitch_, ptr, pitch, w_, h_);
    }

    inline __host__
    void copyTo(T* ptr, size_t pitch) const
    {
        if ((memT == Device) || (memT == Managed)) return Device2HostCopy(ptr, pitch, ptr_, pitch_, w_, h_);
        else return Host2HostCopy(ptr, pitch, ptr_, pitch_, w_, h_);
    }

    inline __host__
    void reset(size_t w, size_t h)
    {
        w_ = w;
        h_ = h;
        if (managed_) CleanUp(ptr_);
        Malloc(ptr_, w_, h_, pitch_);
        managed_ = true;
    }

    inline __host__
    void free()
    {
        if (managed_) CleanUp(ptr_);
        managed_ = false;
        ptr_ = 0;
        w_ = 0;
        h_ = 0;
        pitch_ = 0;
    }

    inline __device__ __host__
    bool isValid() const
    {
        return ptr_ != 0;
    }

    inline  __device__ __host__
    T* rowPtr(size_t y = 0)
    {
        return (T*)((unsigned char*)(ptr_) + y*pitch_);
    }

    inline  __device__ __host__
    const T* rowPtr(size_t y = 0) const
    {
        return (T*)((unsigned char*)(ptr_) + y*pitch_);
    }

    inline __device__ __host__
    T* data()
    {
        return ptr_;
    }

    inline __device__ __host__
    const T* data() const
    {
        return ptr_;
    }

    inline  __device__ __host__
    T& operator()(size_t x, size_t y)
    {
        return rowPtr(y)[x];
    }

    inline  __device__ __host__
    const T& operator()(size_t x, size_t y) const
    {
        return rowPtr(y)[x];
    }

    inline  __device__ __host__
    T& operator[](size_t ix)
    {
        return ptr_[ix];
    }

    inline  __device__ __host__
    const T& operator[](size_t ix) const
    {
        return ptr_[ix];
    }

    inline  __device__ __host__
    const T& get(int x = 0, int y = 0) const
    {
        return rowPtr(y)[x];
    }

    inline __device__ __host__
    T& get(int x = 0, int y = 0)
    {
        return rowPtr(y)[y];
    }

    inline  __device__ __host__
    bool inBounds(float x, float y, float border = 0) const
    {
        return border <= x && x < (w_-border) && border <= y && y < (h_-border);
    }

    inline  __device__ __host__
    const T& getWithClampedRange(int x, int y) const
    {
        x = clamp(x, 0, (int)w_-1);
        y = clamp(y, 0, (int)h_-1);
        return rowPtr(y)[x];
    }

    inline  __device__ __host__
    const T& getConditionNeumann(int x, int y) const
    {
        x = abs(x);
        if(x >= w_) x = (w_-1)-(x-w_);

        y = abs(y);
        if(y >= h_) y = (h_-1)-(y-h_);

        return rowPtr(y)[x];
    }

    template<typename TR>
    inline __device__ __host__
    TR getBilinear(float u, float v, TR ooboundsretval = TR()) const
    {
        const float ix = floorf(u);
        const float iy = floorf(v);
        const float fx = u - ix;
        const float fy = v - iy;

        if ((ix < 0) || (iy < 0) || (iy+1 > h_-1) || (ix+1 > w_-1)) return ooboundsretval;

        const T* bl = rowPtr(iy)  + (size_t)ix;
        const T* tl = rowPtr(iy+1)+ (size_t)ix;

        return lerp(
            lerp( bl[0], bl[1], fx ),
            lerp( tl[0], tl[1], fx ),
            fy
        );
    }

    inline __device__ __host__
    T getNearestNeighbour(float u, float v) const
    {
        return get(u+0.5, v+0.5);
    }

    template<typename TR>
    inline __device__ __host__
    TR getBackwardDiffDx(int x, int y) const
    {
        const T* row = rowPtr(y);
        return ( convertDatatype<TR,T>(row[x]) - convertDatatype<TR,T>(row[x-1]) );
    }

    template<typename TR>
    inline __device__ __host__
    TR getBackwardDiffDy(int x, int y) const
    {
        return ( convertDatatype<TR,T>(get(x,y)) - convertDatatype<TR,T>(get(x,y-1)) );
    }

    template<typename TR>
    inline __device__ __host__
    TR getCentralDiffDx(int x, int y) const
    {
        const T* row = rowPtr(y);
        return ( convertDatatype<TR,T>(row[x+1]) - convertDatatype<TR,T>(row[x-1]) ) / 2;
    }

    template<typename TR>
    inline __device__ __host__
    TR getCentralDiffDy(int x, int y) const
    {
        return ( convertDatatype<TR,T>(get(x,y+1)) - convertDatatype<TR,T>(get(x,y-1)) ) / 2;
    }

    inline __device__ __host__
    T getNearestNeighbour(const float2& p) const
    {
        return getNearestNeighbour(p.x,p.y);
    }

    inline __device__ __host__
    T getNearestNeighbour(const double2& p) const
    {
        return getNearestNeighbour(p.x,p.y);
    }

    template<typename TR>
    inline __device__ __host__
    TR getBilinear(const float2& p, TR ooboundsretval = TR()) const
    {
        return getBilinear<TR>(p.x, p.y, ooboundsretval);
    }

    template<typename TR>
    inline __device__ __host__
    TR getBilinear(const double2& p, TR ooboundsretval = TR()) const
    {
        return getBilinear<TR>(p.x, p.y, ooboundsretval);
    }

    inline __device__ __host__
    T getBilinear(const float2& p, T ooboundsretval = T()) const
    {
        return getBilinear<T>(p.x, p.y, ooboundsretval);
    }

    inline __device__ __host__
    T getBilinear(const double2& p, T ooboundsretval = T()) const
    {
        return getBilinear<T>(p.x, p.y, ooboundsretval);
    }

    inline  __device__ __host__
    bool inBounds(const float2& p, float border = 0) const
    {
        return inBounds(p.x, p.y, border);
    }

    inline  __device__ __host__
    bool inBounds(const double2& p, float border = 0) const
    {
        return inBounds(p.x, p.y, border);
    }

    inline __host__
    Image<T,memT> operator=(const Image<T, Host>& img)
    {
        if (this == &img) return *this;
        // check if image buffers are the same size, if not reallocate this buffer
        if ((w_ != img.width()) || (h_ != img.height())){
            reset(img.width(), img.height());
        }
        copyFrom<Host>(img);
        return *this;
    }

    inline __host__
    Image<T,memT> operator=(const Image<T, Standard>& img)
    {
        if (this == &img) return *this;
        // check if image buffers are the same size, if not reallocate this buffer
        if ((w_ != img.width()) || (h_ != img.height())){
            reset(img.width(), img.height());
        }
        copyFrom<Standard>(img);
        return *this;
    }

    inline __host__
    Image<T,memT> operator=(const Image<T, Device>& img)
    {
        if (this == &img) return *this;
        // check if image buffers are the same size, if not reallocate this buffer
        if ((w_ != img.width()) || (h_ != img.height())){
            reset(img.width(), img.height());
        }
        copyFrom<Device>(img);
        return *this;
    }

    inline __host__
    Image<T,memT> operator=(const Image<T, Managed>& img)
    {
        if (this == &img) return *this;
        // check if image buffers are the same size, if not reallocate this buffer
        if ((w_ != img.width()) || (h_ != img.height())){
            reset(img.width(), img.height());
        }
        copyFrom<Managed>(img);
        return *this;
    }

private:
    size_t pitch_;
    T* ptr_;
    size_t w_;
    size_t h_;
    bool managed_;
};

#endif IMAGE_H
