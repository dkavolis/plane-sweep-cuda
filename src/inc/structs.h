/**
 *  \file structs.h
 *  \brief Header file containing helper structs.
 * All structs inherit from \a Managed class operators \a new and \a delete
 * so that those structs can be passed to kernels by value or reference without
 * needing to call \a cudaMalloc and \a cudaMemcpy first.
 */
#ifndef STRUCTS_H
#define STRUCTS_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_math.h>
#include "memory.h"

// Forward declarations of structs contained in this file:
template<unsigned char _nBins>  struct histogram;
template<unsigned char _nBins>  struct Mhistogram;
template<unsigned char _nBins>  struct fusionvoxel;
template<unsigned char _nBins>  struct Mfusionvoxel;
template<unsigned char _nBins>  struct sortedHist;
template<unsigned char _nBins>  struct MsortedHist;
                                struct Rectangle3D;
                                struct MRectangle3D;
                                struct Matrix3D;
                                struct MMatrix3D;
                                struct Vector3D;
                                struct MVector3D;

/** \brief Convenience typedef for <em>unsigned int</em>*/
typedef unsigned int uint;

/**
*  \brief Simple structure to hold histogram data of each voxel
*  \tparam _nBins   number of histogram bins
*
*  \details Bin index \a 0 refers to occluded voxel (signed distance -1).
* Bin index <em>_nBins-1</em> refers to empty voxel (signed distance 1).
* Other bins hold votes for signed distances in range (-1,1).
* Bin signed distance can be calculated as \f$2 \frac{index}{nBins - 3} - 1\f$.
*/
template<unsigned char _nBins>
struct histogram
{
    /** \brief Array of bins */
    unsigned char bin[_nBins];

    /**
    *  \brief Default constructor
    *
    *  \details All bins are initialized to 0
    */
    __device__ __host__ inline
    histogram()
    {
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = 0;
    }

    /** \brief Copy constructor */
    __device__ __host__ inline
    histogram(const histogram<_nBins> & h)
    {
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = h.bin[i];
    }

    /**
    *  \brief Copy operator
    *
    *  \param hist histogram to be copied from
    *  \return Reference to this histogram
    */
    __device__ __host__ inline
    histogram<_nBins>& operator=(histogram<_nBins> & hist)
    {
        if (this == &hist) return *this;
        for (unsigned char i = 0; i < _nBins; i++) bin[i] = hist.bin[i];
        return *this;
    }

    /**
    *  \brief Access to first bin (occluded voxel)
    *
    *  \return Reference to first bin
    */
    __device__ __host__ inline
    unsigned char& first(){ return bin[0]; }

    /**
    *  \brief Access to last bin (empty voxel)
    *
    *  \return Reference to last bin
    */
    __device__ __host__ inline
    unsigned char& last(){ return bin[_nBins - 1]; }

    /**
    *  \brief Access operator
    *
    *  \param i index of bin
    *  \return Reference to bin at index \a i
    */
    __device__ __host__ inline
    unsigned char& operator()(unsigned char i){ return bin[i]; }
};

/** \brief histogram struct with overloaded \a new and \a delete operators from class \p Manage */
template<unsigned char _nBins>
struct Mhistogram : public histogram<_nBins>, public Manage
{
    __host__ __device__ inline
    Mhistogram(const histogram<_nBins> & h) : histogram<_nBins>(h)
    {}
};

// Simple structure to hold all data of a single voxel
//required for depthmap fusion
/**
*  \brief Simple structure to hold all data of a single voxel required for depthmap fusion
*  \tparam _nBins   number of histogram bins
*/
template<unsigned char _nBins>
struct fusionvoxel
{
    /** \brief Primal variable \f$u\f$ */
    float u;
    /** \brief Helper variable \f$v\f$ */
    float v;
    /** \brief Dual variable \f$p\f$ */
    float3 p;
    /** \brief Histogram */
    histogram<_nBins> h;

    /**
    * \brief Default constructor
    *
    *  \details All variables are intialized to 0.
    */
    __host__ __device__ inline
    fusionvoxel() :
        u(0), p(make_float3(0, 0, 0)), h(), v(0)
    {}

    /**
    *  \brief Constructor overload.
    *
    *  \param u Primal variable \f$u\f$ initialization value
    *
    *  \details Details \f$p\f$, \f$v\f$ and \a histogram are initialized to 0.
    */
    __host__ __device__ inline
    fusionvoxel(const double u) :
        u(u), p(make_float3(0, 0, 0)), h(), v(0)
    {}

    /** \brief Copy constructor */
    __host__ __device__ inline
    fusionvoxel(const fusionvoxel<_nBins>& f) :
        u(f.u), p(f.p), h(f.h), v(f.v)
    {}

    /**
    *  \brief Copy operator.
    *
    *  \param vox Reference to \a fusionvoxel
    *  \return Reference to this \a fusionvoxel
    *
    *  \details Copies data from another \a fusionvoxel
    */
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

/** \brief fusionvoxel struct with overloaded \a new and \a delete operators from class \p Manage */
template<unsigned char _nBins>
struct Mfusionvoxel : public fusionvoxel<_nBins>, public Manage
{
    __host__ __device__ inline
    Mfusionvoxel(const fusionvoxel<_nBins> & f) : fusionvoxel<_nBins>(f)
    {}
};

/**
*  \brief Helper structure for calculating \f$\operatorname{prox}_{hist}(u)\f$.
*  \tparam _nBins   number of histogram bins
*/
template<unsigned char _nBins>
struct sortedHist
{
    /** \brief Array of elements */
    float element[2 * _nBins + 1];
    /** \brief Number of elements */
    unsigned char elements;

    /**
    *  \brief Default constructor
    *
    *  \details Sets \a elements to 0
    */
    __host__ __device__ inline
    sortedHist() : elements(0)
    {}

    /**
    *  \brief Constructor overload
    *
    *  \param bincenter    array of bin centers sorted from least to greatest
    *
    *  \details Sets \a elements to \a _nBins.
    */
    __host__ __device__ inline
    sortedHist(float bincenter[_nBins]) : elements(_nBins)
    {
        for (unsigned char i = 0; i < _nBins; i++) element[i] = bincenter[i];
    }

    /** \brief Copy constructor */
    __host__ __device__ inline
    sortedHist(const sortedHist<_nBins> & sh)
    {
        elements = sh.elements;
        for (unsigned char i = 0; i < elements; i++) element[i] = sh.element[i];
    }

    /**
    *  \brief Insertion sort function
    *
    *  \param val value to be inserted
    */
    __host__ __device__ inline
    void insert(float val)
    {
        unsigned char next;
        if (elements != 0)
        {
            for (char i = elements - 1; i >= 0; i--){
                next = fminf(i + 1, size() - 1);
                if (val < element[i]) element[next] = element[i];
                else {
                    element[next] = val;
                    break;
                }
            }
        }
        else element[0] = val;
        elements = fminf(elements + 1, size());
    }

    /**
    *  \brief Get median element value
    *
    *  \return Value of median element
    */
    __host__ __device__ inline
    float median(){ return element[_nBins]; }

    /** \brief Get size of the array */
    int size(){ return 2 * _nBins + 1; }
};

/** \brief sortedHist struct with overloaded \a new and \a delete operators from class \p Manage */
template<unsigned char _nBins>
struct MsortedHist : public sortedHist<_nBins>, public Manage
{
    __host__ __device__ inline
    MsortedHist(const sortedHist<_nBins> & sh) : sortedHist<_nBins>(sh)
    {}
};

/** \addtogroup rectangle Rectangle3D
* \brief Rectangle3D structure and its operator overloads
* @{
*/

/** \brief Simple struct to hold coordinates of volume rectangle */
struct Rectangle3D
{
    /** \brief Corner of rectangle */
    float3 a;
    /** \brief Opposite corner of rectangle */
    float3 b;

    /**
    *  \brief Default constructor
    *
    *  \details Corners are initialized to (0,0,0)
    */
    __host__ __device__ inline
    Rectangle3D() :
        a(make_float3(0,0,0)), b(make_float3(0,0,0))
    {}

    /**
    *  \brief Constructor overload
    *
    *  \param x corner of rectangle
    *  \param y opposite corner of rectangle
    *
    *  \details Constructs Rectangle3D from given corners
    */
    __host__ __device__ inline
    Rectangle3D(float3 x, float3 y) :
        a(x), b(y)
    {}

    /**
    *  \brief Copy constructor
    *
    *  \param r Rectangle3D to be copied
    *
    *  \details Constructs Rectangle3D from given Rectangle3D
    */
    __host__ __device__ inline
    Rectangle3D(const Rectangle3D& r) :
        a(r.a), b(r.b)
    {}

    /**
    *  \brief Get size of rectangle
    *
    *  \return Size of rectangle
    *
    *  \details Returns <em>b - a</em>
    */
    __host__ __device__ inline
    float3 size()
    {
        return (b - a);
    }

    /** \brief Get coordinates of the center of rectangle */
    __host__ __device__ inline
    float3 center()
    {
        return (b + a) / 2.f;
    }

    /** \brief Copy operator */
    __host__ __device__ inline
    Rectangle3D& operator=(const Rectangle3D & r)
    {
        if (this == &r) return *this;
        a = r.a;
        b = r.b;
        return *this;
    }
};

/** \brief Rectangle3D struct with overloaded \a new and \a delete operators from class \p Manage */
struct MRectangle3D : public Rectangle3D, public Manage
{
    __host__ __device__ inline
    MRectangle3D(const Rectangle3D & r) : Rectangle3D(r)
    {}
};

/** @} */ // group rectangle

/** \addtogroup matrix Matrix3D
* \brief Matrix3D structure and its operator overloads.
* @{
*/

/**
*  \brief 3 by 3 matrix convenience structure that works on device
*
*  \details Useful for storing and performing operations with \f$R\f$ and \f$K\f$ matrices.
*/
struct Matrix3D
{
    /** \brief Row vectors */
    float3 r[3]; // row vectors

    /**
    *  \brief Default constructor
    *
    *  \details Initializes to matrix of zeros
    */
    __host__ __device__ inline
    Matrix3D()
    {
        r[0] = (make_float3(0,0,0));
        r[1] = (make_float3(0,0,0));
        r[2] = (make_float3(0,0,0));
    }

    /**
    *  \brief Constructor overload
    *
    *  \param r1 \f$1^{st}\f$ row vector
    *  \param r2 \f$2^{nd}\f$ row vector
    *  \param r3 \f$3^{rd}\f$ row vector
    *
    *  \details Constructs matrix with given row vectors
    */
    __host__ __device__ inline
    Matrix3D(float3 r1, float3 r2, float3 r3)
    {
        r[0] = (r1);
        r[1] = (r2);
        r[2] = (r3);
    }

    /**
    *  \brief Constructor overload
    *
    *  \param m array of values
    *
    *  \details Constructs Matrix3D from a given 2D array
    */
    __host__ __device__ inline
    Matrix3D(float m[3][3])
    {
        r[0] = (make_float3(m[0][0], m[0][1], m[0][2]));
        r[1] = (make_float3(m[1][0], m[1][1], m[1][2]));
        r[2] = (make_float3(m[2][0], m[2][1], m[2][2]));
    }

    /**
    *  \brief Constructor overload
    *
    *  \param m array of values
    *
    *  \details Constructs Matrix3D from a given 2D array
    */
    __host__ __device__ inline
    Matrix3D(double m[3][3])
    {
        r[0] = (make_float3(m[0][0], m[0][1], m[0][2]));
        r[1] = (make_float3(m[1][0], m[1][1], m[1][2]));
        r[2] = (make_float3(m[2][0], m[2][1], m[2][2]));
    }

    /**
    *  \brief Constructor overload
    *
    *  \param m array of values
    *
    *  \details Constructs Matrix3D from a given row major array
    */
    __host__ __device__ inline
    Matrix3D(float m[9])
    {
        r[0] = (make_float3(m[0], m[1], m[2]));
        r[1] = (make_float3(m[3], m[4], m[5]));
        r[2] = (make_float3(m[6], m[7], m[8]));
    }

    /**
    *  \brief Constructor overload
    *
    *  \param m array of values
    *
    *  \details Constructs Matrix3D from a given row major array
    */
    __host__ __device__ inline
    Matrix3D(double m[9])
    {
        r[0] = (make_float3(m[0], m[1], m[2]));
        r[1] = (make_float3(m[3], m[4], m[5]));
        r[2] = (make_float3(m[6], m[7], m[8]));
    }

    /**
    *  \brief Constructor overload
    *
    *  \details Constructs Matrix3D from given element values
    */
    __host__ __device__ inline
    Matrix3D(float r11, float r12, float r13, float r21, float r22, float r23, float r31, float r32, float r33)
    {
        r[0] = make_float3(r11, r12, r13);
        r[1] = make_float3(r21, r22, r23);
        r[2] = make_float3(r31, r32, r33);
    }

    /** \brief Copy constructor */
    __host__ __device__ inline
    Matrix3D(const Matrix3D & R)
    {
        r[0] = (R.r[0]);
        r[1] = (R.r[1]);
        r[2] = (R.r[2]);
    }

    /**
    *  \brief Get reference to row vector
    *
    *  \param i index of row
    *  \return Reference to row vector
    */
    __host__ __device__ inline
    float3 & row(unsigned char i)
    {
        return r[i];
    }

    /**
    *  \brief Get constant reference to row vector
    *
    *  \param i index of row
    *  \return Constant reference to row vector
    */
    __host__ __device__ inline
    const float3 & row(unsigned char i) const
    {
        return r[i];
    }

    /**
    *  \brief Transpose matrix
    *
    *  \return Transpose of this Matrix3D
    */
    __host__ __device__ inline
    Matrix3D trans()
    {
        float3 c[3];
        c[0] = make_float3(r[0].x, r[1].x, r[2].x);
        c[1] = make_float3(r[0].y, r[1].y, r[2].y);
        c[2] = make_float3(r[0].z, r[1].z, r[2].z);
        return Matrix3D(c[0], c[1], c[2]);
    }

    /**
    *  \brief Calculate determinant of matrix
    *
    *  \return Determinant of this Matrix3D
    */
    __host__ __device__ inline
    float det()
    {
        return 	r[0].x * (r[1].y * r[2].z - r[1].z * r[2].y) -
                r[0].y * (r[1].x * r[2].z - r[1].z * r[2].x) +
                r[0].z * (r[1].x * r[2].y - r[1].y * r[2].x);
    }

    /**
    *  \brief Calculate inverse of matrix
    *
    *  \return Inverse of this Matrix3D
    */
    __host__ __device__ inline
    Matrix3D inv()
    {
        float d = det();
        float3 r1, r2, r3;
        r1 = make_float3(r[1].y*r[2].z - r[1].z*r[2].y, r[0].z*r[2].y - r[0].y*r[2].z, r[0].y*r[1].z - r[0].z*r[1].y) / d;
        r2 = make_float3(r[1].z*r[2].x - r[1].x*r[2].z, r[0].x*r[2].z - r[0].z*r[2].x, r[0].z*r[1].x - r[0].x*r[1].z) / d;
        r3 = make_float3(r[1].x*r[2].y - r[1].y*r[2].x, r[0].y*r[2].x - r[0].x*r[2].y, r[0].x*r[1].y - r[0].y*r[1].x) / d;
        return Matrix3D(r1, r2, r3);
    }

    /** \brief Set this matrix to identity */
    __host__ __device__ inline
    void makeIdentity()
    {
        r[0] = make_float3(1,0,0);
        r[1] = make_float3(0,1,0);
        r[2] = make_float3(0,0,1);
    }

    /**
    *  \brief Construct identity matrix
    *
    *  \return Identity Matrix3D
    */
    __host__ __device__ inline
    static Matrix3D identityMatrix()
    {
        Matrix3D m;
        m.makeIdentity();
        return m;
    }

    /** \brief Copy operator. */
    __host__ __device__ inline
    Matrix3D & operator=(const Matrix3D & R)
    {
        if (this == &R) return *this;
        r[0] = R.r[0];
        r[1] = R.r[1];
        r[2] = R.r[2];
        return *this;
    }

    /** \brief Assignment operator. */
    __host__ __device__ inline
    Matrix3D & operator=(float m[3][3])
    {
        r[0] = make_float3(m[0][0], m[0][1], m[0][2]);
        r[1] = make_float3(m[1][0], m[1][1], m[1][2]);
        r[2] = make_float3(m[2][0], m[2][1], m[2][2]);
        return *this;
    }

    /** \brief Assignment operator. */
    __host__ __device__ inline
    Matrix3D & operator=(double m[3][3])
    {
        r[0] = make_float3(m[0][0], m[0][1], m[0][2]);
        r[1] = make_float3(m[1][0], m[1][1], m[1][2]);
        r[2] = make_float3(m[2][0], m[2][1], m[2][2]);
        return *this;
    }

    /** \brief Assignment operator. */
    __host__ __device__ inline
    Matrix3D & operator=(float m[9])
    {
        r[0] = make_float3(m[0], m[1], m[2]);
        r[1] = make_float3(m[3], m[4], m[5]);
        r[2] = make_float3(m[6], m[7], m[8]);
        return *this;
    }

    /** \brief Assignment operator. */
    __host__ __device__ inline
    Matrix3D & operator=(double m[9])
    {
        r[0] = make_float3(m[0], m[1], m[2]);
        r[1] = make_float3(m[3], m[4], m[5]);
        r[2] = make_float3(m[6], m[7], m[8]);
        return *this;
    }

    /**
    *  \brief Access operator
    *
    *  \param row index of row
    *  \param col index of column
    *  \return Reference to specified element
    */
    __host__ __device__ inline
    float & operator()(unsigned char row, unsigned char col)
    {
        if (col == 0) return r[row].x;
        if (col == 1) return r[row].y;
        if (col == 2) return r[row].z;
        return r[0].x;
    }

    /**
    *  \brief Constant access operator
    *
    *  \param row index of row
    *  \param col index of column
    *  \return Constant reference to specified element
    */
    __host__ __device__ inline
    const float & operator()(unsigned char row, unsigned char col) const
    {
        if (col == 0) return r[row].x;
        if (col == 1) return r[row].y;
        if (col == 2) return r[row].z;
        return r[0].x;
    }

    /**
    *  \brief Access function
    *
    *  \param row index of row
    *  \param col index of column
    *  \return Reference to specified element
    */
    __host__ __device__ inline
    float & at(unsigned char row, unsigned char col)
    {
        if (col == 0) return r[row].x;
        if (col == 1) return r[row].y;
        if (col == 2) return r[row].z;
        return r[0].x;
    }

    /**
    *  \brief Const access function
    *
    *  \param row index of row
    *  \param col index of column
    *  \return Const reference to specified element
    */
    __host__ __device__ inline
    const float & at(unsigned char row, unsigned char col) const
    {
        if (col == 0) return r[row].x;
        if (col == 1) return r[row].y;
        if (col == 2) return r[row].z;
        return r[0].x;
    }

    /**
    *  \brief Access operator
    *
    *  \param row index of row
    *  \return Reference to specified row
    */
    __host__ __device__ inline
    float3 & operator()(unsigned char row)
    {
        return r[row];
    }

    /**
    *  \brief Constant access operator
    *
    *  \param row index of row
    *  \return Constant reference to specified row
    */
    __host__ __device__ inline
    const float3 & operator()(unsigned char row) const
    {
        return r[row];
    }
};

/** \brief Matrix3D structure with overloaded \n new and \a delete operators from class \p Manage */
struct MMatrix3D : public Matrix3D, public Manage
{
    __host__ __device__ inline
    MMatrix3D(const Matrix3D & m) : Matrix3D(m)
    {}
};

/** @} */ // group matrix3D

/** \addtogroup vector Vector3D
* \brief 3D vector structure fully compatible with float3 operators and functions but
* with overloaded \a new and \a delete operators for managed memory.
* @{
*/

/** \brief 3D vector structure */
struct Vector3D
{
    float x, y, z;

    /** \brief Constructor from float values*/
    __host__ __device__ inline
    Vector3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z)
    {}

    /** \brief Constructor overload from array of floats*/
    __host__ __device__ inline
    Vector3D(float v[3]) : x(v[0]), y(v[1]), z(v[2])
    {}

    /** \brief Constructor overload from array of doubles*/
    __host__ __device__ inline
    Vector3D(double v[3]) : x(v[0]), y(v[1]), z(v[2])
    {}

    /** \brief Constructor overload from float3*/
    __host__ __device__ inline
    Vector3D(float3 f) : x(f.x), y(f.y), z(f.z)
    {}

    /** \brief Copy constructor */
    __host__ __device__ inline
    Vector3D(const Vector3D & v) : x(v.x), y(v.y), z(v.z)
    {}

    /** \brief Assignment operator */
    __host__ __device__ inline
    Vector3D& operator=(float3 f)
    {
        x = f.x;
        y = f.y;
        z = f.z;
        return *this;
    }

    /** \brief Copy operator */
    __host__ __device__ inline
    Vector3D& operator=(const Vector3D & f)
    {
        if (this == &f) return *this;
        x = f.x;
        y = f.y;
        z = f.z;
        return *this;
    }

    /** \brief Assignment operator */
    __host__ __device__ inline
    Vector3D& operator=(float v[3])
    {
        x = v[0]; y = v[1]; z = v[2];
        return *this;
    }

    /** \brief Assignment operator */
    __host__ __device__ inline
    Vector3D& operator=(double v[3])
    {
        x = v[0]; y = v[1]; z = v[2];
        return *this;
    }

    /** \brief Typecast to float3 operator */
    __host__ __device__ inline
    operator float3() const
    {
        return make_float3(x, y, z);
    }
};

/** \brief Vector3D struct with overloaded \a new and \a delete operators from class \p Manage */
struct MVector3D : public Vector3D, public Manage
{
    __host__ __device__ inline
    MVector3D(const Vector3D & v) : Vector3D(v)
    {}
};

/** @} */ // group vector

#endif // STRUCTS_H
