/**
 *  \file structs.h
 *  \brief Header file containing helper structs
 */
#ifndef STRUCTS_H
#define STRUCTS_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_math.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/assignment.hpp>

/**
*  \brief Convenience typedef for <em>unsigned int</em>
*/
typedef unsigned int uint;

namespace ublas = boost::numeric::ublas;

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
	/**
         *  \brief Array of bins
	 */
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

    /**
     *  \brief Copy operator
     *
     *  \param hist histogram to be copied from
     *  \return Reference to this histogram
     *
     *  \details
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
     *  
     *  \details
     */
    __device__ __host__ inline
    unsigned char& first(){ return bin[0]; }

    	/**
     *  \brief Access to last bin (empty voxel)
     *
     *  \return Reference to last bin
     *  
     *  \details
     */
    __device__ __host__ inline
    unsigned char& last(){ return bin[_nBins - 1]; }

    	/**
     *  \brief Access operator
     *
     *  \param i index of bin
     *  \return Reference to bin at index \a i
     *  
     *  \details
     */
    __device__ __host__ inline
    unsigned char& operator()(unsigned char i){ return bin[i]; }
};

// Simple structure to hold all data of a single voxel
//required for depthmap fusion
/**
*  \brief Simple structure to hold all data of a single voxel required for depthmap fusion
*  \tparam _nBins   number of histogram bins
*
*  \details
*/
template<unsigned char _nBins>
struct fusionvoxel
{
	/**
         *  \brief Primal variable \f$u\f$
	 */
    float u;
	/**
         *  \brief Helper variable \f$v\f$
	 */
    float v;
	/**
         *  \brief Dual variable \f$p\f$
	 */
    float3 p;
	/**
         *  \brief Histogram
	 */
    histogram<_nBins> h;

    	/**
     *  \brief Default constructor
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

    	/**
     *  \brief Constructor overload
     *  
     *  \param f Reference to \a fusionvoxel
     *
     *  \details Copies data from another \a fusionvoxel
     */
    __host__ __device__ inline
    fusionvoxel(const fusionvoxel<_nBins>& f) :
        u(f.u), p(f.p), h(f.h), v(f.v)
    {}

    /**
     *  \brief Assignment operator.
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

/**
*  \brief Helper structure for calculating \f$\operatorname{prox}_{hist}(u)\f$.
*  \tparam _nBins   number of histogram bins
*
*  \details
*/
template<unsigned char _nBins>
struct sortedHist
{
	/**
         *  \brief Array of elements
	 */
    double element[2 * _nBins + 1];
	/**
         *  \brief Number of elements
	 */
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
    sortedHist(double bincenter[_nBins]) : elements(_nBins)
    {
        for (unsigned char i = 0; i < _nBins; i++) element[i] = bincenter[i];
    }

    	/**
     *  \brief Insertion sort function
     *  
     *  \param val value to be inserted
     *  \return No return value
     *  
     *  \details
     */
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
        elements = fminf(elements + 1, 2 _nBins + 1);
    }

    	/**
     *  \brief Get median element value
     *  
     *  \return Value of median element
     *  
     *  \details
     */
    __host__ __device__ inline
    double median(){ return element[_nBins]; }
};

/** \addtogroup rectangle Rectangle
* \brief Rectangle structure and its operator overloads
* @{
*/

/**
*  \brief Simple struct to hold coordinates of volume rectangle
*/
struct Rectangle
{
	/**
         *  \brief Corner of rectangle
	 */
    float3 a;
	/**
         *  \brief Opposite corner of rectangle
	 */
	float3 b;

    	/**
     *  \brief Default constructor
     *  
     *  \details Corners are initialized to (0,0,0)
     */
    __host__ __device__ inline
    Rectangle() :
        a(make_float3(0,0,0)), b(make_float3(0,0,0))
    {}

    	/**
     *  \brief Constructor overload
     *  
     *  \param x corner of rectangle
     *  \param y opposite corner of rectangle
     *  
     *  \details Constructs Rectangle from given corners
     */
    __host__ __device__ inline
    Rectangle(float3 x, float3 y) :
        a(x), b(y)
    {}

    	/**
     *  \brief Constructor overload
     *  
     *  \param r Rectangle to be copied
     *  
     *  \details Constructs Rectangle from given Rectangle
     */
    __host__ __device__ inline
    Rectangle(const Rectangle& r) :
        a(r.a), b(r.b)
    {}

    	/**
     *  \brief Get size of rectangle
     *  
     *  \return Size of rectangle
     *  
     *  \details Returns <em>a - b</em>
     */
    __host__ __device__ inline
    float3 size()
    {
        return (a - b);
    }

    /**
     *  \brief Assignment operator
     */
    __host__ __device__ inline
    Rectangle& operator=(Rectangle & r)
    {
        if (this == &r) return *this;
        a = r.a;
        b = r.b;
        return *this;
    }
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
* Host operators are overloaded to work with \a boost \a matrix.
*/
struct Matrix3D
{
	/**
         *  \brief Row vectors
	 */
	float3 r[3]; // row vectors
	
	/**
         *  \brief Default constructor
	 *  
         *  \details Initializes to matrix of zeros
	 */
	__host__ __device__ inline
	Matrix3D() : r[0](make_float3(0,0,0)), r[1](make_float3(0,0,0)), r[2](make_float3(0,0,0))
	{}
	
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
	Matrix3D(float3 r1, float3 r2, float3 r3) : r[0](r1), r[1](r2), r[2](r3)
	{}
	
	/**
         *  \brief Constructor overload
	 *  
         *  \param R reference to \a boost matrix of size at least (3,3)
	 *  
         *  \details Constructs Matrix3D from given \a boost matrix
	 */
	__host__ inline
        Matrix3D(ublas::matrix<double>& R) :
		r[0](make_float3(R(0,0),R(0,1),R(0,2))),
		r[1](make_float3(R(1,0),R(1,1),R(1,2))),
		r[2](make_float3(R(2,0),R(2,1),R(2,2)))
	{}
	
        /**
         *  \brief Constructor overload
         *
         *  \param R reference to \a boost matrix of size at least (3,3)
         *
         *  \details Constructs Matrix3D from given \a boost matrix
         */
	__host__ inline
        Matrix3D(ublas::matrix<float> & R) :
		r[0](make_float3(R(0,0),R(0,1),R(0,2))),
		r[1](make_float3(R(1,0),R(1,1),R(1,2))),
		r[2](make_float3(R(2,0),R(2,1),R(2,2)))
	{}
	
	/**
	 *  \brief Constructor overload
	 *  
	 *  \param m array of values
	 *  
	 *  \details Constructs Matrix3D from given array  	 
	 */
	__host__ __device__ inline
	Matrix3D(float m[3][3]) :
		r[0](make_float3(m[0][0], m[0][1], m[0][2])),
		r[1](make_float3(m[1][0], m[1][1], m[1][2])),
		r[2](make_float3(m[2][0], m[2][1], m[2][2]))
		{}
		
	/**
	 *  \brief Constructor overload
	 *  
	 *  \param m array of values
	 *  
	 *  \details Constructs Matrix3D from given array  	 
	 */
	__host__ __device__ inline
	Matrix3D(double m[3][3]) :
		r[0](make_float3(m[0][0], m[0][1], m[0][2])),
		r[1](make_float3(m[1][0], m[1][1], m[1][2])),
		r[2](make_float3(m[2][0], m[2][1], m[2][2]))
		{}
		
	/**
	 *  \brief Constructor overload
	 *  
	 *  \param m array of values
	 *  
	 *  \details Constructs Matrix3D from given array  	 
	 */
	__host__ __device__ inline
	Matrix3D(float m[9]) :
		r[0](make_float3(m[0], m[1], m[2])),
		r[1](make_float3(m[3], m[4], m[5])),
		r[2](make_float3(m[6], m[7], m[8]))
	{}
	
	/**
	 *  \brief Constructor overload
	 *  
	 *  \param m array of values
	 *  
	 *  \details Constructs Matrix3D from given array  	 
	 */
	__host__ __device__ inline
	Matrix3D(double m[9]) :
		r[0](make_float3(m[0], m[1], m[2])),
		r[1](make_float3(m[3], m[4], m[5])),
		r[2](make_float3(m[6], m[7], m[8]))
	{}
	
        /**
         *  \brief Constructor overload
         *
         *  \param R reference to another Matrix3D
         *
         *  \details Constructs Matrix3D from given Matrix3D
         */
	__host__ __device__ inline
	Matrix3D(const Matrix3D & R) : r[0](R.r[0]), r[1](R.r[1]), r[2](R.r[2])
	{}
	
	/**
         *  \brief Get reference to row vector
	 *  
         *  \param i index of row
         *  \return Reference to row vector
	 *  
         *  \details
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
         *
         *  \details
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
	 *  
         *  \details
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
	 *  
	 *  \details
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
	 *  
	 *  \details
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
	
	/**
         *  \brief Set this matrix to identity
	 *  
         *  \return No return value
	 *  
         *  \details
	 */
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
	 *  
         *  \details
	 */
	__host__ __device__ inline
	static Matrix3D identityMatrix()
	{
		Matrix3D m;
		m.makeIdentity();
		return m;
	}
	
        /**
         *  \brief Assignment operator.
         */
	__host__ __device__ inline
	Matrix3D & operator=(Matrix3D & R)
	{
		if (this == &R) return *this;
		r[0] = R.r[0];
		r[1] = R.r[1];
		r[2] = R.r[2];
        return *this;
	}
	
        /**
         *  \brief Assignment operator.
         */
	__host__ inline
	Matrix3D & operator=(ublas::matrix<double> & R)
	{
		r[0] = make_float3(R(0,0),R(0,1),R(0,2));
		r[1] = make_float3(R(1,0),R(1,1),R(1,2));
		r[2] = make_float3(R(2,0),R(2,1),R(2,2));
		return *this;
	}
	
        /**
         *  \brief Assignment operator.
         */
	__host__ inline
	Matrix3D & operator=(ublas::matrix<float> & R)
	{
		r[0] = make_float3(R(0,0),R(0,1),R(0,2));
		r[1] = make_float3(R(1,0),R(1,1),R(1,2));
		r[2] = make_float3(R(2,0),R(2,1),R(2,2));
		return *this;
	}
	
	/**
	 *  \brief Assignment operator.
	 */
	__host__ __device__ inline
	Matrix3D & operator=(float m[3][3])
	{
		r[0] = (make_float3(m[0][0], m[0][1], m[0][2]));
		r[1] = (make_float3(m[1][0], m[1][1], m[1][2]));
		r[2] = (make_float3(m[2][0], m[2][1], m[2][2]));
	}
	
	/**
	 *  \brief Assignment operator.
	 */
	__host__ __device__ inline
	Matrix3D & operator=(double m[3][3])
	{
		r[0] = (make_float3(m[0][0], m[0][1], m[0][2]));
		r[1] = (make_float3(m[1][0], m[1][1], m[1][2]));
		r[2] = (make_float3(m[2][0], m[2][1], m[2][2]));
	}
	
	/**
	 *  \brief Assignment operator.
	 */
	__host__ __device__ inline
	Matrix3D & operator=(float m[9])
	{
		r[0] = (make_float3(m[0], m[1], m[2]));
		r[1] = (make_float3(m[3], m[4], m[5]));
		r[2] = (make_float3(m[6], m[7], m[8]));
	}
	
	/**
	 *  \brief Assignment operator.
	 */
	__host__ __device__ inline
	Matrix3D & operator=(double m[9])
	{
		r[0] = (make_float3(m[0], m[1], m[2]));
		r[1] = (make_float3(m[3], m[4], m[5]));
		r[2] = (make_float3(m[6], m[7], m[8]));
	}
	
	/**
         *  \brief Access operator
	 *  
         *  \param row index of row
         *  \param col index of column
         *  \return Reference to specified element
	 *  
         *  \details
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
         *
         *  \details
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
         *  \brief Access operator
         *
         *  \param row index of row
         *  \return Reference to specified row
         *
         *  \details
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
         *
         *  \details
         */
	__host__ __device__ inline
	const float3 & operator()(unsigned char row) const
	{
		return r[row];
	}
};

 /** @} */ // group matrix3D

#endif // STRUCTS_H
