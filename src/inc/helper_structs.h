#ifndef HELPER_STRUCTS_H
#define HELPER_STRUCTS_H

#include "structs.h"

//////////////////////////////////////////////////////////////
// Rectangle operator overloads
//////////////////////////////////////////////////////////////
/** \addtogroup rectangle
 * @{
 */

/**
*  \brief Scale corner positions by \f$b\f$
*/
__host__ __device__ inline
Rectangle operator*(Rectangle r, float b)
{
    return Rectangle(r.a * b, r.b * b);
}

/**
*  \brief Scale corner positions by \f$b\f$
*/
__host__ __device__ inline
Rectangle operator*(float b, Rectangle r)
{
    return Rectangle(r.a * b, r.b * b);
}

/**
*  \brief Scale corner positions by \f$b\f$
*/
__host__ __device__ inline
void operator*=(Rectangle &r, float b)
{
    r.a *= b; r.b *= b;
}

/**
*  \brief Scale corner positions by \f$b^{-1}\f$
*/
__host__ __device__ inline
Rectangle operator/(Rectangle r, float b)
{
    return Rectangle(r.a / b, r.b / b);
}

/**
*  \brief Scale corner positions by \f$b^{-1}\f$
*/
__host__ __device__ inline
Rectangle operator/(float b, Rectangle r)
{
    return Rectangle(b / r.a, b / r.b);
}

/**
*  \brief Scale corner positions by \f$b^{-1}\f$
*/
__host__ __device__ inline
void operator/=(Rectangle &r, float b)
{
    r.a /= b; r.b /= b;
}

/**
*  \brief Apply vector \f$(b, b, b)^T\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator+(Rectangle r, float b)
{
    return Rectangle(r.a + b, r.b + b);
}

/**
*  \brief Apply vector \f$(b, b, b)^T\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator+(float b, Rectangle r)
{
    return Rectangle(r.a + b, r.b + b);
}

/**
*  \brief Apply vector \f$(b, b, b)^T\f$ translation to corner positions
*/
__host__ __device__ inline
void operator+=(Rectangle &r, float b)
{
    r.a += b; r.b += b;
}

/**
*  \brief Apply vector \f$-(b, b, b)^T\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator-(Rectangle r, float b)
{
    return Rectangle(r.a - b, r.b - b);
}

/**
*  \brief Apply vector \f$(b, b, b)^T\f$ translation to negated corner positions
*/
__host__ __device__ inline
Rectangle operator-(float b, Rectangle r)
{
    return Rectangle(b - r.a, b - r.b);
}

/**
*  \brief Apply vector \f$-(b, b, b)^T\f$ translation to corner positions
*/
__host__ __device__ inline
void operator-=(Rectangle &r, float b)
{
    r.a -= b; r.b -= b;
}

/**
*  \brief Apply vector \f$b\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator+(Rectangle r, float3 b)
{
    return Rectangle(r.a + b, r.b + b);
}

/**
*  \brief Apply vector \f$b\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator+(float3 b, Rectangle r)
{
    return Rectangle(r.a + b, r.b + b);
}

/**
*  \brief Apply vector \f$b\f$ translation to corner positions
*/
__host__ __device__ inline
void operator+=(Rectangle &r, float3 b)
{
    r.a += b; r.b += b;
}

/**
*  \brief Apply vector \f$-b\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator-(Rectangle r, float3 b)
{
    return Rectangle(r.a - b, r.b - b);
}

/**
*  \brief Apply vector \f$-b\f$ translation to corner positions
*/
__host__ __device__ inline
Rectangle operator-(float3 b, Rectangle r)
{
    return Rectangle(b - r.a, b - r.b);
}

/**
*  \brief Apply vector \f$-b\f$ translation to corner positions
*/
__host__ __device__ inline
void operator-=(Rectangle &r, float3 b)
{
    r.a -= b; r.b -= b;
}

 /** @} */ // group rectangle

//////////////////////////////////////////////////////////
// Matrix3D operator overloads
//////////////////////////////////////////////////////////
/** \addtogroup matrix
 * @{
 */

/**
*  \brief Matrix - column vector multiplication
*/
__host__ __device__ inline
float3 operator*(Matrix3D R, float3 vec) // matrix - vector multiplication
{
	return make_float3(dot(R(0), vec), dot(R(1), vec), dot(R(2), vec));
}

/**
*  \brief Matrix - matrix multiplication
*/
__host__ __device__ inline
Matrix3D operator*(Matrix3D A, Matrix3D B) // matrix - matrix multiplication
{
	B.trans();
	Matrix3D r;
	r.r[0] = make_float3(dot(A(0), B(0)), dot(A(0), B(1)), dot(A(0), B(2)));
	r.r[1] = make_float3(dot(A(1), B(0)), dot(A(1), B(1)), dot(A(1), B(2)));
	r.r[2] = make_float3(dot(A(2), B(0)), dot(A(2), B(1)), dot(A(2), B(2)));
	return r;
}

/**
*  \brief Matrix - matrix multiplication
*/
__host__ __device__ inline
void operator*=(Matrix3D & A, Matrix3D B) // matrix - matrix multiplication
{
	A = A * B;
}

/**
*  \brief Scale all matrix elements by \f$b\f$
*/
__host__ __device__ inline
Matrix3D operator*(Matrix3D A, float b)
{
	return Matrix3D(A.r[0] * b, A.r[1] * b, A.r[2] * b);
}

/**
*  \brief Scale all matrix elements by \f$a\f$
*/
__host__ __device__ inline
Matrix3D operator*(float a, Matrix3D B)
{
	return Matrix3D(B.r[0] * a, B.r[1] * a, B.r[2] * a);
}

/**
*  \brief Scale all matrix elements by \f$b\f$
*/
__host__ __device__ inline
void operator*=(Matrix3D & A, float b)
{
	A = A * b;
}

/**
*  \brief Scale all matrix elements by \f$b^{-1}\f$
*/
__host__ __device__ inline
Matrix3D operator/(Matrix3D A, float b)
{
	return Matrix3D(A.r[0] / b, A.r[1] / b, A.r[2] / b);
}

/**
*  \brief Scale all matrix elements by \f$b^{-1}\f$
*/
__host__ __device__ inline
void operator/=(Matrix3D & A, float b)
{
	A = A / b;
}

/**
*  \brief Add constant \f$b\f$ to all matrix elements
*/
__host__ __device__ inline
Matrix3D operator+(Matrix3D A, float b)
{
	return Matrix3D(A.r[0] + b, A.r[1] + b, A.r[2] + b);
}

/**
*  \brief Add constant \f$a\f$ to all matrix elements
*/
__host__ __device__ inline
Matrix3D operator+(float a, Matrix3D B)
{
	return Matrix3D(B.r[0] + a, B.r[1] + a, B.r[2] + a);
}

/**
*  \brief Add constant \f$b\f$ to all matrix elements
*/
__host__ __device__ inline
void operator+=(Matrix3D & A, float b)
{
	A = A + b;
}

/**
*  \brief Subtract constant \f$b\f$ from all matrix elements
*/
__host__ __device__ inline
Matrix3D operator-(Matrix3D A, float b)
{
	return Matrix3D(A.r[0] - b, A.r[1] - b, A.r[2] - b);
}

/**
*  \brief Subtract constant \f$a\f$ from all matrix elements
*/
__host__ __device__ inline
Matrix3D operator-(float a, Matrix3D B)
{
	return Matrix3D(B.r[0] - a, B.r[1] - a, B.r[2] - a);
}

/**
*  \brief Subtract constant \f$b\f$ from all matrix elements
*/
__host__ __device__ inline
void operator-=(Matrix3D & A, float b)
{
	A = A - b;
}

 /** @} */ // group matrix

#endif // HELPER_STRUCTS_H
