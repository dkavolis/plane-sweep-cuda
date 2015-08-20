/**
 *  \file helper_structs.h
 *  \brief Header file containing useful operators for structures in structs.h and other device functions
 */
#ifndef HELPER_STRUCTS_H
#define HELPER_STRUCTS_H

#include "structs.h"

//////////////////////////////////////////////////////////////
// General device functions
//////////////////////////////////////////////////////////////
/** \addtogroup general
* @{
*/

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

/** @} */ // group general

//////////////////////////////////////////////////////////////
// Rectangle3D operator overloads
//////////////////////////////////////////////////////////////
/** \addtogroup rectangle
 * @{
 */

/** \brief Scale corner positions by \f$b\f$ */
__host__ __device__ inline
Rectangle3D operator*(Rectangle3D r, float b)
{
    return Rectangle3D(r.a * b, r.b * b);
}

/** \brief Scale corner positions by \f$b\f$ */
__host__ __device__ inline
Rectangle3D operator*(float b, Rectangle3D r)
{
    return Rectangle3D(r.a * b, r.b * b);
}

/** \brief Scale corner positions by \f$b\f$ */
__host__ __device__ inline
void operator*=(Rectangle3D &r, float b)
{
    r.a *= b;
    r.b *= b;
}

/** \brief Scale corner positions by \f$b^{-1}\f$ */
__host__ __device__ inline
Rectangle3D operator/(Rectangle3D r, float b)
{
    return Rectangle3D(r.a / b, r.b / b);
}

/** \brief Scale corner positions by \f$b^{-1}\f$ */
__host__ __device__ inline
Rectangle3D operator/(float b, Rectangle3D r)
{
    return Rectangle3D(b / r.a, b / r.b);
}

/** \brief Scale corner positions by \f$b^{-1}\f$ */
__host__ __device__ inline
void operator/=(Rectangle3D &r, float b)
{
    r.a /= b;
    r.b /= b;
}

/** \brief Apply vector \f$(b, b, b)^T\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator+(Rectangle3D r, float b)
{
    return Rectangle3D(r.a + b, r.b + b);
}

/** \brief Apply vector \f$(b, b, b)^T\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator+(float b, Rectangle3D r)
{
    return Rectangle3D(r.a + b, r.b + b);
}

/** \brief Apply vector \f$(b, b, b)^T\f$ translation to corner positions */
__host__ __device__ inline
void operator+=(Rectangle3D &r, float b)
{
    r.a += b;
    r.b += b;
}

/** \brief Apply vector \f$-(b, b, b)^T\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator-(Rectangle3D r, float b)
{
    return Rectangle3D(r.a - b, r.b - b);
}

/** \brief Apply vector \f$(b, b, b)^T\f$ translation to negated corner positions */
__host__ __device__ inline
Rectangle3D operator-(float b, Rectangle3D r)
{
    return Rectangle3D(b - r.a, b - r.b);
}

/** \brief Apply vector \f$-(b, b, b)^T\f$ translation to corner positions */
__host__ __device__ inline
void operator-=(Rectangle3D &r, float b)
{
    r.a -= b;
    r.b -= b;
}

/** \brief Apply vector \f$b\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator+(Rectangle3D r, float3 b)
{
    return Rectangle3D(r.a + b, r.b + b);
}

/** \brief Apply vector \f$b\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator+(float3 b, Rectangle3D r)
{
    return Rectangle3D(r.a + b, r.b + b);
}

/** \brief Apply vector \f$b\f$ translation to corner positions */
__host__ __device__ inline
void operator+=(Rectangle3D &r, float3 b)
{
    r.a += b;
    r.b += b;
}

/** \brief Apply vector \f$-b\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator-(Rectangle3D r, float3 b)
{
    return Rectangle3D(r.a - b, r.b - b);
}

/** \brief Apply vector \f$-b\f$ translation to corner positions */
__host__ __device__ inline
Rectangle3D operator-(float3 b, Rectangle3D r)
{
    return Rectangle3D(b - r.a, b - r.b);
}

/** \brief Apply vector \f$-b\f$ translation to corner positions */
__host__ __device__ inline
void operator-=(Rectangle3D &r, float3 b)
{
    r.a -= b;
    r.b -= b;
}

/** @} */ // group rectangle

//////////////////////////////////////////////////////////
// Matrix3D operator overloads
//////////////////////////////////////////////////////////
/** \addtogroup matrix
 * @{
 */

/** \brief Matrix - column vector multiplication */
__host__ __device__ inline
float3 operator*(Matrix3D R, float3 vec) // matrix - vector multiplication
{
    return make_float3(dot(R(0), vec), dot(R(1), vec), dot(R(2), vec));
}

/** \brief Matrix - matrix multiplication */
__host__ __device__ inline
Matrix3D operator*(Matrix3D A, Matrix3D B) // matrix - matrix multiplication
{
    B = B.trans();
    Matrix3D r;
    r.r[0] = make_float3(dot(A(0), B(0)), dot(A(0), B(1)), dot(A(0), B(2)));
    r.r[1] = make_float3(dot(A(1), B(0)), dot(A(1), B(1)), dot(A(1), B(2)));
    r.r[2] = make_float3(dot(A(2), B(0)), dot(A(2), B(1)), dot(A(2), B(2)));
    return r;
}

/** \brief Matrix - matrix multiplication */
__host__ __device__ inline
void operator*=(Matrix3D & A, Matrix3D B) // matrix - matrix multiplication
{
    A = A * B;
}

/** \brief Scale all matrix elements by \f$b\f$ */
__host__ __device__ inline
Matrix3D operator*(Matrix3D A, float b)
{
    return Matrix3D(A.r[0] * b, A.r[1] * b, A.r[2] * b);
}

/** \brief Scale all matrix elements by \f$a\f$ */
__host__ __device__ inline
Matrix3D operator*(float a, Matrix3D B)
{
    return Matrix3D(B.r[0] * a, B.r[1] * a, B.r[2] * a);
}

/** \brief Scale all matrix elements by \f$b\f$ */
__host__ __device__ inline
void operator*=(Matrix3D & A, float b)
{
    A = A * b;
}

/** \brief Scale all matrix elements by \f$b^{-1}\f$ */
__host__ __device__ inline
Matrix3D operator/(Matrix3D A, float b)
{
    return Matrix3D(A.r[0] / b, A.r[1] / b, A.r[2] / b);
}

/** \brief Scale all matrix elements by \f$b^{-1}\f$ */
__host__ __device__ inline
void operator/=(Matrix3D & A, float b)
{
    A = A / b;
}

/** \brief Add constant \f$b\f$ to all matrix elements */
__host__ __device__ inline
Matrix3D operator+(Matrix3D A, float b)
{
    return Matrix3D(A.r[0] + b, A.r[1] + b, A.r[2] + b);
}

/** \brief Add constant \f$a\f$ to all matrix elements */
__host__ __device__ inline
Matrix3D operator+(float a, Matrix3D B)
{
    return Matrix3D(B.r[0] + a, B.r[1] + a, B.r[2] + a);
}

/** \brief Matrix - matrix summation */
__host__ __device__ inline
Matrix3D operator+(Matrix3D a, Matrix3D b)
{
    return Matrix3D(a.r[0] + b.r[0], a.r[1] + b.r[1], a.r[2] + b.r[2]);
}

/** \brief Add constant \f$b\f$ to all matrix elements */
__host__ __device__ inline
void operator+=(Matrix3D & A, float b)
{
    A = A + b;
}

/** \brief Matrix - matrix summation */
__host__ __device__ inline
void operator+=(Matrix3D & A, Matrix3D b)
{
    A = A + b;
}

/** \brief Negate matrix */
__host__ __device__ inline
Matrix3D operator-(Matrix3D & A)
{
    A = Matrix3D(-A.r[0], -A.r[1], -A.r[2]);
}

/** \brief Subtract constant \f$b\f$ from all matrix elements */
__host__ __device__ inline
Matrix3D operator-(Matrix3D A, float b)
{
    return Matrix3D(A.r[0] - b, A.r[1] - b, A.r[2] - b);
}

/** \brief Subtract constant \f$a\f$ from all matrix elements */
__host__ __device__ inline
Matrix3D operator-(float a, Matrix3D B)
{
    return Matrix3D(B.r[0] - a, B.r[1] - a, B.r[2] - a);
}

/** \brief Matrix - matrix subtraction */
__host__ __device__ inline
Matrix3D operator-(Matrix3D a, Matrix3D b)
{
    return Matrix3D(a.r[0] - b.r[0], a.r[1] - b.r[1], a.r[2] - b.r[2]);
}

/** \brief Subtract constant \f$b\f$ from all matrix elements */
__host__ __device__ inline
void operator-=(Matrix3D & A, float b)
{
    A = A - b;
}

/** \brief Matrix - matrix subtraction */
__host__ __device__ inline
Matrix3D operator-=(Matrix3D & A, Matrix3D b)
{
    A = A - b;
}

/** @} */ // group matrix

#endif // HELPER_STRUCTS_H
