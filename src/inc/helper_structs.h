/**
 *  \file helper_structs.h
 *  \brief Header file containing useful operators for structures in structs.h and other device functions
 */
#ifndef HELPER_STRUCTS_H
#define HELPER_STRUCTS_H

#include "structs.h"

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
void operator-(Matrix3D & A)
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
void operator-=(Matrix3D & A, Matrix3D b)
{
    A = A - b;
}

/** @} */ // group matrix

/** @brief Matrix - vector multiplication */
__host__ __device__ inline
float3 operator*(Transformation3D t, float4 x)
{
    return t.R * make_float3(x.x, x.y, x.z) + t.T * x.w;
}

/** @brief Matrix - homogeneous vector multiplication */
__host__ __device__ inline
float3 operator*(Transformation3D t, float3 x)
{
    return t * make_float4(x.x, x.y, x.z, 1);
}

__host__ __device__ inline
Transformation3D operator*(Transformation3D t, Matrix4D m)
{
    m = m.trans();
    Transformation3D r;
    float4 r0 = make_float4(t.R(0), t.T.x),
            r1 = make_float4(t.R(1), t.T.y),
            r2 = make_float4(t.R(2), t.T.z);
    r.T = Vector3D(dot(r0, m.r4), dot(r1, m.r4), dot(r2, m.r4));
    r.R.r[0] = make_float3(dot(r0, m.r1), dot(r0, m.r2), dot(r0, m.r3));
    r.R.r[1] = make_float3(dot(r1, m.r1), dot(r1, m.r2), dot(r1, m.r3));
    r.R.r[2] = make_float3(dot(r2, m.r1), dot(r2, m.r2), dot(r2, m.r3));
    return r;
}

__host__ __device__ inline
Transformation3D operator*(Matrix3D m, Transformation3D t)
{
    Transformation3D r;
    r.R = m * t.R;
    r.T = m * t.T;
    return r;
}

__host__ __device__ inline
void operator*=(Transformation3D & t, Matrix4D m)
{
    t = t * m;
}

inline __host__ __device__ float5 make_float5(float s)
{
    return make_float5(s, s, s, s, s);
}
inline __host__ __device__ float5 make_float5(float4 a)
{
    return make_float5(a.x, a.y, a.z, a.w, 0.0f);
}
inline __host__ __device__ float5 make_float5(float4 a, float v)
{
    return make_float5(a.x, a.y, a.z, a.w, v);
}
inline __host__ __device__ float5 make_float5(int4 a, int b)
{
    return make_float5(float(a.x), float(a.y), float(a.z), float(a.w), float(b));
}
inline __host__ __device__ float5 make_float5(uint4 a, uint b)
{
    return make_float5(float(a.x), float(a.y), float(a.z), float(a.w), float(b));
}
inline __host__ __device__ float5 operator*(float5 a, float5 b)
{
    return make_float5(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w, a.v + b.v);
}
inline __host__ __device__ void operator*=(float5 &a, float5 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    a.v *= b.v;
}
inline __host__ __device__ float5 operator*(float5 a, float b)
{
    return make_float5(a.x * b, a.y * b, a.z * b,  a.w * b, a.v * b);
}
inline __host__ __device__ float5 operator*(float b, float5 a)
{
    return make_float5(b * a.x, b * a.y, b * a.z, b * a.w, b * a.v);
}
inline __host__ __device__ void operator*=(float5 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    a.v *= b;
}
inline __host__ __device__ float5 operator+(float5 a, float5 b)
{
    return make_float5(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w, a.v + b.v);
}
inline __host__ __device__ void operator+=(float5 &a, float5 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    a.v += b.v;
}
inline __host__ __device__ float5 operator+(float5 a, float b)
{
    return make_float5(a.x + b, a.y + b, a.z + b, a.w + b, a.v + b);
}
inline __host__ __device__ float5 operator+(float b, float5 a)
{
    return make_float5(a.x + b, a.y + b, a.z + b, a.w + b, a.v + b);
}
inline __host__ __device__ void operator+=(float5 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    a.v += b;
}
inline __host__ __device__ float5 operator-(float5 &a)
{
    return make_float5(-a.x, -a.y, -a.z, -a.w, -a.v);
}
inline __host__ __device__ float5 operator-(float5 a, float5 b)
{
    return make_float5(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w, a.v - b.v);
}
inline __host__ __device__ void operator-=(float5 &a, float5 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    a.v -= b.v;
}
inline __host__ __device__ float5 operator-(float5 a, float b)
{
    return make_float5(a.x - b, a.y - b, a.z - b,  a.w - b, a.v - b);
}
inline __host__ __device__ void operator-=(float5 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    a.v -= b;
}
inline __host__ __device__ float5 operator/(float5 a, float5 b)
{
    return make_float5(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w, a.v / b.v);
}
inline __host__ __device__ void operator/=(float5 &a, float5 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    a.v /= b.v;
}
inline __host__ __device__ float5 operator/(float5 a, float b)
{
    return make_float5(a.x / b, a.y / b, a.z / b,  a.w / b, a.v / b);
}
inline __host__ __device__ void operator/=(float5 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
    a.v /= b;
}
inline __host__ __device__ float5 operator/(float b, float5 a)
{
    return make_float5(b / a.x, b / a.y, b / a.z, b / a.w, b / a.v);
}
inline __device__ __host__ float5 clamp(float5 v, float a, float b)
{
    return make_float5(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b), clamp(v.v, a, b));
}
inline __device__ __host__ float5 clamp(float5 v, float5 a, float5 b)
{
    return make_float5(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w), clamp(v.v, a.v, b.v));
}
inline __host__ __device__ float dot(float5 a, float5 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w + a.v * b.v;
}
inline __host__ __device__ float5 fabs(float5 v)
{
    return make_float5(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w), fabs(v.v));
}
inline __host__ __device__ float5 floorf(float5 v)
{
    return make_float5(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w), floorf(v.v));
}
inline __host__ __device__ float5 fmaxf(float5 a, float5 b)
{
    return make_float5(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w), fmaxf(a.v, b.v));
}
inline  __host__ __device__ float5 fminf(float5 a, float5 b)
{
    return make_float5(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w), fminf(a.v, b.v));
}
inline __host__ __device__ float5 fmodf(float5 a, float5 b)
{
    return make_float5(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w), fmodf(a.v, b.v));
}
inline __host__ __device__ float5 fracf(float5 v)
{
    return make_float5(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w), fracf(v.v));
}
inline __host__ __device__ float length(float5 v)
{
    return sqrtf(dot(v, v));
}
inline __device__ __host__ float5 lerp(float5 a, float5 b, float t)
{
    return a + t*(b-a);
}
inline __host__ __device__ float5 normalize(float5 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __device__ __host__ float5 smoothstep(float5 a, float5 b, float5 x)
{
    float5 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float5(3.0f) - (make_float5(2.0f)*y)));
}

/** \brief Matrix - column vector multiplication */
__host__ __device__ inline
float4 operator*(Matrix4D R, float4 vec) // matrix - vector multiplication
{
    return make_float4(dot(R(0), vec), dot(R(1), vec), dot(R(2), vec), dot(R(3), vec));
}

/** \brief Matrix - matrix multiplication */
__host__ __device__ inline
Matrix4D operator*(Matrix4D A, Matrix4D B) // matrix - matrix multiplication
{
    B = B.trans();
    Matrix4D r;
    r.r1 = make_float4(dot(A(0), B(0)), dot(A(0), B(1)), dot(A(0), B(2)), dot(A(0), B(3)));
    r.r2 = make_float4(dot(A(1), B(0)), dot(A(1), B(1)), dot(A(1), B(2)), dot(A(1), B(3)));
    r.r3 = make_float4(dot(A(2), B(0)), dot(A(2), B(1)), dot(A(2), B(2)), dot(A(2), B(3)));
    r.r4 = make_float4(dot(A(3), B(0)), dot(A(3), B(1)), dot(A(3), B(2)), dot(A(3), B(3)));
    return r;
}

/** \brief Matrix - matrix multiplication */
__host__ __device__ inline
void operator*=(Matrix4D & A, Matrix4D B) // matrix - matrix multiplication
{
    A = A * B;
}

/** \brief Scale all matrix elements by \f$b\f$ */
__host__ __device__ inline
Matrix4D operator*(Matrix4D A, float b)
{
    return Matrix4D(A.r1 * b, A.r2 * b, A.r3 * b, A.r4 * b);
}

/** \brief Scale all matrix elements by \f$a\f$ */
__host__ __device__ inline
Matrix4D operator*(float a, Matrix4D B)
{
    return Matrix4D(B.r1 * a, B.r2 * a, B.r3 * a, B.r4 * a);
}

/** \brief Scale all matrix elements by \f$b\f$ */
__host__ __device__ inline
void operator*=(Matrix4D & A, float b)
{
    A = A * b;
}

/** \brief Scale all matrix elements by \f$b^{-1}\f$ */
__host__ __device__ inline
Matrix4D operator/(Matrix4D A, float b)
{
    return Matrix4D(A.r1 / b, A.r2 / b, A.r3 / b, A.r4 / b);
}

/** \brief Scale all matrix elements by \f$b^{-1}\f$ */
__host__ __device__ inline
void operator/=(Matrix4D & A, float b)
{
    A = A / b;
}

/** \brief Add constant \f$b\f$ to all matrix elements */
__host__ __device__ inline
Matrix4D operator+(Matrix4D A, float b)
{
    return Matrix4D(A.r1 + b, A.r2 + b, A.r3 + b, A.r4 + b);
}

/** \brief Add constant \f$a\f$ to all matrix elements */
__host__ __device__ inline
Matrix4D operator+(float a, Matrix4D B)
{
    return Matrix4D(B.r1 + a, B.r2 + a, B.r3 + a, B.r4 + a);
}

/** \brief Matrix - matrix summation */
__host__ __device__ inline
Matrix4D operator+(Matrix4D a, Matrix4D b)
{
    return Matrix4D(a.r1 + b.r1, a.r2 + b.r2, a.r3 + b.r3, a.r4 + b.r4);
}

/** \brief Add constant \f$b\f$ to all matrix elements */
__host__ __device__ inline
void operator+=(Matrix4D & A, float b)
{
    A = A + b;
}

/** \brief Matrix - matrix summation */
__host__ __device__ inline
void operator+=(Matrix4D & A, Matrix4D b)
{
    A = A + b;
}

/** \brief Negate matrix */
__host__ __device__ inline
void operator-(Matrix4D & A)
{
    A = Matrix4D(-A.r1, -A.r2, -A.r3, -A.r4);
}

/** \brief Subtract constant \f$b\f$ from all matrix elements */
__host__ __device__ inline
Matrix4D operator-(Matrix4D A, float b)
{
    return Matrix4D(A.r1 - b, A.r2 - b, A.r3 - b, A.r4 - b);
}

/** \brief Subtract constant \f$a\f$ from all matrix elements */
__host__ __device__ inline
Matrix4D operator-(float a, Matrix4D B)
{
    return Matrix4D(B.r1 - a, B.r2 - a, B.r3 - a, B.r4 - a);
}

/** \brief Matrix - matrix subtraction */
__host__ __device__ inline
Matrix4D operator-(Matrix4D a, Matrix4D b)
{
    return Matrix4D(a.r1 - b.r1, a.r2 - b.r2, a.r3 - b.r3, a.r4 - b.r4);
}

/** \brief Subtract constant \f$b\f$ from all matrix elements */
__host__ __device__ inline
void operator-=(Matrix4D & A, float b)
{
    A = A - b;
}

/** \brief Matrix - matrix subtraction */
__host__ __device__ inline
void operator-=(Matrix4D & A, Matrix4D b)
{
    A = A - b;
}

#endif // HELPER_STRUCTS_H
