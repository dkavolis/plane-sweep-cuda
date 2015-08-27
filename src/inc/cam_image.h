#ifndef CAM_IMAGE_H
#define CAM_IMAGE_H

#include "image.h"
#include "structs.h"

template<typename T>
struct CamImage : public Image<T, Standard>
{
    virtual ~CamImage() { free(); }

    Matrix3D R;
    Vector3D t;
};

#endif // CAM_IMAGE_H
