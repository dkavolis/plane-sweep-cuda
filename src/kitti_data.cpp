#include "kitti_data.h"
#include <QDir>

namespace KITTI
{
bool KITTIData::loadTimestamps()
{
    bool sc = true;
    if (!ReadTimestampFile(tsoxts, TSOxTS)) sc = false;
    if (!ReadTimestampFile(tsvelo, TSVelodyne)) sc = false;
    if (!ReadTimestampFile(tsvelostart, TSVelodyneStart)) sc = false;
    if (!ReadTimestampFile(tsveloend, TSVelodyneEnd)) sc = false;

    unsigned char cam = 0;
    while (QDir(KITTI_DIR_AND_NAME(bdir, KITTI_CAM_DIR(cam))).exists()){
        tsimg.resize(cam + 1);
        if (!ReadTimestampFile(tsimg[cam], TSImages, cam)) sc = false;
        cam++;
    }

    return sc;
}

bool KITTIData::loadCalibration()
{
    bool sc = ReadCalibration(cprops, cdist, imu2velo, velo2cam);

    if (cprops.size() > 0)
    {
        camrt.resize(cprops.size());
        for (int i = 0; i < cprops.size(); i++)
            camrt[i] = extractRT(cprops[i].P_rect);
    }

    calculateTransforms(K, velo2rectcam, velo2cam, cprops);

    return sc;
}

bool KITTIData::loadData(int index)
{
    return loadImages(index) & loadOxTS(index) & loadVelo(index);
}

bool KITTIData::loadImages(int index)
{
    bool sc = true;
    for (int i = 0; i < tsimg.size(); i++){
        sc &= ReadImageFile(imgs[i], i, index);
    }
    return sc;
}

bool KITTIData::loadOxTS(int index)
{
    return ReadOxTSFile(oxts, index);
}

bool KITTIData::loadVelo(int index)
{
    return ReadVeloFile(points, index);
}

void KITTIData::loadReference()
{
    OxTS o;
    if (!ReadOxTSFile(o, 0)) return;
    scale = latToScale(o.lat);
    Tr_0_inv = Matrix4D(convertOxtsToTForm(o, scale), make_float4(0,0,0,1)).inv();
}

} // namespace KITTI
