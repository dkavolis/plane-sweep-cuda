#ifndef KITTI_DATA_H
#define KITTI_DATA_H

#include "kitti_reader.h"
#include <QObject>

namespace KITTI
{

    /**
     * @brief KITTIData class for storing data read by KITTIReader
     */
    class KITTIData : public QObject, public KITTIReader
    {
        Q_OBJECT

    public:
        // load data
        bool loadTimestamps();
        bool loadCalibration();
        bool loadData(int index);
        bool loadImages(int index);
        bool loadOxTS(int index);
        bool loadVelo(int index);

        // get calibration parameters:
        Matrix3D getK() const                                                   { return K; }
        Transformation3D getTFormVelo2Cam() const                               { return velo2cam; }
        Transformation3D getTFormIMU2Velo() const                               { return imu2velo; }
        const QVector<Matrix4D> & getTFormVelo2RectCam() const                  { return velo2rectcam; }
        Matrix4D getTFormVelo2RectCam(int index) const                          { return getTFormVelo2RectCam()[index]; }
        double getCornerDistance() const                                        { return cdist; }

        const QVector<CamProperties> & getCamProperties() const                 { return cprops; }
        const QVector<Transformation3D> & getTFormRectCam() const               { return camrt; }
        CamProperties getCamProperties(int index) const                         { return cprops[index]; }
        Transformation3D getTFormRectCam(int index) const                       { return camrt[index]; }
        int getNumberOfCameras() const                                          { return cprops.size(); }

        // get timestamps:
        const QVector<QVector<double>> & getTimestampsCamera() const            { return tsimg; }
        const QVector<double> & getTimestampsCamera(unsigned char cam) const    { return tsimg[cam]; }
        const QVector<double> & getTimestampsOxTS() const                       { return tsoxts; }
        const QVector<double> & getTimestampsVelo() const                       { return tsvelo; }
        const QVector<double> & getTimestampsVeloStart() const                  { return tsvelostart; }
        const QVector<double> & getTimestampsVeloEnd() const                    { return tsveloend; }
        double getTimestampsCamera(unsigned char cam, int index) const          { return tsimg[cam][index]; }
        double getTimestampsOxTS(int index) const                               { return tsoxts[index]; }
        double getTimestampsVelo(int index) const                               { return tsvelo[index]; }
        double getTimestampsVeloStart(int index) const                          { return tsvelostart[index]; }
        double getTimestampsVeloEnd(int index) const                            { return tsveloend[index]; }
        int getNumberOfTimestamps() const                                       { return tsoxts.size(); }

        // get data:
        const QVector<QImage> & getLoadedImages() const                         { return imgs; }
        const QVector<VeloPoint> & getLoadedVeloPoints() const                  { return points; }
        const QImage & getLoadedImages(int index) const                         { return imgs[index]; }
        OxTS getLoadedOxTS() const                                              { return oxts; }
        Transformation3D getLoadedOxTSTForm() const                             { return convertOxtsToTForm(oxts, scale); }
        Matrix4D getLoadedOxTSPose() const                                      { return Tr_0_inv * Matrix4D(getLoadedOxTSTForm(), make_float4(0,0,0,1)); }
        int getNumberOfImages() const                                           { return imgs.size(); }
        int getNumberOfVeloPoints() const                                       { return points.size(); }
        double getScale() const                                                 { return scale; }

        // redefine setters of KITTIReader as slots so they can be easily connected to widget signals
    public slots:
        void setCalibrationDir(const QString & c_dir)                           { KITTIReader::setCalibrationDir(c_dir); loadCalibration(); }
        void setBaseDir(const QString & b_dir)                                  { KITTIReader::setBaseDir(b_dir); loadReference(); loadTimestamps(); }
        void setFileNameLength(const int length = KITTI_FILENAME_LENGTH)        { KITTIReader::setFileNameLength(length); }

    protected:
        QVector<double> tsoxts, tsvelo, tsvelostart, tsveloend;
        QVector<QVector<double>> tsimg;
        Transformation3D velo2cam, imu2velo;
        QVector<Matrix4D> velo2rectcam;
        QVector<CamProperties> cprops;
        double cdist;
        QVector<Transformation3D> camrt;
        Matrix3D K;

        QVector<QImage> imgs;
        QVector<VeloPoint> points;
        OxTS oxts;
        Matrix4D Tr_0_inv;
        double scale;

    private:
        void loadReference();
    };

} // namespace KITTI

#endif // KITTI_DATA_H
