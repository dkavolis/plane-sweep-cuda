#ifndef KITTI_KITTIReader_H
#define KITTI_KITTIReader_H

#include "kitti_helper.h"
#include <QString>

namespace KITTI
{
    enum Timestamp
    {
        TSVelodyne,
        TSVelodyneStart,
        TSVelodyneEnd,
        TSOxTS,
        TSImages
    };

    class KITTIReader
    {
    public:
        bool ReadCalibrationCam2Cam(QVector<CamProperties> & cprops, double & cornerdist) const;
        bool ReadCalibrationIMU2Velo(Transformation3D & T) const;
        bool ReadCalibrationVelo2Cam(Transformation3D & T) const;
        bool ReadCalibration(QVector<CamProperties> & cprops, double & cornerdist, Transformation3D & imu2velo, Transformation3D & velo2cam) const;

        static bool ReadCalibrationCam2Cam(QVector<CamProperties> & cprops, double & cornerdist, const QString & fname);
        static bool ReadCalibrationIMU2Velo(Transformation3D & T, const QString & fname);
        static bool ReadCalibrationVelo2Cam(Transformation3D & T, const QString & fname);
        static bool ReadCalibration(QVector<CamProperties> & cprops, double & cornerdist, Transformation3D & imu2velo,
                                    Transformation3D & velo2cam, const QString & fname);

        bool ReadImages(QVector<TimedImage> & img, const QVector<int> & indexes, const int cam) const;
        bool ReadImages(QVector<QImage> & img, const QVector<int> & indexes, const int cam) const;

        bool ReadOxTSData(QVector<TimedOxTS> & oxts, const QVector<int> & indexes) const;
        bool ReadOxTSData(QVector<OxTS> & oxts, const QVector<int> & indexes) const;

        bool ReadVelodyneData(QVector<TimedVelo> & velo, const QVector<int> & indexes) const;
        bool ReadVelodyneData(QVector<QVector<VeloPoint> > &velo, const QVector<int> & indexes) const;

        static bool ReadTimestampFile(QStringList & ts, const QString & fname);
        bool ReadTimestampFile(QVector<double> & tstamps, Timestamp file, unsigned char cam = 0) const;

        static bool ReadOxTSFile(OxTS & data, const QString & fname);
        static bool ReadVeloFile(QVector<VeloPoint> & points, const QString & fname);

        void setFileNameLength(int length = KITTI_FILENAME_LENGTH) { fnw = length; }
        int FileNameLength() const { return fnw; }
        bool setCalibrationDir(const QString & calib_dir);
        bool setBaseDir(const QString & base_dir);

        QString & CalibrationDir() { return cdir; }
        QString & BaseDir() { return bdir; }
        QString & CalibrationCam2CamFile() { return cam2cam; }
        QString & CalibrationIMU2VeloFile() { return imu2velo; }
        QString & CalibrationVelo2CamFile() { return velo2cam; }
        const QString & CalibrationDir() const { return cdir; }
        const QString & BaseDir() const { return bdir; }
        const QString & CalibrationCam2CamFile() const { return cam2cam; }
        const QString & CalibrationIMU2VeloFile() const { return imu2velo; }
        const QString & CalibrationVelo2CamFile() const { return velo2cam; }

    private:
        QString cdir,
                bdir,
                cam2cam,
                imu2velo,
                velo2cam;
        int fnw = KITTI_FILENAME_LENGTH;

        static bool ReadRT(Transformation3D & T, const QString & fname);
        static Matrix3D List2Matrix(const QStringList & n);
        static Vector3D List2Vector3(const QStringList & n);
        static float5 List2Float5(const QStringList & n);
        static int2 List2Int2(const QStringList & n);
        static Transformation3D List2Transform(const QStringList & n);
    };
}
#endif // KITTI_KITTIReader_H
