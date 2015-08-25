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
        // Load calibration using stored directories
        bool ReadCalibrationCam2Cam(QVector<CamProperties> & cprops, double & cornerdist) const;
        bool ReadCalibrationIMU2Velo(Transformation3D & T) const;
        bool ReadCalibrationVelo2Cam(Transformation3D & T) const;
        bool ReadCalibration(QVector<CamProperties> & cprops, double & cornerdist, Transformation3D & imu2velo, Transformation3D & velo2cam) const;

        // Load calibration using fname
        static bool ReadCalibrationCam2Cam(QVector<CamProperties> & cprops, double & cornerdist, const QString & fname);
        static bool ReadCalibrationIMU2Velo(Transformation3D & T, const QString & fname);
        static bool ReadCalibrationVelo2Cam(Transformation3D & T, const QString & fname);
        static bool ReadCalibration(QVector<CamProperties> & cprops, double & cornerdist, Transformation3D & imu2velo,
                                    Transformation3D & velo2cam, const QString & fname);

        // Load multiple data files
        bool ReadImages(QVector<TimedImage> & img, const QVector<int> & indexes, const int cam) const;
        bool ReadImages(QVector<QImage> & img, const QVector<int> & indexes, const int cam) const;

        bool ReadOxTSData(QVector<TimedOxTS> & oxts, const QVector<int> & indexes) const;
        bool ReadOxTSData(QVector<OxTS> & oxts, const QVector<int> & indexes) const;

        bool ReadVelodyneData(QVector<TimedVelo> & velo, const QVector<int> & indexes) const;
        bool ReadVelodyneData(QVector<QVector<VeloPoint> > &velo, const QVector<int> & indexes) const;

        // Read timestamp files from fname or stored directories
        static bool ReadTimestampFile(QStringList & ts, const QString & fname);
        static bool ReadTimestampFile(QVector<double> & ts, const QString & fname);
        bool ReadTimestampFile(QVector<double> & tstamps, Timestamp file, unsigned char cam = 0) const;
        bool ReadTimestampFile(QStringList &ts, Timestamp file, unsigned char cam = 0) const;

        // Load single files from fname or stored directories
        static bool ReadOxTSFile(OxTS & data, const QString & fname);
        bool ReadOxTSFile(OxTS & data, const int index) const { return ReadOxTSFile(data, KITTI_OXTS_NAME(bdir, index, fnw)); }

        static bool ReadVeloFile(QVector<VeloPoint> & points, const QString & fname);
        bool ReadVeloFile(QVector<VeloPoint> &points, const int index) const { return ReadVeloFile(points, KITTI_VELO_NAME(bdir, index, fnw)); }

        static bool ReadImageFile(QImage & img, const QString & fname){ return img.load(fname); }
        bool ReadImageFile(QImage &img, const unsigned char cam, const int index) const { return img.load(KITTI_IMAGE_NAME(bdir, cam, index, fnw)); }

        // set required members
        void setFileNameLength(int length = KITTI_FILENAME_LENGTH) { fnw = length; }
        int FileNameLength() const { return fnw; }
        bool setCalibrationDir(const QString & calib_dir);
        bool setBaseDir(const QString & base_dir);

        // get directories and file names stored
        QString & CalibrationDir() { return cdir; }
        QString & BaseDir() { return bdir; }
        QString & CalibrationCam2CamFileName() { return cam2cam; }
        QString & CalibrationIMU2VeloFileName() { return imu2velo; }
        QString & CalibrationVelo2CamFileName() { return velo2cam; }
        const QString & CalibrationDir() const { return cdir; }
        const QString & BaseDir() const { return bdir; }
        const QString & CalibrationCam2CamFileName() const { return cam2cam; }
        const QString & CalibrationIMU2VeloFileName() const { return imu2velo; }
        const QString & CalibrationVelo2CamFileName() const { return velo2cam; }

    protected:
        QString cdir,
                bdir,
                cam2cam,
                imu2velo,
                velo2cam;
        int fnw = KITTI_FILENAME_LENGTH;

        static bool ReadRT(Transformation3D & T, const QString & fname);
        QString timestampFileName(Timestamp file, unsigned char cam = 0) const;
        static Matrix3D List2Matrix(const QStringList & n);
        static Vector3D List2Vector3(const QStringList & n);
        static float5 List2Float5(const QStringList & n);
        static int2 List2Int2(const QStringList & n);
        static Transformation3D List2Transform(const QStringList & n);
    };
}
#endif // KITTI_KITTIReader_H
