#include "kitti_reader.h"
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QMessageBox>
#include <QDir>
#include <QByteArray>
#include <QList>

namespace KITTI
{
bool KITTIReader::ReadCalibrationCam2Cam(QVector<CamProperties> & cprops, double & cornerdist, const QString & fname)
{
    // try opening file
    QFile file(fname);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading KITTI calibration file", QString("%1: %2").arg(fname).arg(file.errorString()));
        return false;
    }

    QTextStream in(&file);
    int first;
    int curr = 0;
    int max = 0;

    // read all lines
    while(!in.atEnd()) {
        QString line = in.readLine();
        QString numbers = line, name = line;
        QStringList n;

        first = line.indexOf(':');

        // get string between ':' and end and split
        if (first != -1) {
            name.truncate(first);
            numbers.remove(0, first + 1);
            curr = string2index(name.trimmed());
        }
        if (first != -1) n = numbers.split(QRegExp("\\s"), QString::SkipEmptyParts);

        if (curr >= cprops.size()) cprops.resize(curr + 1);
        if (curr > max) max = curr;

        // find correct lines and assign matrix values
        if (line.startsWith("S_")){
            if (line.startsWith("S_rect_")){
                cprops[curr].S_rect = List2Int2(n);
            }
            else{
                cprops[curr].S = List2Int2(n);
            }
        }

        if (line.startsWith("K_")){
            cprops[curr].K = List2Matrix(n);
        }

        if (line.startsWith("D_")){
            cprops[curr].D = List2Float5(n);
        }

        if (line.startsWith("R_")){
            if (line.startsWith("R_rect")){
                cprops[curr].R_rect = List2Matrix(n);
            }
            else{
                cprops[curr].R = List2Matrix(n);
            }
        }

        if (line.startsWith("T_")){
            cprops[curr].T = List2Vector3(n);
        }

        if (line.startsWith("P_rect_")){
            cprops[curr].P_rect = List2Transform(n);
        }

        if (line.startsWith("corner_dist")){
            cornerdist = n.at(0).trimmed().toDouble();
        }
    }

    cprops.resize(max + 1);

    file.close();
    return true;
}

bool KITTIReader::ReadCalibrationIMU2Velo(Transformation3D & T, const QString & fname)
{
    return ReadRT(T, fname);
}

bool KITTIReader::ReadCalibrationVelo2Cam(Transformation3D & T, const QString & fname)
{
    return ReadRT(T, fname);
}

bool KITTIReader::ReadCalibration(QVector<CamProperties> & cprops, double & cornerdist, Transformation3D & imu2velo, Transformation3D & velo2cam,
                                  const QString & fname)
{
    bool s = true;
    if (!ReadCalibrationCam2Cam(cprops, cornerdist, fname)) s = false;
    if (!ReadCalibrationIMU2Velo(imu2velo, fname)) s = false;
    if (!ReadCalibrationVelo2Cam(velo2cam, fname)) s = false;
    return s;
}

bool KITTIReader::ReadCalibrationCam2Cam(QVector<CamProperties> & cprops, double & cornerdist) const
{
    return ReadCalibrationCam2Cam(cprops, cornerdist, KITTI_CAM2CAM(cdir));
}

bool KITTIReader::ReadCalibrationIMU2Velo(Transformation3D & T) const
{
    return ReadRT(T, KITTI_IMU2VELO(cdir));
}

bool KITTIReader::ReadCalibrationVelo2Cam(Transformation3D & T) const
{
    return ReadRT(T, KITTI_VELO2CAM(cdir));
}

bool KITTIReader::ReadCalibration(QVector<CamProperties> & cprops, double & cornerdist, Transformation3D & imu2velo, Transformation3D & velo2cam) const
{
    bool s = true;
    if (!ReadCalibrationCam2Cam(cprops, cornerdist)) s = false;
    if (!ReadCalibrationIMU2Velo(imu2velo)) s = false;
    if (!ReadCalibrationVelo2Cam(velo2cam)) s = false;
    return s;
}

bool KITTIReader::ReadImages(QVector<TimedImage> & img, const QVector<int> & indexes, const int cam) const
{
    img.resize(indexes.size());
    bool success = true, tss = true;
    QStringList list;

    if (!ReadTimestampFile(list, KITTI_CAM_TIMESTAMPS(bdir, cam))) tss = false;

    for (int j = 0; j < indexes.size(); j++)
    {
        img[j].cam = cam;
        if (!img[j].img.load(KITTI_IMAGE_NAME(bdir, cam, indexes[j], fnw))) success = false;
        if (tss) img[j].tstamp = string2seconds(list.at(indexes[j]));
    }

    return success & tss;
}

bool KITTIReader::ReadImages(QVector<QImage> & img, const QVector<int> & indexes, const int cam) const
{
    img.resize(indexes.size());
    bool success = true;

    for (int j = 0; j < indexes.size(); j++)
    {
        if (!img[j].load(KITTI_IMAGE_NAME(bdir, cam, indexes[j], fnw))) success = false;
    }

    return success;
}

bool KITTIReader::ReadOxTSData(QVector<TimedOxTS> & oxts, const QVector<int> & indexes) const
{
    oxts.resize(indexes.size());
    bool success = true, tss = true;

    QStringList list;

    if (!ReadTimestampFile(list, KITTI_OXTS_TIMESTAMPS(bdir))) tss = false;

    for (int j = 0; j < indexes.size(); j++)
    {
        if (!ReadOxTSFile(oxts[j].data, KITTI_OXTS_NAME(bdir, indexes[j], fnw))) success = false;
        if (tss) oxts[j].tstamp = string2seconds(list.at(indexes[j]));
    }

    return success & tss;
}

bool KITTIReader::ReadOxTSData(QVector<OxTS> & oxts, const QVector<int> & indexes) const
{
    oxts.resize(indexes.size());
    bool success = true;

    for (int j = 0; j < indexes.size(); j++)
    {
        if (!ReadOxTSFile(oxts[j], KITTI_OXTS_NAME(bdir, indexes[j], fnw))) success = false;
    }

    return success;
}

bool KITTIReader::ReadVelodyneData(QVector<TimedVelo> & velo, const QVector<int> & indexes) const
{
    velo.resize(indexes.size());
    bool success = true, tss1 = true, tss2 = true, tss3 = true;

    QStringList list, liststart, listend;

    if (!ReadTimestampFile(list, KITTI_VELO_TIMESTAMPS(bdir))) tss1 = false;
    if (!ReadTimestampFile(liststart, KITTI_VELO_TIMESTAMPS_START(bdir))) tss2 = false;
    if (!ReadTimestampFile(listend, KITTI_VELO_TIMESTAMPS_END(bdir))) tss3 = false;

    for (int j = 0; j < indexes.size(); j++)
    {
        if (!ReadVeloFile(velo[j].points, KITTI_VELO_NAME(bdir, indexes[j], fnw))) success = false;
        if (tss1) velo[j].tstamp = string2seconds(list.at(indexes[j]));
        if (tss2) velo[j].tstamp_start = string2seconds(liststart.at(indexes[j]));
        if (tss3) velo[j].tstamp_end = string2seconds(listend.at(indexes[j]));
    }

    return success & tss1 & tss2 & tss3;
}

bool KITTIReader::ReadVelodyneData(QVector<QVector<VeloPoint>> & velo, const QVector<int> & indexes) const
{
    velo.resize(indexes.size());
    bool success = true;

    for (int j = 0; j < indexes.size(); j++)
    {
        if (!ReadVeloFile(velo[j], KITTI_VELO_NAME(bdir, indexes[j], fnw))) success = false;
    }

    return success;
}

bool KITTIReader::ReadRT(Transformation3D & T, const QString & fname)
{
    // try opening file
    QFile file(fname);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading KITTI calibration file", QString("%1: %2").arg(fname).arg(file.errorString()));
        return false;
    }

    QTextStream in(&file);
    int first;

    // read all lines
    while(!in.atEnd()) {
        QString line = in.readLine();
        QString numbers = line;
        QStringList n;

        first = line.indexOf(':');

        // get string between ':' and end and split
        if (first != -1) numbers.remove(0, first + 1);
        if (first != -1) n = numbers.split(QRegExp("\\s"), QString::SkipEmptyParts);

        // find correct lines and assign matrix values
        if (line.startsWith("R:")){
            T.R = List2Matrix(n);
        }

        if (line.startsWith("T:")){
            T.T = List2Vector3(n);
        }
    }

    file.close();
    return true;
}

bool KITTIReader::ReadTimestampFile(QStringList &ts, const QString &fname)
{
    QFile file(fname);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading KITTI timestamp file", QString("%1: %2").arg(fname).arg(file.errorString()));
        return false;
    }

    QTextStream in(&file);
    QString stamps = in.readAll();
    ts = stamps.split(QRegExp("[\r\n]"),QString::SkipEmptyParts);
    return true;
}

bool KITTIReader::ReadTimestampFile(QVector<double> &ts, const QString &fname)
{
    QStringList list;
    bool success = ReadTimestampFile(list, fname);

    ts.resize(list.size());
    for (int i = 0; i < ts.size(); i++) ts[i] = string2seconds(list.at(i).trimmed());

    return success;
}

bool KITTIReader::ReadTimestampFile(QVector<double> & tstamps, Timestamp file, unsigned char cam) const
{
    return ReadTimestampFile(tstamps, timestampFileName(file, cam));
}

bool KITTIReader::ReadTimestampFile(QStringList &ts, Timestamp file, unsigned char cam) const
{
    return ReadTimestampFile(ts, timestampFileName(file, cam));
}

bool KITTIReader::ReadOxTSFile(OxTS &data, const QString &fname)
{
    QFile file(fname);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading KITTI OxTS file", QString("%1: %2").arg(fname).arg(file.errorString()));
        return false;
    }

    QTextStream in(&file);
    QString d = in.readAll();
    QStringList n = d.split(QRegExp("\\s"), QString::SkipEmptyParts);

    data.lat = n.at(0).trimmed().toDouble();
    data.lon = n.at(1).trimmed().toDouble();
    data.alt = n.at(2).trimmed().toDouble();
    data.roll = n.at(3).trimmed().toDouble();
    data.pitch = n.at(4).trimmed().toDouble();
    data.yaw = n.at(5).trimmed().toDouble();
    data.vn = n.at(6).trimmed().toDouble();
    data.ve = n.at(7).trimmed().toDouble();
    data.vf = n.at(8).trimmed().toDouble();
    data.vl = n.at(9).trimmed().toDouble();
    data.vu = n.at(10).trimmed().toDouble();
    data.ax = n.at(11).trimmed().toDouble();
    data.ay = n.at(12).trimmed().toDouble();
    data.az = n.at(13).trimmed().toDouble();
    data.af = n.at(14).trimmed().toDouble();
    data.al = n.at(15).trimmed().toDouble();
    data.au = n.at(16).trimmed().toDouble();
    data.wx = n.at(17).trimmed().toDouble();
    data.wy = n.at(18).trimmed().toDouble();
    data.wz = n.at(19).trimmed().toDouble();
    data.wf = n.at(20).trimmed().toDouble();
    data.wl = n.at(21).trimmed().toDouble();
    data.wu = n.at(22).trimmed().toDouble();
    data.pos_accuracy = n.at(23).trimmed().toDouble();
    data.vel_accuracy = n.at(24).trimmed().toDouble();
    data.navstat = n.at(25).trimmed().toInt();
    data.numsats = n.at(26).trimmed().toInt();
    data.posmode = n.at(27).trimmed().toInt();
    data.velmode = n.at(28).trimmed().toInt();
    data.orimode = n.at(29).trimmed().toInt();

    return true;
}

bool KITTIReader::ReadVeloFile(QVector<VeloPoint> &points, const QString &fname)
{
    QFile file(fname);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading KITTI Velodyne file", QString("%1: %2").arg(fname).arg(file.errorString()));
        return false;
    }

    QByteArray data = file.readAll();
    float *px = (float *)data.data()+0;
    float *py = (float *)data.data()+1;
    float *pz = (float *)data.data()+2;
    float *pr = (float *)data.data()+3;
    points.resize(data.size() / 4 / 4);
    for (int i = 0; i < points.size(); i++)
    {
        points[i] = VeloPoint(*px, *py, *pz, *pr);
        px+=4; py+=4; pz+=4; pr+=4;
    }

    file.close();
    return true;
}

bool KITTIReader::setCalibrationDir(const QString & calib_dir)
{
    if (!QDir(calib_dir).exists()) return false;
    if (calib_dir.endsWith('/')) cdir = calib_dir.left(calib_dir.size() - 1);
    else cdir = calib_dir;
    cam2cam = KITTI_CAM2CAM(cdir);
    imu2velo = KITTI_IMU2VELO(cdir);
    velo2cam = KITTI_VELO2CAM(cdir);
    return true;
}

bool KITTIReader::setBaseDir(const QString & base_dir)
{
    if (!QDir(base_dir).exists()) return false;
    if (base_dir.endsWith('/')) bdir = base_dir.left(base_dir.size() - 1);
    else bdir = base_dir;
    return true;
}

Matrix3D KITTIReader::List2Matrix(const QStringList & n)
{
    return Matrix3D(n.at(0).trimmed().toFloat(),
                    n.at(1).trimmed().toFloat(),
                    n.at(2).trimmed().toFloat(),
                    n.at(3).trimmed().toFloat(),
                    n.at(4).trimmed().toFloat(),
                    n.at(5).trimmed().toFloat(),
                    n.at(6).trimmed().toFloat(),
                    n.at(7).trimmed().toFloat(),
                    n.at(8).trimmed().toFloat());
}

Vector3D KITTIReader::List2Vector3(const QStringList & n)
{
    return Vector3D(n.at(0).trimmed().toFloat(),
                    n.at(1).trimmed().toFloat(),
                    n.at(2).trimmed().toFloat());
}

float5 KITTIReader::List2Float5(const QStringList & n)
{
    return make_float5(n.at(0).trimmed().toFloat(),
                       n.at(1).trimmed().toFloat(),
                       n.at(2).trimmed().toFloat(),
                       n.at(3).trimmed().toFloat(),
                       n.at(4).trimmed().toFloat());
}

int2 KITTIReader::List2Int2(const QStringList & n)
{
    return make_int2((int)n.at(0).trimmed().toFloat(),
                     (int)n.at(1).trimmed().toFloat());
}

Transformation3D KITTIReader::List2Transform(const QStringList & n)
{
    Transformation3D t;
    t.R = Matrix3D(n.at(0).trimmed().toFloat(),
                   n.at(1).trimmed().toFloat(),
                   n.at(2).trimmed().toFloat(),
                   n.at(4).trimmed().toFloat(),
                   n.at(5).trimmed().toFloat(),
                   n.at(6).trimmed().toFloat(),
                   n.at(8).trimmed().toFloat(),
                   n.at(9).trimmed().toFloat(),
                   n.at(10).trimmed().toFloat());
    t.T = Vector3D(n.at(3).trimmed().toFloat(),
                   n.at(7).trimmed().toFloat(),
                   n.at(11).trimmed().toFloat());
    return t;
}

QString KITTIReader::timestampFileName(Timestamp file, unsigned char cam) const
{
    switch (file){
    case TSVelodyne :
        return KITTI_VELO_TIMESTAMPS(bdir);

    case TSVelodyneEnd:
        return KITTI_VELO_TIMESTAMPS_END(bdir);

    case TSVelodyneStart:
        return KITTI_VELO_TIMESTAMPS_START(bdir);

    case TSOxTS:
        return KITTI_OXTS_TIMESTAMPS(bdir);

    case TSImages:
        return KITTI_CAM_TIMESTAMPS(bdir, cam);
    }

    return "";
}

} // namespace KITTI
