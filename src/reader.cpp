#include "reader.h"
#include "helper_structs.h"
#include "defines.h"

#include <QFile>
#include <QTextStream>
#include <QMessageBox>
#include <QStringList>

bool Reader::Read_FromSource(QImage &ref, Matrix3D &Rref, Vector3D &tref,
                             QVector<QImage> &src, QVector<Matrix3D> &Rsrc, QVector<Vector3D> tsrc,
                             Matrix3D &K)
{
    // Load 0th image from source directory
    QString loc = "/PlaneSweep/im";
    QString refr = SOURCE_DIR;
    refr += loc;
    refr += QString::number(0);
    refr += ".png";
    if (!ref.load(refr)) return false;

    // All required camera matrices (found in 'calibration.txt')
    K = Matrix3D({
                     {0.709874*640, (1-0.977786)*640,   0.493648*640},
                     {0,            0.945744*480,       0.514782*480},
                     {0,            0,                  1}
                 });

    src.resize(9);
    Rsrc.resize(9);
    tsrc.resize(9);

    Rref = Matrix3D(    0.993701,       0.110304,   -0.0197854,
                        0.0815973,     -0.833193,   -0.546929,
                       -0.0768135,      0.541869,   -0.836945);
    tref = Vector3D(    0.280643,      -0.255355,    0.810979);

    Rsrc[0] = Matrix3D( 0.993479,       0.112002,   -0.0213286,
                        0.0822353,     -0.83349,    -0.54638,
                       -0.0789729,      0.541063,   -0.837266);
    tsrc[0] = Vector3D( 0.287891,      -0.255839,    0.808608);

    Rsrc[1] = Matrix3D( 0.993199,       0.114383,   -0.0217434,
                        0.0840021,     -0.833274,   -0.546442,
                       -0.0806218,      0.540899,   -0.837215);
    tsrc[1] = Vector3D( 0.295475,      -0.25538,     0.805906);

    Rsrc[2] = Matrix3D( 0.992928,       0.116793,   -0.0213061,
                        0.086304,      -0.833328,   -0.546001,
                       -0.081524,       0.5403,     -0.837514);
    tsrc[2] = Vector3D( 0.301659,      -0.254563,    0.804653);

    Rsrc[3] = Matrix3D( 0.992643,       0.119107,   -0.0217442,
                        0.0880017,     -0.833101,   -0.546075,
                       -0.0831565,      0.540144,   -0.837454);
    tsrc[3] = Vector3D( 0.309666,      -0.254134,    0.802222);

    Rsrc[4] = Matrix3D( 0.992429,       0.121049,   -0.0208028,
                        0.0901911,     -0.833197,   -0.545571,
                       -0.0833736,      0.539564,   -0.837806);
    tsrc[4] = Vector3D( 0.314892,      -0.253009,    0.801559);

    Rsrc[5] = Matrix3D( 0.992226,       0.122575,   -0.0215154,
                        0.0911582,     -0.833552,   -0.544869,
                       -0.0847215,      0.538672,   -0.838245);
    tsrc[5] = Vector3D( 0.32067,       -0.254142,    0.799812);

    Rsrc[6] = Matrix3D( 0.992003,       0.124427,   -0.0211509,
                        0.0930933,     -0.834508,   -0.543074,
                       -0.0852237,      0.536762,   -0.839418);
    tsrc[6] = Vector3D( 0.325942,      -0.254865,    0.799037);

    Rsrc[7] = Matrix3D( 0.991867,       0.125492,   -0.021234,
                        0.0938678,     -0.833933,   -0.543824,
                       -0.0859533,      0.537408,   -0.838931);
    tsrc[7] = Vector3D( 0.332029,      -0.252767,    0.797979);

    Rsrc[8] = Matrix3D( 0.991515,       0.128087,   -0.0221943,
                        0.095507,      -0.833589,   -0.544067,
                       -0.0881887,      0.53733,    -0.838748);
    tsrc[8] = Vector3D( 0.33934,       -0.250995,    0.796756);

    // load source images
    QString source;
    for (int i = 0; i < 9; i++){
        source = SOURCE_DIR;
        source += loc;
        source += QString::number(i + 1);
        source += ".png";
        if (!src[i].load(source)) return false;
    }

    return true;
}

bool Reader::Read_ICL_NUIM_RGB(QImage & ref, Matrix3D & Rref, Vector3D & tref, const int refn,
                               QVector<QImage> & src, QVector<Matrix3D> & Rsrc, QVector<Vector3D> & tsrc, const QVector<int> & srcn,
                               Matrix3D & K, const QString & directory, const QString & fname, const QString & format, const int digits)
{
    // get image name strings
    QString impos;
    QString imname = ImageName(impos, refn, digits, directory, fname, format);

    // initialize variables for camera parameters
    Vector3D cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint;
    double cam_angle;

    // get camera parameters
    if (!getcamParameters(impos, cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, cam_angle)) return false;
    getcamK(K, cam_dir, cam_up, cam_right);

    computeRT(Rref, tref, cam_dir, cam_pos, cam_up);

    if (!ref.load(imname)) return false;

    int nsrc = srcn.size();

    // load source views
    int half = (nsrc + 1) / 2;
    int offset;
    int loaded = 0;
    src.resize(nsrc);
    Rsrc.resize(nsrc);
    tsrc.resize(nsrc);

    for (int i = 0; i < nsrc; i++){
        if (i < half) offset = i + 1;
        else offset = half - i - 1;
        imname = ImageName(impos, srcn[loaded], digits, directory, fname, format);
        if (getcamParameters(impos, cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, cam_angle)){
            computeRT(Rsrc[loaded], tsrc[loaded], cam_dir, cam_pos, cam_up);
            if (src[loaded].load(imname)) loaded++;
        }
    }

    src.resize(loaded);
    Rsrc.resize(loaded);
    tsrc.resize(loaded);

    return true;
}

bool Reader::Read_ICL_NUIM_depth(QImage & depth, const int number, const QString & directory,
                                 const QString &fname, const QString &format, const int digits)
{
    QString impos;
    ImageName(impos, number, digits, directory, fname, format);
    QString dp = impos;
    int dot = dp.lastIndexOf('.');
    if (dot != -1) dp.truncate(dot);
    dp += ".depth";
    return depth.load(dp);
}

bool Reader::Read_TUM_RGBD_RGB(QImage & ref, Matrix3D & Rref, Vector3D & tref, const int refindex,
                               QVector<QImage> & src, QVector<Matrix3D> & Rsrc, QVector<Vector3D> & tsrc, const QVector<int> & srcindex,
                               const QString & rgbtextfile, const TUM_RGBD_line &line)
{
    return false;
}

QString Reader::ImageName(QString & imagetxt, const int number, const int digits, const QString & dir, const QString & name,
                          const QString & format)
{
    QString r = dir;
    if (!r.endsWith('/')) r += '/';
    r += name;
    int n;
    for (int i = 0; i < digits; i++) {
        n = number / (int)pow(10, digits - i - 1);
        n = n % 10;
        r += QString::number(n);
    }
    r += '.';
    imagetxt = r;
    r += format;
    imagetxt += "txt";
    return r;
}

void Reader::getcamK(Matrix3D & K, const Vector3D & cam_dir,
                     const Vector3D & cam_up, const Vector3D & cam_right)
{
    double focal = length(cam_dir);
    double aspect = length(cam_right);
    double angle = 2 * atan(aspect / 2 / focal);
    aspect = aspect / length(cam_up);

    // height and width
    int M = 480, N = 640;

    int width = N, height = M;

    // pixel size
    double psx = 2*focal*tan(0.5*angle)/N ;
    double psy = 2*focal*tan(0.5*angle)/aspect/M ;

    psx   = psx / focal;
    psy   = psy / focal;

    double Ox = (width+1)*0.5;
    double Oy = (height+1)*0.5;

    K = Matrix3D(   1.f/psx,    0.f,        Ox,
                    0.f,       -1.f/psy,    Oy,
                    0.f,        0.f,        1.f);
}

void Reader::computeRT(Matrix3D & R, Vector3D & t, const Vector3D & cam_dir,
                       const Vector3D & cam_pos, const Vector3D & cam_up)
{
    Vector3D x, y, z;

    z = cam_dir / length(cam_dir);

    x = cross(cam_up, z);
    x = normalize(x);

    y = cross(z, x);

    R = Matrix3D(x, y, z);
    R = R.trans();

    t = cam_pos;
}

bool Reader::getcamParameters(QString filename, Vector3D & cam_pos, Vector3D & cam_dir,
                              Vector3D & cam_up, Vector3D & cam_lookat,
                              Vector3D & cam_sky, Vector3D & cam_right,
                              Vector3D & cam_fpoint, double & cam_angle)
{
    // try opening file
    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading file", file.errorString());
        return false;
    }

    QTextStream in(&file);
    int first, last;

    // read all lines
    while(!in.atEnd()) {
        QString line = in.readLine();
        QString numbers = line;
        QStringList n;

        first = line.lastIndexOf('[');
        last = line.lastIndexOf(']');

        // get string between '[' and ']' and split
        if (last != -1) numbers.truncate(last);
        if (first != -1) numbers.remove(0, first + 1);
        if ((first != -1) && (last != -1)) n = numbers.split(',', QString::SkipEmptyParts);

        // find correct lines and assign camera parameter values
        if (line.startsWith(CAM_POS)){
            cam_pos = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_DIR)){
            cam_dir = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_UP)){
            cam_up = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_LOOKAT)){
            cam_lookat = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_SKY)){
            cam_sky = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_RIGHT)){
            cam_right = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_FPOINT)){
            cam_fpoint = Vector3D(n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble());
        }

        if (line.startsWith(CAM_ANGLE)){
            // no '[]' characters
            first = line.lastIndexOf('=');
            last = line.lastIndexOf(';');
            if (last != -1) numbers.truncate(last);
            if (first != -1) numbers.remove(0, first + 1);
            if ((first != -1) && (last != -1)) n = numbers.split(',', QString::SkipEmptyParts);
            cam_angle = n.at(0).trimmed().toDouble();
        }
    }

    file.close();
    return true;
}

Reader::TUM_RGBD_data Reader::Line2data(const QString & line)
{
    QStringList n;
    n = line.split(',', QString::SkipEmptyParts);
    TUM_RGBD_data d;
    d.timestamp = string2seconds(n.at(0).trimmed());
    d.t = make_float3(n.at(1).trimmed().toFloat(), n.at(2).trimmed().toFloat(), n.at(3).trimmed().toFloat());
    d.q = make_float4(n.at(4).trimmed().toFloat(), n.at(5).trimmed().toFloat(), n.at(6).trimmed().toFloat(), n.at(7).trimmed().toFloat());
    return d;
}

Reader::TUM_RGBD_data Reader::interpData(const double timestamp, const TUM_RGBD_data & data1, const TUM_RGBD_data & data2)
{
    double frac = (timestamp - data1.timestamp) / (data2.timestamp - data1.timestamp);
    TUM_RGBD_data r;
    r.timestamp = timestamp;
    r.t = lerp(data1.t, data2.t, frac);
    r.q = lerp(data1.q, data2.q, frac);
    return r;
}
