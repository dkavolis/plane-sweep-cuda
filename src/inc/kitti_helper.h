#ifndef KITTI_HELPER_H
#define KITTI_HELPER_H

// maths
#ifdef M_PI
#   undef M_PI
#endif
#define M_PI 3.14159265358979323846264338327950288419716939937510
#define DEG_2_RAD M_PI / 180.f
#define RAD_2_DEG 180.f / M_PI
#define EARTH_RADIUS 6378137

// calibration files:
#define KITTI_CAM2CAM_NAME                              "/calib_cam_to_cam.txt"
#define KITTI_IMU2VELO_NAME                             "/calib_imu_to_velo.txt"
#define KITTI_VELO2CAM_NAME                             "/calib_velo_to_cam.txt"
#define KITTI_DIR_AND_NAME(dir, name)                   QString("%1%2").arg(dir).arg(name)
#define KITTI_CAM2CAM(calib_dir)                        KITTI_DIR_AND_NAME(calib_dir, KITTI_CAM2CAM_NAME)
#define KITTI_IMU2VELO(calib_dir)                       KITTI_DIR_AND_NAME(calib_dir, KITTI_IMU2VELO_NAME)
#define KITTI_VELO2CAM(calib_dir)                       KITTI_DIR_AND_NAME(calib_dir, KITTI_VELO2CAM_NAME)

// directories in kitti base dir
#define KITTI_OXTS_DIR                                  "/oxts"
#define KITTI_CAM_DIR(n)                                QString("/image_%1").arg(n, 2, 10, QLatin1Char('0'))
#define KITTI_VELO_DIR                                  "/velodyne_points"
#define KITTI_DATA_DIR                                  "/data"

// data file formats
#define KITTI_OXTS_FORMAT                               "txt"
#define KITTI_VELODYNE_FORMAT                           "bin"
#define KITTI_IMAGE_FORMAT                              "png"
#define KITTI_FILENAME_LENGTH                           10

// timestamp file names
#define KITTI_TIMESTAMPS                                "/timestamps.txt"
#define KITTI_TIMESTAMPS_START                          "/timestamps_start.txt"
#define KITTI_TIMESTAMPS_END                            "/timestamps_end.txt"
#define KITTI_TIMESTAMPS_NAME(base_dir, subdir, ts)     QString("%1%2%3").arg(base_dir).arg(subdir).arg(ts)
#define KITTI_CAM_TIMESTAMPS(base_dir, cam)             KITTI_TIMESTAMPS_NAME(base_dir, KITTI_CAM_DIR(cam), KITTI_TIMESTAMPS)
#define KITTI_OXTS_TIMESTAMPS(base_dir)                 KITTI_TIMESTAMPS_NAME(base_dir, KITTI_OXTS_DIR, KITTI_TIMESTAMPS)
#define KITTI_VELO_TIMESTAMPS(base_dir)                 KITTI_TIMESTAMPS_NAME(base_dir, KITTI_VELO_DIR, KITTI_TIMESTAMPS)
#define KITTI_VELO_TIMESTAMPS_START(base_dir)           KITTI_TIMESTAMPS_NAME(base_dir, KITTI_VELO_DIR, KITTI_TIMESTAMPS_START)
#define KITTI_VELO_TIMESTAMPS_END(base_dir)             KITTI_TIMESTAMPS_NAME(base_dir, KITTI_VELO_DIR, KITTI_TIMESTAMPS_END)

// data file names
#define KITTI_FILE_NAME(n,w,f)                          QString("%1.%2").arg(n,w,10,QLatin1Char('0')).arg(f)
#define KITTI_FULL_FILE_NAME(base_dir, dir, n, w, f)    QString("%1%2%3%4").arg(base_dir).arg(dir).arg(KITTI_DATA_DIR).arg(KITTI_FILE_NAME(n,w,f))
#define KITTI_IMAGE_NAME(base_dir, cam, n, w)           KITTI_FULL_FILE_NAME(base_dir, KITTI_CAM_DIR(cam), n, w, KITTI_IMAGE_FORMAT)
#define KITTI_OXTS_NAME(base_dir, n, w)                 KITTI_FULL_FILE_NAME(base_dir, KITTI_OXTS_DIR, n, w, KITTI_OXTS_FORMAT)
#define KITTI_VELO_NAME(base_dir, n, w)                 KITTI_FULL_FILE_NAME(base_dir, KITTI_VELO_DIR, n, w, KITTI_VELODYNE_FORMAT)

#include <cmath>
#include <structs.h>
#include <QImage>
#include <helper_structs.h>

namespace KITTI
{
    /**
     * @brief The OxTS struct
    lat:            latitude of the oxts-unit (deg)
    lon:            longitude of the oxts-unit (deg)
    alt:            altitude of the oxts-unit (m)
    roll:           roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
    pitch:          pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
    yaw:            heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
    vn:             velocity towards north (m/s)
    ve:             velocity towards east (m/s)
    vf:             forward velocity, i.e. parallel to earth-surface (m/s)
    vl:             leftward velocity, i.e. parallel to earth-surface (m/s)
    vu:             upward velocity, i.e. perpendicular to earth-surface (m/s)
    ax:             acceleration in x, i.e. in direction of vehicle front (m/s^2)
    ay:             acceleration in y, i.e. in direction of vehicle left (m/s^2)
    az:             acceleration in z, i.e. in direction of vehicle top (m/s^2)
    af:             forward acceleration (m/s^2)
    al:             leftward acceleration (m/s^2)
    au:             upward acceleration (m/s^2)
    wx:             angular rate around x (rad/s)
    wy:             angular rate around y (rad/s)
    wz:             angular rate around z (rad/s)
    wf:             angular rate around forward axis (rad/s)
    wl:             angular rate around leftward axis (rad/s)
    wu:             angular rate around upward axis (rad/s)
    pos_accuracy:   velocity accuracy (north/east in m)
    vel_accuracy:   velocity accuracy (north/east in m/s)
    navstat:        navigation status (see navstat_to_string)
    numsats:        number of satellites tracked by primary GPS receiver
    posmode:        position mode of primary GPS receiver (see gps_mode_to_string)
    velmode:        velocity mode of primary GPS receiver (see gps_mode_to_string)
    orimode:        orientation mode of primary GPS receiver (see gps_mode_to_string)
     */
    struct OxTS
    {
        double lat;
        double lon;
        double alt;
        double roll;
        double pitch;
        double yaw;
        double vn;
        double ve;
        double vf;
        double vl;
        double vu;
        double ax;
        double ay;
        double az;
        double af;
        double al;
        double au;
        double wx;
        double wy;
        double wz;
        double wf;
        double wl;
        double wu;
        double pos_accuracy;
        double vel_accuracy;
        int navstat;
        int numsats;
        int posmode;
        int velmode;
        int orimode;
    };

    struct TimedOxTS
    {
        OxTS data;
        double tstamp;
    };

    struct TimedImage
    {
        QImage img;
        double tstamp;
        unsigned char cam;
    };

    /**
     * @brief The Velodyne point struct
     * x, y, z - 3D point coordinates
     * r - reflectance value
     */
    struct VeloPoint
    {
        float x, y, z, r;

        VeloPoint() : x(0), y(0), z(0), r(0) {}

        VeloPoint(float x, float y, float z, float r) : x(x), y(y), z(z), r(r) {}

        VeloPoint(float3 x) : x(x.x), y(x.y), z(x.z), r(0) {}

        VeloPoint(float4 x) : x(x.x), y(x.y), z(x.z), r(x.w) {}

        VeloPoint(const VeloPoint & v) : x(v.x), y(v.y), z(v.y), r(v.y) {}

        operator float3() const { return make_float3(x, y, z); }

        operator float4() const { return make_float4(x, y, z, r); }
    };

    struct TimedVelo
    {
        QVector<VeloPoint> points;
        double  tstamp,
                tstamp_start,
                tstamp_end;
    };

    /**
     * @brief The CamProperties struct, containing data for one KITTI camera
      - S: 1x2 size of image before rectification
      - K: 3x3 calibration matrix of camera before rectification
      - D: 1x5 distortion vector of camera before rectification
      - R: 3x3 rotation matrix of camera (extrinsic)
      - T: 3x1 translation vector of camera (extrinsic)
      - S_rect: 1x2 size of image after rectification
      - R_rect: 3x3 rectifying rotation to make image planes co-planar
      - P_rect: 3x4 projection matrix after rectification
     */
    struct CamProperties
    {
        int2 S;
        Matrix3D K;
        float5 D;
        Matrix3D R;
        Vector3D T;
        int2 S_rect;
        Matrix3D R_rect;
        Transformation3D P_rect;
    };

    /**
     * @brief Compute mercator scale from latitude
     * @param lat   latitude in degrees
     * @return Mercator scale
     */
    inline double latToScale(double lat)
    {
        return cos(lat * DEG_2_RAD);
    }

    /**
     * @brief Converts lat/lon coordinates to mercator coordinates using mercator scale
     * @param mx    mercator x coordinate output
     * @param my    mercator y coordinate output
     * @param lat   latitute in degrees
     * @param lon   longitude in degrees
     * @param scale mercator scale obtained from latToScale
     */
    inline void latlonToMercator(double & mx, double & my, double lat, double lon, double scale)
    {
        mx = scale * lon * EARTH_RADIUS * DEG_2_RAD;
        my = scale * EARTH_RADIUS * log( tan((90+lat) * DEG_2_RAD / 2.f) );
    }

    /**
     * @brief String name to index conversion
     * @param str   string to convert
     * @return Index in the string
     *
     * @details \p str must end with an index of numerical characters. Inside loop will stop when it encounters non-numerical character.
     */
    inline int string2index(const QString & str)
    {
        if (str.isEmpty()) return 0;
        int i = 0, j = 0;
        while ((str.at(str.size() - 1 - j).isNumber()) || (j >= str.size())){
            i += (str.at(str.size() - 1 - j).unicode() - QChar('0').unicode()) * pow(10, j);
            j++;
        }
        return i;
    }

    /**
     * @brief Extract time in seconds from a string
     * @param date  date in string format
     * @return Time in seconds since 0:00:00 of the same day
     *
     * @details \p date must be of format <em>date hours:minutes:seconds.*</em>, where everything is optional. \a date is ignored if given.
     * If nothing was given, returns 0
     */
    inline double string2seconds(const QString & date)
    {
        if (date.isEmpty()) return 0;
        QString d = date.trimmed();
        int space = d.lastIndexOf(' ');
        if (space != -1) d.remove(0, space + 1);
        QStringList time = d.split(':', QString::SkipEmptyParts);
        double h = 0.f, m = 0.f, s = 0.f;
        int sz = time.size();
        if (sz > 2) h = time.at(sz - 3).trimmed().toDouble();
        if (sz > 1) m = time.at(sz - 2).trimmed().toDouble();
        if (sz > 0) s = time.at(sz - 1).trimmed().toDouble();
        return 3600 * h + 60 * m + s;
    }

    /**
     * @brief Calculate OxTS unit poses w.r.t. to the first one
     * @param pose  output poses, same size as \p oxts
     * @param oxts  input OxTS data, first element is used as reference
     */
    inline void convertOxtsToPose(QVector<Matrix4D> & pose, const QVector<OxTS> & oxts)
    {
        if (oxts.size() > 0){
            pose.resize(oxts.size());

            // scale from first lat value
            double scale = latToScale(oxts[0].lat);

            Matrix4D pose0inv;
            Vector3D t; Matrix3D R, Rx, Ry, Rz;
            double rx, ry, rz;

            // for all oxts:
            for (int i = 0; i < oxts.size(); i++)
            {
                // translation vector:
                latlonToMercator(rx, ry, oxts[i].lat, oxts[i].lon, scale);
                t.x = rx;
                t.y = ry;
                t.z = oxts[i].alt;

                // rotation matrix:
                rx = oxts[i].roll;
                ry = oxts[i].pitch;
                rz = oxts[i].yaw;
                Rx = Matrix3D(1.f,          0.f,        0.f,
                              0.f,          cos(rx),    -sin(rx),
                              0.f,          sin(rx),    cos(rx));
                Ry = Matrix3D(cos(ry),      0.f,        sin(ry),
                              0.f,          1.f,        0.f,
                              -sin(ry),     0.f,        cos(ry));
                Rz = Matrix3D(cos(rz),      -sin(rz),   0.f,
                              sin(rz),      cos(rz),    0.f,
                              0.f,          0.f,        1.f);
                R = Rx * Ry * Rz;

                // add pose
                pose[i] = Matrix4D(Transformation3D(R, t), make_float4(0,0,0,1));

                // normalize rotation and translation (starts at 0/0/0)
                if (i == 0) pose0inv = pose[i].inv();
                pose[i] = pose0inv * pose[i];
            }
        }
    }

    /**
     * @brief Wrap angle \p alpha in radians to range [-pi, pi]
     */
    inline double wrapToPi(double alpha)
    {
        // get modulus
        int mod = int(floorf(alpha / 2.f / M_PI));

        // wrap to [0, 2*pi]
        alpha = alpha - 2.f * M_PI * mod;

        // wrap to [-pi, pi]
        if (alpha > M_PI) alpha -= M_PI;
        return alpha;
    }

    /**
     * @brief Calculate velodyne unit to rectified camera coordinates transformations
     * @param K             extrinsic camera calibration matrix after rectification, equal for all cameras
     * @param velo2rectcam  output transformations: velodyne -> rectified camera coordinates, same size as \p cprops
     * @param velo2cam      input transformation: velodyne -> first camera coordinates
     * @param cprops        input camera properties for all cameras
     */
    inline void calculateTransforms(Matrix3D & K, QVector<Matrix4D> & velo2rectcam,
                                    const Transformation3D & velo2cam, const QVector<CamProperties> & cprops)
    {
        Matrix4D tr(velo2cam, make_float4(0,0,0,1));

        if (cprops.size() > 0){
            Matrix4D Rrect00(cprops[0].R_rect);
            Rrect00(3,3) = 1.f;

            velo2rectcam.resize(cprops.size());
            for (int i = 0; i < cprops.size(); i++){
                Matrix4D T;
                T.makeIdentity();
                T(0,3) = cprops[i].P_rect(0,3) / cprops[i].P_rect(0,0);

                velo2rectcam[i] = T * Rrect00 * tr;
            }

            K = cprops[0].P_rect.R;
        }
    }

} // namespace KITTI

#endif // KITTI_HELPER_H
