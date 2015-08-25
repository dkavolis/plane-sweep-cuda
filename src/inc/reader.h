#ifndef READER_H
#define READER_H

#include "structs.h"
#include <QImage>
#include "kitti_helper.h"

class Reader
{
public:
    /** \brief Structure holding a single line of TUM-RGBD format image data */
    struct TUM_RGBD_data{
        double timestamp;   // time
        float3 t;           // translation vector
        float4 q;           // quaternion
    };

    /** \brief Enum indicating TUM-RGBD property indexes in line */
    enum TUM_RGBD_property
    {
        timestamp = 0,      // timestamp field
        RGB = 1,            // rgb image file name field
        depth = 2           // depth image file name field
    };

    /** \brief Struct containing TUM-RGBD line properties */
    struct TUM_RGBD_line
    {
        unsigned char nfields;      // number of fields
        TUM_RGBD_property prop[3];  // properties in each field
    };

    /** \brief Struct containing TUM-RGBD file names and their properties */
    struct TUM_RGBD_file
    {
        QString         RGBfname,   // rgb image file name
                        depthfname; // depth image file name
        TUM_RGBD_data   RGBdata,    // rgb image data
                        depthdata;  // depth image data
    };

    static bool Read_FromSource(QImage & ref, Matrix3D & Rref, Vector3D & tref,
                                QVector<QImage> & src, QVector<Matrix3D> & Rsrc, QVector<Vector3D> tsrc,
                                Matrix3D & K);
    static bool Read_ICL_NUIM_RGB(QImage & ref, Matrix3D & Rref, Vector3D & tref, const int refn,
                                  QVector<QImage> & src, QVector<Matrix3D> & Rsrc, QVector<Vector3D> & tsrc, const QVector<int> & srcn,
                                  Matrix3D & K, const QString & directory, const QString &fname, const QString &format, const int digits);
    static bool Read_ICL_NUIM_depth(QImage & depth, const int number, const QString & directory,
                                    const QString &fname, const QString &format, const int digits);

    /** @brief ONLY A PLACEHOLDER AT THE TIME, STILL EMPTY */
    static bool Read_TUM_RGBD_RGB(QImage & ref, Matrix3D & Rref, Vector3D & tref, const int refindex,
                                  QVector<QImage> & src, QVector<Matrix3D> & Rsrc, QVector<Vector3D> & tsrc, const QVector<int> & srcindex,
                                  const QString & rgbtextfile, const TUM_RGBD_line & line);

    /**
    *  \brief Concatenate strings and \p number to create image file name
    *
    *  \param imagetxt  string of image \a txt file
    *  \param number    image number
    *  \param digits    number of digits that make up number in a file name
    *  \param dir       directory where image is located
    *  \param name      image file name before number
    *  \param format    format of the image
    *  \return Image file name
    */
    static QString ImageName(QString & imagetxt, const int number, const int digits, const QString & dir, const QString & name,
                             const QString & format);

    /**
    *  \brief Compute camera calibration matrix \f$K\f$
    *
    *  \param K         camera calibration matrix \f$K\f$
    *  \param cam_dir   camera parameter
    *  \param cam_up    camera parameter
    *  \param cam_right camera parameter
    *
    *  \details Camera parameters must first be obtained via getcamParameters().
    */
    static void getcamK(Matrix3D & K, const Vector3D & cam_dir,
                        const Vector3D & cam_up, const Vector3D & cam_right);

    /**
    *  \brief Compute rotation matrix and translation vector
    *
    *  \param R       rotation matrix
    *  \param t       translation vector
    *  \param cam_dir camera parameter
    *  \param cam_pos camera parameter
    *  \param cam_up  camera parameter
    *
    *  \details Camera parameters must first be obtained via getcamParameters
    */
    static void computeRT(Matrix3D & R, Vector3D & t, const Vector3D & cam_dir,
                          const Vector3D & cam_pos, const Vector3D & cam_up);

    /**
    *  \brief Get cam parameters from \a txt file
    *
    *  \param filename   name of \a .txt file, including extension
    *  \param cam_pos    camera parameter
    *  \param cam_dir    camera parameter
    *  \param cam_up     camera parameter
    *  \param cam_lookat camera parameter
    *  \param cam_sky    camera parameter
    *  \param cam_right  camera parameter
    *  \param cam_fpoint camera parameter
    *  \param cam_angle  camera parameter
    *  \return Success/failure of opening \a filename
    *
    *  \details Must be the same format as ones found on http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
    */
    static bool getcamParameters(QString filename, Vector3D & cam_pos, Vector3D & cam_dir,
                                 Vector3D & cam_up, Vector3D & cam_lookat,
                                 Vector3D & cam_sky, Vector3D & cam_right,
                                 Vector3D & cam_fpoint, double & cam_angle);

    static inline double string2seconds(const QString & date)
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

    static TUM_RGBD_data Line2data(const QString & line);
    static TUM_RGBD_data interpData(const double timestamp, const TUM_RGBD_data & data1, const TUM_RGBD_data & data2);
};

#endif // READER_H
