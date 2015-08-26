#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include "pclviewer.h"
#include "ui_pclviewer.h"
#include <iostream>
#include <QPixmap>
#include <QColor>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QMessageBox>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <chrono>
#include <QVector>
#include <QRgb>
#include <dev_functions.h>

PCLViewer::PCLViewer (int argc, char **argv, QWidget *parent) :
    QMainWindow (parent),
    ui (new Ui::PCLViewer),
    scene (new QGraphicsScene()),
    depthscene (new QGraphicsScene()),
    dendepthsc (new QGraphicsScene()),
    tgvscene (new QGraphicsScene()),
    ps(argc, argv),
    fd(DEFAULT_FUSION_VOXELS_X, DEFAULT_FUSION_VOXELS_Y, DEFAULT_FUSION_VOXELS_Z),
    f(DEFAULT_FUSION_VOXELS_X, DEFAULT_FUSION_VOXELS_Y, DEFAULT_FUSION_VOXELS_Z)
{
//    kitti.setBaseDir("D:/Software/2011_09_26_drive_0091_sync/2011_09_26/2011_09_26_drive_0091_sync");
//    kitti.setCalibrationDir("D:/Software/2011_09_26_calib/2011_09_26");

    std::cout << "\nSize of fusion data = " << fd.sizeMBytes() << "MB\n\n";

//    printf("kitti dirs:\ncam_to_cam: %s\nvelo_to_cam: %s\nimu_to_velo: %s\nbase dir: %s\n",
//           kitti.CalibrationCam2CamFileName().toStdString().c_str(),
//           kitti.CalibrationVelo2CamFileName().toStdString().c_str(),
//           kitti.CalibrationIMU2VeloFileName().toStdString().c_str(),
//           kitti.BaseDir().toStdString().c_str());
//    fflush(stdout);

    ui->setupUi (this);
    this->setWindowTitle ("3D Reconstruction Program");

    setupPlanesweep();
    setupTGV();
    setupFusion();

    LoadImages();
    uchar3 c;
    QColor col;

    ctable.resize(256);
    for (int i = 0; i < 256; i++){
        c = RGBdepthmapColor(i);
        col.setRed(c.x);
        col.setGreen(c.y);
        col.setBlue(c.z);
        ctable[i] = col.rgb();
    }

    ui->cbardenoised->setColorTable(ctable);
    ui->cbar->setColorTable(ctable);
    ui->cbarTGV->setColorTable(ctable);

    connect(ui->cbar, SIGNAL(selected(double)), this, SLOT(colorbar_selected(double)));
    connect(ui->cbardenoised, SIGNAL(selected(double)), this, SLOT(colorbar_selected(double)));
    connect(ui->cbarTGV, SIGNAL(selected(double)), this, SLOT(colorbar_selected(double)));
}

void PCLViewer::colorbar_selected(double value)
{
    printf("Value of colorbar clicked: %f\n\n", value);
    fflush(stdout);
}

void PCLViewer::setupPlanesweep()
{
    // Setup the raw planesweep cloud pointer
    cloud.reset (new PointCloudT);

    // Set up the raw planesweep QVTK window
    viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
    viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
    ui->qvtkWidget->update ();

    // Setup the tvl1 denoised cloud pointer
    clouddenoised.reset (new PointCloudT);

    // Set up the tvl1 denoised QVTK window
    viewerdenoised.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkDenoised->SetRenderWindow (viewerdenoised->getRenderWindow ());
    viewerdenoised->setupInteractor (ui->qvtkDenoised->GetInteractor (), ui->qvtkDenoised->GetRenderWindow ());
    ui->qvtkDenoised->update ();

    connect(ui->pSlider, SIGNAL(valueChanged(int)), this, SLOT(pSliderValueChanged(int)));

    // Add point clouds to widgets
    viewer->addPointCloud (cloud, "cloud");
    pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();

    viewerdenoised->addPointCloud (clouddenoised, "cloud");
    on_pSlider2_valueChanged (2);
    viewerdenoised->resetCamera ();
    ui->qvtkDenoised->update ();

    ui->depthview->setScene(depthscene);

    // Setup widget values
    ui->imNumber->setValue(ps.getNumberofImages());
    ui->winSize->setValue((ps.getWindowSize()));
    ui->nccThresh->setValue(ps.getNCCthreshold());
    ui->stdThresh->setValue(ps.getSTDthreshold());
    ui->numberPlanes->setValue(ps.getNumberofPlanes());
    ui->zNear->setValue(ps.getZnear());
    ui->zFar->setValue(ps.getZfar());
    ui->stdThresh->setMaximum(255 * 1.f);

    ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
    ui->threadsx->setMaximum(ui->maxthreads->value());
    ui->threadsy->setMaximum(ui->maxthreads->value());
    dim3 t = ps.getThreadsPerBlock();
    ui->threadsx->setValue(t.x);
    ui->threadsy->setValue(t.y);

    ui->cbar->setNumberOfTicks(11);
    ui->cbar->setRange(ui->zNear->value(), ui->zFar->value());

    ui->altmethod->setChecked(ps.getAlternativeRelativeMatrixMethod());

    QChar sigma(0x03C3), tau(0x03C4), beta(0x03B2), gamma(0x03B3), theta(0x03B8);

    // Setup correct greek letters in labels
    ui->tvl1_lambda_label->setText( trUtf8( "\xce\xbb" ) );
    ui->tvl1_sigma_label->setText(sigma);
    ui->tvl1_tau_label->setText(tau);
    ui->tvl1_theta_label->setText(theta);
    ui->tvl1_beta_label->setText(beta);
    ui->tvl1_gamma_label->setText(gamma);

    // Setup widget values
    ui->lambda->setValue(DEFAULT_TVL1_LAMBDA);
    ui->nIters->setValue(DEFAULT_TVL1_ITERATIONS);
    ui->tvl1_sigma->setValue(DEFAULT_TVL1_SIGMA);
    ui->tvl1_tau->setValue(DEFAULT_TVL1_TAU);
    ui->tvl1_theta->setValue(DEFAULT_TVL1_THETA);
    ui->tvl1_beta->setValue(DEFAULT_TVL1_BETA);
    ui->tvl1_gamma->setValue(DEFAULT_TVL1_GAMMA);

    ui->cbardenoised->setNumberOfTicks(11);
    ui->cbardenoised->setRange(ui->zNear->value(), ui->zFar->value());
}

void PCLViewer::setupTGV()
{
    // Setup the TGV cloud pointer
    cloudtgv.reset (new PointCloudT);

    // Set up the TGV QVTK window
    viewertgv.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtktgv->SetRenderWindow (viewertgv->getRenderWindow ());
    viewertgv->setupInteractor (ui->qvtktgv->GetInteractor (), ui->qvtktgv->GetRenderWindow ());
    ui->qvtktgv->update ();

    // Add point cloud to widget
    viewertgv->addPointCloud (cloudtgv, "cloud");
    ui->tgv_psize->setValue(2);
    viewertgv->resetCamera ();
    ui->qvtktgv->update ();

    QChar alpha = QChar(0xb1, 0x03);
    QChar sigma(0x03C3), tau(0x03C4), beta(0x03B2), gamma(0x03B3);

    // Setup labels with greek letters
    QString a0 = alpha, a1 = alpha;
    a0 += "<sub>";
    a0 += QString::number(0);
    a0 += "</sub>";
    a1 += "<sub>";
    a1 += QString::number(1);
    a1 += "</sub>";
    ui->alpha0_label->setText(a0);
    ui->alpha1_label->setText(a1);
    ui->tgv_lambda_label->setText(trUtf8( "\xce\xbb" ));
    ui->tgv_sigma_label->setText(sigma);
    ui->tgv_tau_label->setText(tau);
    ui->tgv_beta_label->setText(beta);
    ui->tgv_gamma_label->setText(gamma);

    // Setup values
    ui->tgv_alpha0->setValue(DEFAULT_TGV_ALPHA0);
    ui->tgv_alpha1->setValue(DEFAULT_TGV_ALPHA1);
    ui->tgv_lambda->setValue(DEFAULT_TGV_LAMBDA);
    ui->tgv_niters->setValue(DEFAULT_TGV_NITERS);
    ui->tgv_warps->setValue(DEFAULT_TGV_NWARPS);
    ui->tgv_sigma->setValue(DEFAULT_TGV_SIGMA);
    ui->tgv_tau->setValue(DEFAULT_TGV_TAU);
    ui->tgv_beta->setValue(DEFAULT_TGV_BETA);
    ui->tgv_gamma->setValue(DEFAULT_TGV_GAMMA);

    ui->cbarTGV->setNumberOfTicks(11);
    ui->cbarTGV->setRange(ui->zNear->value(), ui->zFar->value());
}

void PCLViewer::setupFusion()
{
    // Setup the fusion cloud pointer
    cloudfusion.reset (new PointCloudT);

    // Set up the fusion QVTK window
    viewerfusion.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkfusion->SetRenderWindow (viewerfusion->getRenderWindow ());
    viewerfusion->setupInteractor (ui->qvtkfusion->GetInteractor (), ui->qvtkfusion->GetRenderWindow ());
    ui->qvtkfusion->update ();

    // Add cloud
    viewerfusion->addPointCloud (cloudtgv, "cloud");
    on_fusion_psize_valueChanged(2);
    viewerfusion->resetCamera ();
    ui->qvtkfusion->update ();

    QChar sigma(0x03C3), tau(0x03C4);

    // Setup labels with greek letters
    ui->fusion_lambda_label->setText(trUtf8( "\xce\xbb" ));
    ui->fusion_sigma_label->setText(sigma);
    ui->fusion_tau_label->setText(tau);

    // Connect volume coordinate widgets to correct slots
    connect(ui->fusion_volx1, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_corner_x_changed()));
    connect(ui->fusion_voly1, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_corner_y_changed()));
    connect(ui->fusion_volz1, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_corner_z_changed()));
    connect(ui->fusion_volx2, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_corner_x_changed()));
    connect(ui->fusion_voly2, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_corner_y_changed()));
    connect(ui->fusion_volz2, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_corner_z_changed()));

    connect(ui->fusion_cx, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_center_x_changed()));
    connect(ui->fusion_cy, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_center_y_changed()));
    connect(ui->fusion_cz, SIGNAL(valueChanged(double)), this, SLOT(fusion_volume_center_z_changed()));

    // Setup values
    ui->fusion_d->setValue(fd.depth());
    ui->fusion_h->setValue(fd.height());
    ui->fusion_w->setValue(fd.width());
    ui->fusion_imstep->setValue(DEFAULT_FUSION_IMSTEP);
    ui->fusion_iters->setValue(DEFAULT_FUSION_ITERATIONS);
    ui->fusion_lambda->setValue(DEFAULT_FUSION_LAMBDA);
    ui->fusion_sigma->setValue(DEFAULT_FUSION_SIGMA);
    ui->fusion_tau->setValue(DEFAULT_FUSION_TAU);
    ui->fusion_threshold->setValue(DEFAULT_FUSION_SD_THRESHOLD);
    ui->fusion_volx1->setValue(DEFAULT_FUSION_VOLUME_X1);
    ui->fusion_voly1->setValue(DEFAULT_FUSION_VOLUME_Y1);
    ui->fusion_volz1->setValue(DEFAULT_FUSION_VOLUME_Z1);
    ui->fusion_volx2->setValue(DEFAULT_FUSION_VOLUME_X2);
    ui->fusion_voly2->setValue(DEFAULT_FUSION_VOLUME_Y2);
    ui->fusion_volz2->setValue(DEFAULT_FUSION_VOLUME_Z2);

    ui->fusion_threadsw->setMaximum(ui->maxthreads->value());
    ui->fusion_threadsh->setMaximum(ui->maxthreads->value());
    ui->fusion_threadsd->setMaximum(ui->maxthreads->value());

    ui->fusion_threadsw->setValue(DEFAULT_FUSION_THREADS_X);
    ui->fusion_threadsh->setValue(DEFAULT_FUSION_THREADS_Y);
    ui->fusion_threadsd->setValue(ui->maxthreads->value() / ui->fusion_threadsw->value() / ui->fusion_threadsh->value());
}

void PCLViewer::fusion_volume_center_x_changed()
{
    // Translate corners
    float t;
    t = ui->fusion_cx->value() - (ui->fusion_volx1->value() + ui->fusion_volx2->value()) / 2.f;

    ui->fusion_volx1->setValue(ui->fusion_volx1->value() + t);
    ui->fusion_volx2->setValue(ui->fusion_volx2->value() + t);
}

void PCLViewer::fusion_volume_center_y_changed()
{
    // Translate corners
    float t;
    t = ui->fusion_cy->value() - (ui->fusion_voly1->value() + ui->fusion_voly2->value()) / 2.f;

    ui->fusion_voly1->setValue(ui->fusion_voly1->value() + t);
    ui->fusion_voly2->setValue(ui->fusion_voly2->value() + t);
}

void PCLViewer::fusion_volume_center_z_changed()
{
    // Translate corners
    float t;
    t = ui->fusion_cz->value() - (ui->fusion_volz1->value() + ui->fusion_volz2->value()) / 2.f;

    ui->fusion_volz1->setValue(ui->fusion_volz1->value() + t);
    ui->fusion_volz2->setValue(ui->fusion_volz2->value() + t);
}

void PCLViewer::fusion_volume_corner_x_changed()
{
    // Recalculate new center coordinates
    ui->fusion_cx->setValue((ui->fusion_volx1->value() + ui->fusion_volx2->value()) / 2.f);
}

void PCLViewer::fusion_volume_corner_y_changed()
{
    // Recalculate new center coordinate
    ui->fusion_cy->setValue((ui->fusion_voly1->value() + ui->fusion_voly2->value()) / 2.f);
}

void PCLViewer::fusion_volume_corner_z_changed()
{
    // Recalculate new center coordinate
    ui->fusion_cz->setValue((ui->fusion_volz1->value() + ui->fusion_volz2->value()) / 2.f);
}

void
PCLViewer::pSliderValueChanged (int value)
{
    ui->cloudSize->setValue(value);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtkWidget->update ();
}

void PCLViewer::LoadImages()
{
    // Load 0th image from source directory
    QString loc = "/PlaneSweep/im";
    QString ref = SOURCE_DIR;
    ref += loc;
    ref += QString::number(0);
    ref += ".png";
    refim.load(ref);
    int w = refim.width(), h = refim.height();

    // show the reference image
    image = QPixmap::fromImage(refim);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->refView->setScene(scene);

    // All required camera matrices (found in 'calibration.txt')
    Matrix3D K( 0.709874*640, (1-0.977786)*640,   0.493648*640,
                0,            0.945744*480,       0.514782*480,
                0,            0,                  1);

    ps.setK(K);
    QVector<Matrix3D> Rsrc(9);
    QVector<Vector3D> tsrc(9);
    Matrix3D Rref;
    Vector3D tref;

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

    // setup reference image
    ps.HostRef.setSize(w, h);
    rgb2gray<float>(ps.HostRef.data, refim);
    ps.HostRef.R = Rref; ps.HostRef.t = tref;

    // setup source images
    QString src;
    QImage sources;
    ps.HostSrc.resize(9);
    for (int i = 0; i < 9; i++){
        src = SOURCE_DIR;
        src += loc;
        src += QString::number(i + 1);
        src += ".png";
        sources.load(src);
        ps.HostSrc[i].setSize(w,h);
        rgb2gray<float>(ps.HostSrc[i].data, sources);
        ps.HostSrc[i].R = Rsrc[i]; ps.HostSrc[i].t = tsrc[i];
    }

    //ps.Convert8uTo32f(argc, argv);
}

PCLViewer::~PCLViewer ()
{
    // free resources
    delete[] argv;
    delete ui;
}

void PCLViewer::on_pushButton_pressed()
{
    if (ps.RunAlgorithm(argc, argv)){
        // get depthmaps
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        depth = ps.getDepthmap();
        depth8u = ps.getDepthmap8u();

        // resize cloud if reference image has changed
        if (refchanged) {
            if ((cloud->height != depth8u->height) || (cloud->width != depth8u->width))
            {
                // The number of points in the cloud
                cloud->points.resize(depth8u->width * depth8u->height);
                cloud->width = depth8u->width;
                cloud->height = depth8u->height;
            }
        }

        QColor c;
        int  i;
        Matrix3D k = ps.getInverseK();
        double z;

        // update colorbar range
        ui->cbar->setRangeMin(ui->zNear->value());
        ui->cbar->setRangeMax(ui->zFar->value());

        // Fill the cloud with points
        for (size_t x = 0; x < depth8u->width; ++x)
            for (size_t y = 0; y < depth8u->height; ++y)
            {

                i = x + y * depth8u->width;
                z = depth->data[i];

                cloud->points[i].z = sign(k(1,1)) * z;
                cloud->points[i].x = z * (k(0,0) * x + k(0,1) * y + k(0,2));
                cloud->points[i].y = z * (k(1,0) * x + k(1,1) * y + k(1,2));

                // Only update colors if reference image has changed
                if (refchanged)
                {
                    c = refim.pixel(x, y);
                    cloud->points[i].r = c.red();
                    cloud->points[i].g = c.green();
                    cloud->points[i].b = c.blue();
                }
                //depth8u->data[i] = (uchar)d;
            }

        // Show grayscale depthmap
        QImage img((const uchar *)depth8u->data, depth8u->width, depth8u->height, QImage::Format_Indexed8);
        img.setColorTable(ctable);

        depthim = QPixmap::fromImage(img);
        depthscene->addPixmap(depthim);
        depthscene->setSceneRect(depthim.rect());
        ui->depthview->setScene(depthscene);

        // show point cloud
        viewer->updatePointCloud(cloud, "cloud");
        if (refchanged) viewer->resetCamera();
        ui->qvtkWidget->update();
        refchanged = false;
    }
}

void PCLViewer::on_imNumber_valueChanged(int arg1)
{
    ps.setNumberofImages(arg1);
}

void PCLViewer::on_winSize_valueChanged(int arg1)
{
    // Make sure arg1 is odd
    arg1 = 2 * (arg1 / 2) + 1;
    ui->winSize->setValue(arg1);
    ps.setWindowSize(arg1);
}

void PCLViewer::on_numberPlanes_valueChanged(int arg1)
{
    ps.setNumberofPlanes(arg1);
}

void PCLViewer::on_zNear_valueChanged(double arg1)
{
    ps.setZnear(arg1);
}

void PCLViewer::on_zFar_valueChanged(double arg1)
{
    ps.setZfar(arg1);
}

void PCLViewer::on_stdThresh_valueChanged(double arg1)
{
    ps.setSTDthreshold(arg1);
}

void PCLViewer::on_nccThresh_valueChanged(double arg1)
{
    ps.setNCCthreshold(arg1);
}

void PCLViewer::on_pSlider2_valueChanged(int value)
{
    ui->cloudSize2->setValue(value);
    viewerdenoised->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtkDenoised->update ();
}

void PCLViewer::on_denoiseBtn_clicked()
{
    if (ps.CudaDenoise(argc, argv, ui->nIters->value(), ui->lambda->value(), ui->tvl1_tau->value(),
                       ui->tvl1_sigma->value(), ui->tvl1_theta->value(), ui->tvl1_beta->value(), ui->tvl1_gamma->value())){

        // get depthmaps
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        dendepth = ps.getDepthmapDenoised();
        dendepth8u = ps.getDepthmap8uDenoised();

        // resize cloud if reference image has changed
        if (refchangedtvl1)
        {
            if ((clouddenoised->height != dendepth8u->height) || (clouddenoised->width != dendepth8u->width))
            {
                // The number of points in the cloud
                clouddenoised->points.resize(dendepth8u->width * dendepth8u->height);
                clouddenoised->width = dendepth8u->width;
                clouddenoised->height = dendepth8u->height;
            }
        }

        QColor c;
        int  i;

        Matrix3D k = ps.getInverseK();
        double z;

        // update colorbar range
        ui->cbardenoised->setRangeMin(ui->zNear->value());
        ui->cbardenoised->setRangeMax(ui->zFar->value());

        // Fill the cloud
        for (size_t x = 0; x < dendepth8u->width; ++x)
            for (size_t y = 0; y < dendepth8u->height; ++y)
            {

                i = x + y * dendepth8u->width;

                z = dendepth->data[i];
                clouddenoised->points[i].z = sign(k(1,1)) * z;
                clouddenoised->points[i].x = z * (k(0,0) * x + k(0,1) * y + k(0,2));
                clouddenoised->points[i].y = z * (k(1,0) * x + k(1,1) * y + k(1,2));

                // only update colors if reference image has changed
                if (refchangedtvl1)
                {
                    c = refim.pixel(x, y);
                    clouddenoised->points[i].r = c.red();
                    clouddenoised->points[i].g = c.green();
                    clouddenoised->points[i].b = c.blue();
                }
            }

        // show grayscale depthmap
        QImage img((const uchar *)dendepth8u->data, dendepth8u->width, dendepth8u->height, QImage::Format_Indexed8);
        img.setColorTable(ctable);

        dendepthim = QPixmap::fromImage(img);
        dendepthsc->addPixmap(dendepthim);
        dendepthsc->setSceneRect(dendepthim.rect());
        ui->denview->setScene(dendepthsc);

        // show point cloud
        viewerdenoised->updatePointCloud(clouddenoised, "cloud");
        if (refchangedtvl1) viewerdenoised->resetCamera();
        ui->qvtkDenoised->update();
        refchangedtvl1 = false;
    }
}

void PCLViewer::on_threadsx_valueChanged(int arg1)
{
    ps.setBlockXdim(arg1);
    ui->threadsx->setValue(arg1);
}

void PCLViewer::on_threadsy_valueChanged(int arg1)
{
    ps.setBlockYdim(arg1);
    ui->threadsy->setValue(arg1);
}

void PCLViewer::on_tgv_button_pressed()
{
    if (ps.TGV(argc, argv, ui->tgv_niters->value(), ui->tgv_warps->value(), ui->tgv_lambda->value(),
               ui->tgv_alpha0->value(), ui->tgv_alpha1->value(), ui->tgv_tau->value(), ui->tgv_sigma->value(),
               ui->tgv_beta->value(), ui->tgv_gamma->value())){
//    if (ps.TGVdenoiseFromSparse(argc, argv, sparsedepth, ui->tgv_niters->value(), ui->tgv_alpha0->value(),
//                                ui->tgv_alpha1->value(), ui->tgv_tau->value(), ui->tgv_sigma->value(), ui->tgv_lambda->value(),
//                                ui->tgv_beta->value(), ui->tgv_gamma->value())){
        // get depthmaps
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        tgvdepth = ps.getDepthmapTGV();
        tgvdepth8u = ps.getDepthmap8uTGV();

        // resize cloud if reference image has changed
        if (refchangedtgv)
        {
            // The number of points in the cloud
            if ((cloudtgv->height != tgvdepth8u->height) || (cloudtgv->width != tgvdepth8u->width))
            {
                cloudtgv->points.resize(tgvdepth8u->width * tgvdepth8u->height);
                cloudtgv->width = tgvdepth8u->width;
                cloudtgv->height = tgvdepth8u->height;
            }
        }

        QColor c;
        int  i;

        Matrix3D k = ps.getInverseK();
        float z;

        // update colorbar range
        ui->cbarTGV->setRangeMin(ui->zNear->value());
        ui->cbarTGV->setRangeMax(ui->zFar->value());

        // Fill the cloud with points
        for (size_t x = 0; x < tgvdepth8u->width; ++x)
            for (size_t y = 0; y < tgvdepth8u->height; ++y)
            {
                i = x + y * tgvdepth8u->width;

                z = tgvdepth->data[i];
                cloudtgv->points[i].z = sign(k(1,1)) * z;
                cloudtgv->points[i].x = z * (k(0,0) * x + k(0,1) * y + k(0,2));
                cloudtgv->points[i].y = z * (k(1,0) * x + k(1,1) * y + k(1,2));

                // only update colors if reference image has changed
                if (refchangedtgv)
                {
                    c = refim.pixel(x, y);
                    cloudtgv->points[i].r = c.red();
                    cloudtgv->points[i].g = c.green();
                    cloudtgv->points[i].b = c.blue();
                }
            }

        // show grayscale depthmap
        QImage img((const uchar *)tgvdepth8u->data, tgvdepth8u->width, tgvdepth8u->height, QImage::Format_Indexed8);
        img.setColorTable(ctable);

        tgvdepthim = QPixmap::fromImage(img);
        tgvscene->addPixmap(tgvdepthim);
        tgvscene->setSceneRect(tgvdepthim.rect());
        ui->tgvview->setScene(tgvscene);

        // show cloud
        viewertgv->updatePointCloud(cloudtgv, "cloud");
        if (refchangedtgv) viewertgv->resetCamera();
        ui->qvtktgv->update();
        refchangedtgv = false;
    }
}

void PCLViewer::on_tgv_psize_valueChanged(int value)
{
    ui->tgv_psizebox->setValue(value);
    viewertgv->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtktgv->update ();
}

void PCLViewer::on_imagePathButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    "/home",
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    ui->imagePath->setText(dir);
}

void PCLViewer::on_imageNameButton_clicked()
{
    // Get full image file name
    QString loc = ui->imagePath->text();
    if (loc.isEmpty()) loc = "/home";
    QString name = QFileDialog::getOpenFileName(this, tr("Select Reference Image"),
                                                loc,
                                                tr("Images (*.png *.xpm *.jpg *.jpeg *.bmp *.dds *.mng *.tga *.tiff)"));

    // if nothing was selected, return
    if (name.isEmpty()) return;

    // Find the last forward slash, meaning end of directory path
    int i = name.lastIndexOf('/');

    // Split full file name into name and path
    QString n = name, path = name;
    if (i != -1) {
        path.truncate(i);
        n.remove(0, i + 1);
    }

    // Split file name into name and format
    QString format = n;
    i = n.lastIndexOf('.');
    if (i != 1) {
        format.remove(0, i + 1);
        n.truncate(i);
    }

    // Going from the end, count the number of digits and selected image number
    int digits = 0, ref = 0;
    QChar s = n.at(n.size() - 1);
    while (s.isDigit()){
        ref += pow(10, digits) * s.digitValue();
        digits++;
        n.truncate(n.size() - 1);
        s = n.at(n.size() - 1);
    }

    // Set appropriate boxes
    ui->imagePath->setText(path);
    ui->imageName->setText(n);
    ui->imageFormat->setText(format);
    ui->imageDigits->setValue(digits);
    ui->refNumber->setValue(ref);
    ui->fusion_simage->setValue(ref);
}

void PCLViewer::on_altmethod_toggled(bool checked)
{
    ps.setAlternativeRelativeMatrixMethod(checked);
}

void PCLViewer::on_loadfromsrc_clicked()
{
    refchanged = true;
    refchangedtvl1 = true;
    refchangedtgv = true;
    LoadImages();
}

void PCLViewer::on_loadfromdir_clicked()
{
    refchanged = true;
    refchangedtvl1 = true;
    refchangedtgv = true;

    // get image name strings
    QString impos;
    QString imname = ImageName(ui->refNumber->value(), impos);

    // initialize variables for camera parameters
    Vector3D cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, t;
    Matrix3D K, R;
    double cam_angle;

    // get camera parameters
    if (!reader.getcamParameters(impos, cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, cam_angle)) return;
    reader.getcamK(K, cam_dir, cam_up, cam_right);

    // set reference view parameters
    ps.setK(K);
    reader.computeRT(R, t, cam_dir, cam_pos, cam_up);

    refim.load(imname);
    int w = refim.width(), h = refim.height();
    ps.HostRef.setSize(w, h);

    // show reference image
    image = QPixmap::fromImage(refim);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->refView->setScene(scene);

    ps.HostRef.R = R;
    ps.HostRef.t = t;
    rgb2gray<float>(ps.HostRef.data, refim);

    int nsrc = ui->imNumber->value() - 1;

    // load and setup source views
    ps.HostSrc.resize(0);
    QImage src;
    int half = (nsrc + 1) / 2;
    int offset;

    for (int i = 0; i < nsrc; i++){
        if (i < half) offset = i + 1;
        else offset = half - i - 1;
        imname = ImageName(ui->refNumber->value() + offset, impos);
        if (reader.getcamParameters(impos, cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, cam_angle)){
            ps.HostSrc.resize(ps.HostSrc.size() + 1);
            reader.computeRT(R, t, cam_dir, cam_pos, cam_up);
            src.load(imname);
            ps.HostSrc.back().setSize(w,h);
            ps.HostSrc.back().R = R;
            ps.HostSrc.back().t = t;
            rgb2gray<float>(ps.HostSrc.back().data, src);
        }
    }

    // load sparse groundtruth depthmap the same size as reference image
    QString depth = impos;
    int dot = depth.lastIndexOf('.');
    if (dot != -1) depth.truncate(dot);
    depth += ".depth";
    loadSparseDepthmap(depth);
}

QString PCLViewer::ImageName(int number, QString &imagePos)
{
    QString name = ui->imagePath->text();
    name += '/';
    name += ui->imageName->text();
    int n;
    for (int i = 0; i < ui->imageDigits->value(); i++) {
        n = number / (int)pow(10, ui->imageDigits->value() - i - 1);
        n = n % 10;
        name += QString::number(n);
    }
    name += '.';
    imagePos = name;
    name += ui->imageFormat->text();
    imagePos += "txt";
    return name;
}

template<typename T>
void PCLViewer::rgb2gray(T * data, const QImage & img)
{
    int w = img.width(), h = img.height();
    QColor c;

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){

            c = img.pixel(x, y);
            data[x + y*w] = T(RGB2GRAY_WEIGHT_RED * c.red() +
                              RGB2GRAY_WEIGHT_BLUE * c.blue() +
                              RGB2GRAY_WEIGHT_GREEN * c.green());
        }
    }
}

void PCLViewer::on_save_clicked()
{
    QFile file;

    // only save images if they contain data
    if (!refim.isNull())
    {
        file.setFileName("reference.png");
        file.open(QIODevice::WriteOnly);
        refim.save(&file, "PNG");
        file.close();
    }

    if (!depthim.isNull())
    {
        file.setFileName("planesweep.png");
        file.open(QIODevice::WriteOnly);
        depthim.save(&file, "PNG");
        file.close();
    }

    if (!dendepthim.isNull())
    {
        file.setFileName("planesweep_tvl1.png");
        file.open(QIODevice::WriteOnly);
        dendepthim.save(&file, "PNG");
        file.close();
    }

    if (!tgvdepthim.isNull())
    {
        file.setFileName("tgv.png");
        file.open(QIODevice::WriteOnly);
        tgvdepthim.save(&file, "PNG");
        file.close();
    }

    // only save cloud if they contain points
    try {
        if (cloud->points.size() > 0) pcl::io::savePLYFileASCII("planesweep.ply", *cloud);
        if (clouddenoised->points.size() > 0) pcl::io::savePLYFileASCII("planesweep_tvl1.ply", *clouddenoised);
        if (cloudtgv->points.size() > 0) pcl::io::savePLYFileASCII("tgv.ply", *cloudtgv);
        if (cloudfusion->points.size() > 0) pcl::io::savePLYFileASCII("reconstructed.ply", *cloudfusion);
    }
    catch (pcl::IOException & excep){
        std::cerr << "Error occured while saving PCD:\n" << excep.detailedMessage() << std::endl;
        return;
    }
}

void PCLViewer::on_nIters_valueChanged(int arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_lambda_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_tau_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_sigma_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_theta_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_beta_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_gamma_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_reconstruct_button_clicked()
{
    // Set 3D volume for the voxels
    Rectangle3D volm(make_float3(ui->fusion_volx1->value(),
                                 ui->fusion_voly1->value(),
                                 ui->fusion_volz1->value()),
                     make_float3(ui->fusion_volx2->value(),
                                 ui->fusion_voly2->value(),
                                 ui->fusion_volz2->value()));
    fd.setVolume(volm);

    // Initialize variables
    float * ptr = 0; // null pointer
    double  threshold = ui->fusion_threshold->value(),
            tau = ui->fusion_tau->value(),
            lambda = ui->fusion_lambda->value(),
            sigma = ui->fusion_sigma->value();

    // Create matrices and translation vector
    Matrix3D K;
    Matrix3D R;
    Vector3D T;

    // Create world coordinate matrices:
    Matrix3D I; I.makeIdentity();
    Vector3D tm(0,0,0);  // 0 translation vector

    // Calculate 3D threads per blocks and blocks per grid
    dim3 threads(ui->fusion_threadsw->value(),
                 ui->fusion_threadsh->value(),
                 ui->fusion_threadsd->value());
    int3 th = make_int3(threads.x, threads.y, threads.z);
    int3 b = make_int3(fd.width(), fd.height(), fd.depth());
    b = (b + th - 1);
    b = make_int3(b.x / th.x, b.y / th.y, b.z / th.z);
    dim3 blocks(b.x, b.y, b.z);
    std::cerr << blocks.x << '\t' << blocks.y << '\t' << blocks.z << std::endl;

    // Run iterations
    int iterations = ui->fusion_iters->value();
//    size_t pitch;
//    checkCudaErrors(cudaMallocPitch(&ptr, &pitch, 640 * sizeof(float), 480));

    for (int i = 0; i < iterations; i++){
        printf("Reconstruction iteration: %d/%d\n", i+1, iterations);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Load images and set K
        ui->refNumber->setValue(ui->fusion_simage->value() + i * ui->fusion_imstep->value());
        on_loadfromdir_clicked();
        K = ps.getK();
//        MemoryManagement<float>::Host2DeviceCopy(ptr, pitch, sparsedepth.data, sparsedepth.pitch, 640, 480);

        // Calculate and set R and T
        ps.RelativeMatrices(R, T, I, tm, ps.HostRef.R, ps.HostRef.t); // from world to ref

        // Get planesweep depthmap
        ps.RunAlgorithm(argc, argv);

        // Get TVL1 denoised planesweep depthmap
        ps.CudaDenoise(argc, argv, ui->nIters->value(), ui->lambda->value(), ui->tvl1_tau->value(),
                       ui->tvl1_sigma->value(), ui->tvl1_theta->value(), ui->tvl1_beta->value(), ui->tvl1_gamma->value());
//        ps.TGVdenoiseFromSparse(argc, argv, sparsedepth, ui->tgv_niters->value(), ui->tgv_alpha0->value(),
//                                ui->tgv_alpha1->value(), ui->tgv_tau->value(), ui->tgv_sigma->value(), ui->tgv_lambda->value(),
//                                ui->tgv_beta->value(), ui->tgv_gamma->value());
        ptr = ps.getDepthmapDenoisedPtr();

        // Fuse the depthmap
        FusionUpdateIteration<8>(fd, ptr, K, R, T, threshold, tau, lambda, sigma, ps.HostRef.width, ps.HostRef.height, blocks, threads);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cerr << "Time of 1 fusion iteration: " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";
    }

    // Resize point cloud to fit all voxels in the worst case scenario
    cloudfusion->points.resize(fd.elements());
    size_t voxels = 0;
    float3 c;
    uchar gray;
    uchar3 color;

    // Copy data to host
    fd.copyTo(f.voxelPtr(), f.pitch());

    // Only show voxels which are occluded but not too far from the surface
    for (int z = 0; z < fd.depth(); z++)
        for (int y = 0; y < fd.height(); y++)
            for (int x = 0; x < fd.width(); x++)
            {
                if ((f.u(x,y,z) < 0) && (f.u(x,y,z) > -1)){
                    c = fd.worldCoords(x,y,z);
                    cloudfusion->points[voxels].x = -c.x;
                    cloudfusion->points[voxels].y = c.y;
                    cloudfusion->points[voxels].z = c.z;
                    gray = 255 * (c.x - fd.volume().a.x) / fd.volume().size().x;
                    color = RGBdepthmapColor(gray);
                    cloudfusion->points[voxels].r = color.x;
                    cloudfusion->points[voxels].g = color.y;
                    cloudfusion->points[voxels].b = color.z;
                    voxels++;
                }
            }
//    cudaFree(ptr);
    // update point cloud and qvtkwidget
    cloudfusion->points.resize(voxels);
    cloudfusion->width = 1;
    cloudfusion->height = voxels;
    viewerfusion->updatePointCloud(cloudfusion, "cloud");
    viewerfusion->resetCamera();
    ui->qvtkfusion->update();
}

uchar3 PCLViewer::RGBdepthmapColor(uchar depth)
{
    uchar3 color = make_uchar3(0,0,0);

    if (depth < 32){ // blue
        color.z = 128 + depth * 4;
        return color;
    }
    if (depth < 96){ // light blue
        color.y = (depth - 32) * 4;
        color.z = 255;
        return color;
    }
    if (depth < 160){ // yellow
        color.x = (depth - 96) * 4;
        color.y = 255;
        color.z = 255 - color.x;
        return color;
    }
    if (depth < 224){ // red
        color.x = 255;
        color.y = 255 - (depth - 160) * 4;
        return color;
    }

    // dark red
    color.x = 255 - (depth - 224) * 4;

//    if (depth == 255) color = make_uchar3(0,0,0);
    return color;
}

void PCLViewer::on_fusion_psize_valueChanged(int value)
{
    ui->fusion_psizebox->setValue(value);
    viewerfusion->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtkfusion->update ();
}

void PCLViewer::on_fusion_resize_clicked()
{
    if ((ui->fusion_d->value() != fd.depth()) || (ui->fusion_h->value() != fd.height()) || (ui->fusion_w->value() != fd.width()))
    {
        fd.Resize(ui->fusion_w->value(), ui->fusion_h->value(), ui->fusion_d->value());
        f.Resize(ui->fusion_w->value(), ui->fusion_h->value(), ui->fusion_d->value());

        std::cerr << "New size of fusion data is " << fd.sizeMBytes() << "MB\n\n";
    }
}

void PCLViewer::on_fusion_threadsw_valueChanged(int arg1)
{
    ui->fusion_threadsw->setValue(min(arg1, ui->maxthreads->value() / ui->fusion_threadsh->value() / ui->fusion_threadsd->value()));
}

void PCLViewer::on_fusion_threadsh_valueChanged(int arg1)
{
    ui->fusion_threadsh->setValue(min(arg1, ui->maxthreads->value() / ui->fusion_threadsw->value() / ui->fusion_threadsd->value()));
}

void PCLViewer::on_fusion_threadsd_valueChanged(int arg1)
{
    ui->fusion_threadsd->setValue(min(arg1, ui->maxthreads->value() / ui->fusion_threadsh->value() / ui->fusion_threadsw->value()));
}

bool PCLViewer::loadSparseDepthmap(const QString & fileName)
{
    QFile file;

    // try opening file
    file.setFileName(fileName);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading depth file", file.errorString());
        return false;
    }

    QTextStream in(&file);

    // all values are in single line, read it and split it into separate strings with a single depth value
    QString line = in.readLine();
    QStringList n;
    n = line.split(' ', QString::SkipEmptyParts);

    int w = refim.width(), h = refim.height();
    sparsedepth.setSize(w, h);
    float depth;
    float u, v;

    for (int y = 0; y < h; y = y + 1){
        for (int x = 0; x < w; x = x + 1){
            // correct for radial depth
            u = (x - 320.5) / 481.2043;
            v = (y - 240.5) / 479.9998;
            depth = n.at(x + w * y).trimmed().toFloat();
            sparsedepth.data[x + w * y] = depth / sqrt(pow(u, 2) + pow(v, 2) + 1);
        }
    }

    file.close();
    return true;
}
