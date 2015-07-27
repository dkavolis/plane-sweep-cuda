#include "pclviewer.h"
#include "ui_pclviewer.h"
#include <iostream>
#include <QDir>
#include <QByteArray>
#include <QBuffer>
#include <boost/numeric/ublas/assignment.hpp>
#include <QPixmap>
#include <QColor>

PCLViewer::PCLViewer (int argc, char **argv, QWidget *parent) :
    QMainWindow (parent),
    ui (new Ui::PCLViewer),
    scene (new QGraphicsScene()),
    depthscene (new QGraphicsScene()),
    dendepthsc (new QGraphicsScene()),
    tgvscene (new QGraphicsScene()),
    ps(argc, argv)
{
    ui->setupUi (this);
    this->setWindowTitle ("PCL viewer");

    // Setup the cloud pointer
    cloud.reset (new PointCloudT);

    // Set up the QVTK window
    viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
    viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
    ui->qvtkWidget->update ();

    // Setup the cloud pointer
    clouddenoised.reset (new PointCloudT);

    // Set up the QVTK window
    viewerdenoised.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkDenoised->SetRenderWindow (viewerdenoised->getRenderWindow ());
    viewerdenoised->setupInteractor (ui->qvtkDenoised->GetInteractor (), ui->qvtkDenoised->GetRenderWindow ());
    ui->qvtkDenoised->update ();

    // Setup the cloud pointer
    tgvcloud.reset (new PointCloudT);

    // Set up the QVTK window
    tgvviewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtktgv->SetRenderWindow (tgvviewer->getRenderWindow ());
    tgvviewer->setupInteractor (ui->qvtktgv->GetInteractor (), ui->qvtktgv->GetRenderWindow ());
    ui->qvtktgv->update ();

    connect(ui->pSlider, SIGNAL(valueChanged(int)), this, SLOT(pSliderValueChanged(int)));

    viewer->addPointCloud (cloud, "cloud");
    pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();

    viewerdenoised->addPointCloud (clouddenoised, "cloud");
    on_pSlider2_valueChanged (2);
    viewerdenoised->resetCamera ();
    ui->qvtkDenoised->update ();

    tgvviewer->addPointCloud (tgvcloud, "cloud");
    ui->tgv_psize->setValue(2);
    tgvviewer->resetCamera ();
    ui->qvtktgv->update ();

    ui->depthview->setScene(depthscene);

    ui->imNumber->setValue(ps.getNumberofImages());
    ui->winSize->setValue((ps.getWindowSize()));
    ui->nccThresh->setValue(ps.getNCCthreshold());
    ui->stdThresh->setValue(ps.getSTDthreshold());
    ui->numberPlanes->setValue(ps.getNumberofPlanes());
    ui->zNear->setValue(ps.getZnear());
    ui->zFar->setValue(ps.getZfar());
    ui->stdThresh->setMaximum(255 * 1.f);

    ui->lambda_label->setText( trUtf8( "\xce\xbb" ) );
    ui->lambda->setValue(DEFAULT_TVL1_LAMBDA);
    ui->nIters->setValue(DEFAULT_TVL1_ITERATIONS);

    ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
    ui->threadsx->setMaximum(ui->maxthreads->value());
    ui->threadsy->setMaximum(ui->maxthreads->value());
    dim3 t = ps.getThreadsPerBlock();
    ui->threadsx->setValue(t.x);
    ui->threadsy->setValue(t.y);

    QChar alpha = QChar(0xb1, 0x03);
    QChar sigma(0x03C3), tau(0x03C4);
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

    ui->tgv_alpha0->setValue(DEFAULT_TGV_ALPHA0);
    ui->tgv_alpha1->setValue(DEFAULT_TGV_ALPHA1);
    ui->tgv_lambda->setValue(DEFAULT_TGV_LAMBDA);
    ui->tgv_niters->setValue(DEFAULT_TGV_NITERS);
    ui->tgv_warps->setValue(DEFAULT_TGV_NWARPS);
    ui->tgv_sigma->setValue(DEFAULT_TGV_SIGMA);
    ui->tgv_tau->setValue(DEFAULT_TGV_TAU);

    LoadImages();
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
    QString loc = "/../src/PlaneSweep/im";
    QString ref = QDir::currentPath();
    ref += loc;
    ref += QString::number(0);
    ref += ".png";
    refim.load(ref);

    image = QPixmap::fromImage(refim);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->refView->setScene(scene);

    double K[3][3] = {
        {0.709874*640, (1-0.977786)*640, 0.493648*640},
        {0, 0.945744*480, 0.514782*480},
        {0, 0, 1}
    };

    ps.setK(K);
    ublas::matrix<double> Cr(3,4);
    std::vector<ublas::matrix<double>> C(9);

    Cr <<= 0.993701, 0.110304, -0.0197854, 0.280643,
          0.0815973, -0.833193, -0.546929, -0.255355,
          -0.0768135, 0.541869, -0.836945, 0.810979;

    C[0].resize(3,4);
    C[0] <<= 0.993479, 0.112002, -0.0213286, 0.287891,
            0.0822353, -0.83349, -0.54638, -0.255839,
            -0.0789729, 0.541063, -0.837266, 0.808608;

    C[1].resize(3,4);
    C[1] <<= 0.993199, 0.114383, -0.0217434, 0.295475,
            0.0840021, -0.833274, -0.546442, -0.25538,
            -0.0806218, 0.540899, -0.837215, 0.805906;

    C[2].resize(3,4);
    C[2] <<= 0.992928, 0.116793, -0.0213061, 0.301659,
            0.086304, -0.833328, -0.546001, -0.254563,
            -0.081524, 0.5403, -0.837514, 0.804653;

    C[3].resize(3,4);
    C[3] <<= 0.992643, 0.119107, -0.0217442, 0.309666,
            0.0880017, -0.833101, -0.546075, -0.254134,
            -0.0831565, 0.540144, -0.837454, 0.802222;

    C[4].resize(3,4);
    C[4] <<= 0.992429, 0.121049, -0.0208028, 0.314892,
            0.0901911, -0.833197, -0.545571, -0.253009,
            -0.0833736, 0.539564, -0.837806, 0.801559;

    C[5].resize(3,4);
    C[5] <<= 0.992226, 0.122575, -0.0215154, 0.32067,
            0.0911582, -0.833552, -0.544869, -0.254142,
            -0.0847215, 0.538672, -0.838245, 0.799812;

    C[6].resize(3,4);
    C[6] <<= 0.992003, 0.124427, -0.0211509, 0.325942,
            0.0930933, -0.834508, -0.543074, -0.254865,
            -0.0852237, 0.536762, -0.839418, 0.799037;

    C[7].resize(3,4);
    C[7] <<= 0.991867, 0.125492, -0.021234, 0.332029,
            0.0938678, -0.833933, -0.543824, -0.252767,
            -0.0859533, 0.537408, -0.838931, 0.797979;

    C[8].resize(3,4);
    C[8] <<= 0.991515, 0.128087, -0.0221943, 0.33934,
            0.095507, -0.833589, -0.544067, -0.250995,
            -0.0881887, 0.53733, -0.838748, 0.796756;

    refgray = refim.convertToFormat(QImage::Format_Grayscale8);
    ps.HostRef8u.CopyFrom(refgray.constBits(), refgray.bytesPerLine(), refgray.width(), refgray.height());
    ps.CmatrixToRT(Cr, ps.HostRef8u.R, ps.HostRef8u.t);

    QString src;
	sources.resize(9);
    ps.HostSrc8u.resize(9);
    for (int i = 0; i < 9; i++){
        src = QDir::currentPath();
        src += loc;
        src += QString::number(i + 1);
        src += ".png";
        sources[i].load(src);
        sources[i] = sources[i].convertToFormat(QImage::Format_Grayscale8);
        ps.HostSrc8u[i].CopyFrom(sources[i].constBits(), sources[i].bytesPerLine(), sources[i].width(), sources[i].height());
        ps.CmatrixToRT(C[i], ps.HostSrc8u[i].R, ps.HostSrc8u[i].t);
    }

    ps.Convert8uTo32f(argc, argv);
}

PCLViewer::~PCLViewer ()
{
    delete[] argv;
    delete ui;
}

void PCLViewer::on_pushButton_pressed()
{
	if (ps.RunAlgorithm(argc, argv)){
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        depth = *ps.getDepthmap();
		// The number of points in the cloud
        cloud->points.resize(depth.width * depth.height);

		QColor c;
        int  i;
        float zn = ps.getZnear();
        float zf = ps.getZfar();
        float d;
        depth8u.setSize(depth.width, depth.height);

		// Fill the cloud with some points
        for (size_t x = 0; x < depth.width; ++x)
            for (size_t y = 0; y < depth.height; ++y)
            {

                i = x + y * depth.width;
                // Check if QNAN
                if (depth.data[i] == depth.data[i]) d = 255 * (depth.data[i] - zn) / (zf - zn);
                else d = 255;
                cloud->points[i].x = x;
                cloud->points[i].y = depth.height - y;
                cloud->points[i].z = -d;

                c = refim.pixel(x, y);
                cloud->points[i].r = c.red();
                cloud->points[i].g = c.green();
                cloud->points[i].b = c.blue();
                depth8u.data[i] = (uchar)d;
            }

        QImage img((const uchar *)depth8u.data, depth.width, depth.height, QImage::Format_Grayscale8);

        depthim = QPixmap::fromImage(img);
        depthscene->addPixmap(depthim);
        depthscene->setSceneRect(depthim.rect());
        ui->depthview->setScene(depthscene);

		viewer->updatePointCloud(cloud, "cloud");
		viewer->resetCamera();
		ui->qvtkWidget->update();
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
//    if (ps.Denoise(ui->nIters->value(), ui->lambda->value())){
    if (ps.CudaDenoise(argc, argv, ui->nIters->value(), ui->lambda->value())){
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        dendepth8u = *ps.getDepthmap8uDenoised();
        // The number of points in the cloud
        clouddenoised->points.resize(dendepth8u.width * dendepth8u.height);

        QColor c;
        int  i;

        // Fill the cloud with some points
        for (size_t x = 0; x < dendepth8u.width; ++x)
            for (size_t y = 0; y < dendepth8u.height; ++y)
            {

                i = x + y * dendepth8u.width;

                clouddenoised->points[i].x = x;
                clouddenoised->points[i].y = dendepth8u.height - y;
                clouddenoised->points[i].z = -dendepth8u.data[i];

                c = refim.pixel(x, y);
                clouddenoised->points[i].r = c.red();
                clouddenoised->points[i].g = c.green();
                clouddenoised->points[i].b = c.blue();
            }

        QImage img((const uchar *)dendepth8u.data, dendepth8u.width, dendepth8u.height, QImage::Format_Grayscale8);

        dendepth = QPixmap::fromImage(img);
        dendepthsc->addPixmap(dendepth);
        dendepthsc->setSceneRect(dendepth.rect());
        ui->denview->setScene(dendepthsc);

        viewerdenoised->updatePointCloud(clouddenoised, "cloud");
        viewerdenoised->resetCamera();
        ui->qvtkDenoised->update();
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
               ui->tgv_alpha0->value(), ui->tgv_alpha1->value(), ui->tgv_tau->value(), ui->tgv_sigma->value())){
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        tgvdepth8u = *ps.getDepthmap8uTGV();
        // The number of points in the cloud
        tgvcloud->points.resize(tgvdepth8u.width * tgvdepth8u.height);

        QColor c;
        int  i;

        // Fill the cloud with some points
        for (size_t x = 0; x < tgvdepth8u.width; ++x)
            for (size_t y = 0; y < tgvdepth8u.height; ++y)
            {

                i = x + y * tgvdepth8u.width;

                tgvcloud->points[i].x = x;
                tgvcloud->points[i].y = tgvdepth8u.height - y;
                tgvcloud->points[i].z = -tgvdepth8u.data[i];

                c = refim.pixel(x, y);
                tgvcloud->points[i].r = c.red();
                tgvcloud->points[i].g = c.green();
                tgvcloud->points[i].b = c.blue();
            }

        QImage img((const uchar *)tgvdepth8u.data, tgvdepth8u.width, tgvdepth8u.height, QImage::Format_Grayscale8);

        tgvdepth = QPixmap::fromImage(img);
        tgvscene->addPixmap(tgvdepth);
        tgvscene->setSceneRect(tgvdepth.rect());
        ui->tgvview->setScene(tgvscene);

        tgvviewer->updatePointCloud(tgvcloud, "cloud");
        tgvviewer->resetCamera();
        ui->qvtktgv->update();
    }
}

void PCLViewer::on_tgv_psize_valueChanged(int value)
{
    ui->tgv_psizebox->setValue(value);
    tgvviewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtktgv->update ();
}
