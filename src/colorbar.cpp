#include <QMouseEvent>
#include <QPixmap>
#include <QImage>
#include <QPainter>
#include "colorbar.h"

ColorBar::ColorBar(QWidget *parent):
    QWidget(parent),
    d_orientation(Qt::Vertical)
{
    ctable.resize(2);
    ctable[0] = Qt::black;
    ctable[1] = Qt::white;
#ifndef QT_NO_CURSOR
#if QT_VERSION < 0x040000
    setCursor(Qt::pointingHandCursor);
#else
    setCursor(Qt::PointingHandCursor);
#endif
#endif
}

ColorBar::ColorBar(Qt::Orientation o, QWidget *parent):
    QWidget(parent),
    d_orientation(o)
{
    ctable.resize(2);
    ctable[0] = Qt::black;
    ctable[1] = Qt::white;
#ifndef QT_NO_CURSOR
#if QT_VERSION < 0x040000
    setCursor(Qt::pointingHandCursor);
#else
    setCursor(Qt::PointingHandCursor);
#endif
#endif
}

void ColorBar::setOrientation(Qt::Orientation o)
{
    d_orientation = o;
    update();
}

void ColorBar::setColorTable(const QVector<QRgb> &table)
{
    ctable = table;
    update();
}

void ColorBar::mousePressEvent(QMouseEvent *e)
{
    if( e->button() ==  Qt::LeftButton )
    {
        // emit the color of the position where the mouse click
        // happened

        const QPixmap pm = QWidget::grab();
#if QT_VERSION < 0x040000
        const QRgb rgb = pm.convertToImage().pixel(e->x(), e->y());
#else
        const QRgb rgb = pm.toImage().pixel(e->x(), e->y());
#endif

        emit selected(QColor(rgb));
        e->accept();
    }
}

void ColorBar::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    drawColorBar(&painter, rect());
}

void ColorBar::drawColorBar(QPainter *painter, const QRect &rect) const
{
    painter->save();
    painter->setClipRect(rect);
    painter->setClipping(true);

    painter->fillRect(rect, ctable[0]);

    int size;
    const int numIntervalls = ctable.size();
    if ( d_orientation == Qt::Horizontal )
        size = rect.width();
    else
        size = rect.height();

    int lower, upper, sectionSize;

    for ( int i = 0; i < numIntervalls; i++ )
    {
        QRect section;
        lower = size * i / numIntervalls;
        upper = size * (i + 1) / numIntervalls;
        sectionSize = upper - lower;

        if ( d_orientation == Qt::Horizontal )
        {
            section.setRect(rect.x() + lower, rect.y(),
                sectionSize, rect.height());
        }
        else
        {
            section.setRect(rect.x(), rect.y() + lower,
                rect.width(), sectionSize);
        }

        painter->fillRect(section, QColor::fromRgb(ctable[numIntervalls - i - 1]));
    }

    painter->restore();
}
