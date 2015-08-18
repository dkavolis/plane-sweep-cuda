#include <QMouseEvent>
#include <QPixmap>
#include <QImage>
#include <QPainter>
#include <iostream>
#include "colorbar.h"

ColorBar::ColorBar(QWidget *parent):
    QWidget(parent),
    d_orientation(Qt::Vertical),
    tickp(QSlider::TicksBelow)
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
    d_orientation(o),
    tickp(QSlider::TicksBelow)
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

void ColorBar::setNumberOfTicks(const int ticks)
{
    nticks = ticks;
    update();
}

void ColorBar::setTickPosition(const QSlider::TickPosition pos)
{
    tickp = pos;
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
    drawColorBarTicks(&painter, rect());
}

void ColorBar::drawColorBar(QPainter *painter, const QRect &rect) const
{
    painter->save();
    painter->setClipRect(rect);
    painter->setClipping(true);

    int size;
    const int numIntervalls = ctable.size();
    if ( d_orientation == Qt::Horizontal )
        size = rect.width();
    else
        size = rect.height();

    int lower, upper, sectionSize;

    // fill in colors
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

    // draw rectangle border
    QPen pen(painter->pen());
    pen.setWidth(2);
    pen.setColor(Qt::black);
    painter->setPen(pen);
    painter->drawRect(rect);

    painter->restore();
}

void ColorBar::drawColorBarTicks(QPainter *painter, const QRect &rect) const
{
    if (tickp == QSlider::NoTicks) return;

    painter->save();

    int size;
    if ( d_orientation == Qt::Horizontal )
        size = rect.width();
    else
        size = rect.height();

    // setup pen
    QPen pen(painter->pen());
    pen.setWidth(1);
    pen.setColor(Qt::black);
    painter->setPen(pen);

    // check where to draw ticks
    bool above = true, below = true;
    if (tickp == QSlider::TicksAbove) below = false; // == TicksLeft for vertical bar
    if (tickp == QSlider::TicksBelow) above = false; // == TicksRight for vertical bar

    // draw ticks
    for (int i = 0; i < nticks; i++){

        if (d_orientation == Qt::Horizontal){
            int x = rect.left() + i * size / (nticks - 1);
            if (above) painter->drawLine(x, rect.top(), x, rect.top() + 5);
            if (below) painter->drawLine(x, rect.bottom(), x, rect.bottom() - 5);
        }
        else { // == Qt::Vertical
            int y = rect.top() + i * size / (nticks - 1);
            if (above) painter->drawLine(rect.left(), y, rect.left() + 5, y);
            if (below) painter->drawLine(rect.right(), y, rect.right() - 5, y);
        }
    }

    painter->restore();
}
