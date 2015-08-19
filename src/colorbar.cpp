#include <QMouseEvent>
#include <QPixmap>
#include <QImage>
#include <QPainter>
#include <iostream>
#include <algorithm>
#include "colorbar.h"

ColorBar::ColorBar(QWidget *parent):
    QWidget(parent),
    d_orientation(Qt::Vertical),
    tickp(QSlider::TicksBelow),
    labelp(tickp)
{
    setup();
}

ColorBar::ColorBar(Qt::Orientation o, QWidget *parent):
    QWidget(parent),
    d_orientation(o),
    tickp(QSlider::TicksBelow),
    labelp(tickp)
{
    setup();
}

void ColorBar::setup()
{
    ctable.resize(2);
    ctable[0] = Qt::black;
    ctable[1] = Qt::white;
    setMinimumWidth(25 + 100);
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

void ColorBar::setColorBarSize(int size)
{
    sz = size;
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

void ColorBar::setTickLabelPosition(const QSlider::TickPosition pos)
{
    labelp = pos;
    update();
}

void ColorBar::setRangeMin(double rmin)
{
    r.rmin = rmin;
    update();
}

void ColorBar::setRangeMax(double rmax)
{
    r.rmax = rmax;
    update();
}

void ColorBar::setRange(double rmin, double rmax)
{
    r = range(rmin, rmax);
    update();
}

void ColorBar::setRange(range r)
{
    this->r = r;
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
    QRect rct = ColorBarRect();
    drawColorBar(&painter, rct);
    drawColorBarTicks(&painter, rct);
    drawColorBarTickLabels(&painter, rct);
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

    // setup pen
    QPen pen(painter->pen());
    pen.setWidth(1);
    pen.setColor(Qt::black);
    painter->setPen(pen);

    // check where to draw ticks
    bool above, below;
    positions(above, below, tickp);

    int x1, y1, x2, y2;
    // draw ticks
    for (int i = 0; i < nticks; i++){
        if (above) { // above / left
            ColorBarTickLine(i, true, x1, y1, x2, y2, rect);
            painter->drawLine(x1, y1, x2, y2);
        }
        if (below) { // below / right
            ColorBarTickLine(i, false, x1, y1, x2, y2, rect);
            painter->drawLine(x1, y1, x2, y2);
        }
    }

    painter->restore();
}

void ColorBar::drawColorBarTickLabels(QPainter *painter, const QRect &rect) const
{
    if (tickp == QSlider::NoTicks) return;

    painter->save();

    // use widget font
    painter->setFont(font());

    // check where to draw ticks
    bool above, below;
    positions(above, below, labelp);

    Qt::Alignment flagsa, flagsb;

    if (d_orientation == Qt::Vertical) {
        flagsa = flagsb = Qt::AlignVCenter;
        if (above) flagsa |= Qt::AlignRight;
        if (below) flagsb |= Qt::AlignLeft;
    }
    else {
        flagsa = flagsb = Qt::AlignHCenter;
        if (above) flagsa |= Qt::AlignBottom;
        if (below) flagsb |= Qt::AlignTop;
    }

    int x1, y1, x2, y2; // only need x1,y1 as they are on the border
    QString text;

    // draw ticks
    for (int i = 0; i < nticks; i++){
        text = QString::number(r.rmin + r.Range() * i / (nticks - 1), 'g', 2);
        if (above) { // above / left
            ColorBarTickLine(i, true, x1, y1, x2, y2, rect);
            if (d_orientation == Qt::Vertical) x1 -= 5;
            else y1 -= 5;
            drawText(*painter, (qreal)x1, (qreal)y1, flagsa, text);
        }
        if (below) { // below / right
            ColorBarTickLine(i, false, x1, y1, x2, y2, rect);
            if (d_orientation == Qt::Vertical) x1 += 5;
            else y1 += 5;
            drawText(*painter, (qreal)x1, (qreal)y1, flagsb, text);
        }
    }

    painter->restore();
}

void ColorBar::ColorBarTickLine(unsigned int i, bool above, int & x1, int & y1, int & x2, int & y2,
                                const QRect &rect) const
{
    int size;
    if ( d_orientation == Qt::Horizontal )
        size = rect.width();
    else
        size = rect.height();

    unsigned int j = nticks - 1 - i;
    if (d_orientation == Qt::Horizontal){
        x1 = std::min<int>(rect.right(), rect.left() + j * size / (nticks - 1));
        x2 = x1;
        if (above) { y1 = rect.top(); y2 = y1 + 5; }
        else { y1 = rect.bottom(); y2 = y1 - 5; }
    }
    else { // == Qt::Vertical
        y1 = std::min<int>(rect.bottom(), rect.top() + j * size / (nticks - 1));
        y2 = y1;
        if (above) { x1 = rect.left(); x2 = x1 + 5; }
        else { x1 = rect.right(); x2 = x1 - 5; }
    }
}

void ColorBar::positions(bool &above, bool &below, QSlider::TickPosition pos) const
{
    above = true;
    below = true;
    if (pos == QSlider::TicksAbove) below = false; // == TicksLeft for vertical bar
    if (pos == QSlider::TicksBelow) above = false; // == TicksRight for vertical bar
}

QRect ColorBar::ColorBarRect() const
{
    int left, top, w, h;
    bool above, below;
    positions(above, below, labelp);
    if (d_orientation == Qt::Vertical) {
        w = sz;
        h = rect().height() - 20;
        top = rect().top() + 10;
        if (above && below) left = rect().left() + rect().width() / 2 - w / 2;
        else if (above) left = rect().right() - w;
        else left = rect().left();
    }
    else {
        w = rect().width() - 30;
        left = rect().left() + 15;
        h = sz;
        if (above && below) top = rect().top() + rect().height() / 2 - h / 2;
        else if (above) top = rect().bottom() - h;
        else top = rect().top();
    }
    return QRect(left, top, w, h);
}
