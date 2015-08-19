#ifndef COLORBAR_H
#define COLORBAR_H

#include <QWidget>
#include <QSlider>
#include <Qpainter>

// class for drawing aligned wrt to a point text with QPainter
class PainterText
{
public:
    inline
    void drawText(QPainter & painter, qreal x, qreal y, Qt::Alignment flags,
                  const QString & text, QRectF * boundingRect = 0) const
    {
       const qreal size = 32767.0;
       QPointF corner(x, y - size);
       if (flags & Qt::AlignHCenter) corner.rx() -= size/2.0;
       else if (flags & Qt::AlignRight) corner.rx() -= size;
       if (flags & Qt::AlignVCenter) corner.ry() += size/2.0;
       else if (flags & Qt::AlignTop) corner.ry() += size;
       else flags |= Qt::AlignBottom;
       QRectF rect(corner, QSizeF(size, size));
       painter.drawText(rect, flags, text, boundingRect);
    }

    inline
    void drawText(QPainter & painter, const QPointF & point, Qt::Alignment flags,
                  const QString & text, QRectF * boundingRect = 0) const
    {
       drawText(painter, point.x(), point.y(), flags, text, boundingRect);
    }
};

class ColorBar: public QWidget, protected PainterText
{
    struct range {
        double rmin, rmax;

        inline range(double min = 0, double max = 0) :
            rmin(min), rmax(max)
        {}

        inline range(const range & r) :
            rmin(r.rmin), rmax(r.rmax)
        {}

        inline double Range() const { return rmax - rmin; }

        inline range & operator=(const range & r){
            if (this == &r) return *this;
            rmin = r.rmin;
            rmax = r.rmax;
            return *this;
        }
    };

    Q_OBJECT
    Q_PROPERTY(QSlider::TickPosition TickPosition READ TickPosition WRITE setTickPosition)
    Q_PROPERTY(QSlider::TickPosition TickLabelPosition READ TickLabelPosition WRITE setTickLabelPosition)
    Q_PROPERTY(char TickLabelFormat READ TickLabelFormat WRITE setTickLabelFormat)
    Q_PROPERTY(int TickLabelPrecision READ TickLabelPrecision WRITE setTickLabelPrecision)
    Q_PROPERTY(int NumberOfTicks READ NumberOfTicks WRITE setNumberOfTicks)
    Q_PROPERTY(int ColorBarSize READ ColorBarSize WRITE setColorBarSize)
    Q_PROPERTY(Qt::Orientation orientation READ orientation WRITE setOrientation)
    Q_PROPERTY(range Range READ Range WRITE setRange NOTIFY RangeChanged)
    Q_PROPERTY(double RangeMin READ RangeMin WRITE setRangeMin NOTIFY RangeMinChanged)
    Q_PROPERTY(double RangeMax READ RangeMax WRITE setRangeMax NOTIFY RangeMaxChanged)

public:
    ColorBar(QWidget * = NULL); // vertical orientation default
    ColorBar(Qt::Orientation, QWidget * = NULL);

    virtual void setOrientation(Qt::Orientation o);
    Qt::Orientation orientation() const { return d_orientation; }

    void setColorTable(const QVector<QRgb> & table);
    QVector<QRgb> & ColorTable(){ return ctable; }
    const QVector<QRgb> & ColorTable() const { return ctable; }

    void setColorBarSize(int size);
    int ColorBarSize() const { return sz; }

    void setNumberOfTicks(const int ticks);
    int NumberOfTicks() const { return nticks; }

    void setTickPosition(const QSlider::TickPosition pos);
    QSlider::TickPosition TickPosition() const { return tickp; }

    void setTickLabelPosition(const QSlider::TickPosition pos);
    QSlider::TickPosition TickLabelPosition() const { return labelp; }

    double RangeMin() const { return r.rmin; }
    double RangeMax() const { return r.rmax; }
    range Range() const { return r; }

    // see QString::number(double, char, int)
    char TickLabelFormat() const { return dformat; }
    int TickLabelPrecision() const { return precs; }

signals:
    void selected(const QColor &);
    void RangeChanged(const range &);
    void RangeMinChanged(const double);
    void RangeMaxChanged(const double);

public slots:
    void setRangeMin(double rmin);
    void setRangeMax(double rmax);
    void setRange(double rmin, double rmax);
    void setRange(range r);

    void setTickLabelFormat(char format);
    void setTickLabelPrecision(int precision);

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void paintEvent(QPaintEvent *);

    void drawColorBar(QPainter *, const QRect &) const;
    void drawColorBarTicks(QPainter *, const QRect &) const;
    void drawColorBarTickLabels(QPainter *, const QRect &) const;

    void ColorBarTickLine(unsigned int i, bool above, int & x1, int & y1, int & x2, int & y2,
                          const QRect &rect) const;
    QRect ColorBarRect() const;

private:
    Qt::Orientation d_orientation;
    QVector<QRgb> ctable;
    int nticks = 2;
    QSlider::TickPosition tickp, labelp;
    int sz = 25;
    range r;
    char dformat = 'g';
    int precs = 3;

    void setup();
    void positions(bool & above, bool & below, QSlider::TickPosition pos) const;
};

#endif // COLORBAR_H
