#include <QWidget>
#include <QSlider>

class ColorBar: public QWidget
{
    Q_OBJECT

public:
    ColorBar(QWidget * = NULL); // vertical orientation default
    ColorBar(Qt::Orientation, QWidget * = NULL);

    virtual void setOrientation(Qt::Orientation o);
    Qt::Orientation orientation() const { return d_orientation; }

    void setColorTable(const QVector<QRgb> & table);
    QVector<QRgb> & getColorTable(){ return ctable; }
    const QVector<QRgb> & getColorTable() const { return ctable; }

    void setNumberOfTicks(const int ticks);
    void setTickPosition(const QSlider::TickPosition pos);
    int getNumberOfTicks() const { return nticks; }
    QSlider::TickPosition getTickPosition() const {return tickp; }

signals:
    void selected(const QColor &);

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void paintEvent(QPaintEvent *);

    void drawColorBar(QPainter *, const QRect &) const;
    void drawColorBarTicks(QPainter *, const QRect &) const;

private:
    Qt::Orientation d_orientation;
    QVector<QRgb> ctable;
    int nticks = 2;
    QSlider::TickPosition tickp;
};
