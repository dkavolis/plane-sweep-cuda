#include <QWidget>

class ColorBar: public QWidget
{
    Q_OBJECT

public:
    ColorBar(QWidget * = NULL); // vertical orientation default
    ColorBar(Qt::Orientation, QWidget * = NULL);

    virtual void setOrientation(Qt::Orientation o);
    Qt::Orientation orientation() const { return d_orientation; }

    void setColorTable(const QVector<QRgb> & table);
    const QVector<QRgb> & getColorTable(){ return ctable; }

signals:
    void selected(const QColor &);

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void paintEvent(QPaintEvent *);

    void drawColorBar(QPainter *, const QRect &) const;

private:
    Qt::Orientation d_orientation;
    QVector<QRgb> ctable;
};
