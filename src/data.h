#ifndef DATA_H
#define DATA_H

#include <QDebug>
#include <QDir>
#include <QFile>
#include <QMap>
#include <QMapIterator>
#include <QString>
#include <QTextStream>
#include <QWidget>

struct object {
  int id;
  QMap<QString, double> data;
};

class Data {
 private:
  QString dir;
  QMap<int, QVector<object>> data;

 public:
  explicit Data(QString dataPath);
  ~Data();

  QVector<object> getData(int imageIndex);
  QMap<QString, double> getData(int imageIndex, int id);
  QList<int> getId(int imageIndex);
  void swapData(int firstObject, int secondObject, int from);
  void deleteData(int objectId, int from);
  void save();
};

#endif
