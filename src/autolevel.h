/*
This file is part of Fast Track.

    FastTrack is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    FastTrack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FastTrack.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef AUTOLEVEL_H
#define AUTOLEVEL_H

#include <QMap>
#include <opencv2/core/types.hpp>
#include <string>
#include "data.h"
#include "tracking.h"

using namespace std;

class AutoLevel : public QObject {
  Q_OBJECT

  Data *data; /*!< Tracking data. */

  double computeStdAngle(const Data &data);
  double computeStdDistance(const Data &data);
  double computeStdArea(const Data &data);
  double computeStdPerimeter(const Data &data);

 public:
  AutoLevel(string path, UMat background);
  ~AutoLevel();

 public slots:
  QMap<QString, double> level();
};

#endif