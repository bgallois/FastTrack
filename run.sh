#!/bin/bash

set -e
qmake Fishy.pro
make
cd build
./Fishy
cd ..
