#!/usr/bin/env bash
set -e
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
  mkdir $BUILD_DIR
fi
cd $BUILD_DIR

cmake ..

cmake --build .

# PRIMERA FUNCION (DataSet a) #
#./programa -gaussianas 5 -sup 3 1 0.3 -inf -2 0 0.05 -poblacion 500 -generaciones 2000 -cruza 0.8 -mutacion 0.01 -bits 20

# ruleta

./programa -variables 10 -sup 20 -inf -20 -poblacion 100 -generaciones 100000 -cruza 0.2 -mutacion 0.1 -bits 16

# torneo
#./programa -variables 10 -sup 20 -inf -20 -poblacion 100 -generaciones 100000 -cruza 0.2 -mutacion 0.1 -bits 16
