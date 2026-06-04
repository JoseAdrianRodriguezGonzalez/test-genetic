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

#./programa -variables 30 -sup 5.12 -inf -5.12 -poblacion 1000 -generaciones 1000 -cruza 0.4 -mutacion 0.01 -bits 16 -fun 1 &
./programa -variables 30 -sup 30 -inf -30 -poblacion 2000 -generaciones 4000 -cruza 0.97 -mutacion 0.01 -bits 16 -fun 2
#./programa -variables 30 -sup 5.0 -inf -5.0 -poblacion 10 -generaciones 500 -cruza 0.85 -mutacion 0.05 -bits 16 -fun 3
wait
echo "Todos los procesos terminaron exitosamente"
# torneo
#./programa -variables 10 -sup 20 -inf -20 -poblacion 100 -generaciones 100000 -cruza 0.2 -mutacion 0.1 -bits 16
