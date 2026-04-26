#!/bin/bash
set -e
echo "📁 Creando carpeta build..."
mkdir -p build
cd build
echo "⚙️ Ejecutando CMake..."
cmake ..
echo "🔨 Compilando..."
make -j$(nproc)
echo "🚀 Ejecutando programa..."
for j in $(seq 0 100); do
  for i in 0 1 2; do
    ./main_experiment $i &
  done
  wait
done
