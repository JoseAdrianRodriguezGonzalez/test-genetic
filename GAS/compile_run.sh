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
./main_experiment
