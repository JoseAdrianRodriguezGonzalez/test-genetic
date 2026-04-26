for i in {1..10}; do
  echo "Run $i" >>resultados.txt
  ./salida >>resultados.txt
  echo "-----------------------------" >>resultados.txt
done
