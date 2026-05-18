#include "experiments/program.hpp"
#include "ga/sga.hpp"
#include "io/reader.h"
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
int main(int argc, char *argv[]) {

  srand((unsigned int)(time(NULL)));
  Programa experimentos = Asignacion(argc, argv);
  metodo m = {"unico", "torneo"};
  auto f = [](const std::vector<float> &x) -> float {
    return 1000 - std::pow(x[0] + 7.5, 2) - std::pow(x[1] + 3, 2) -
           std::pow(x[2] - 3, 2) - std::pow(x[3] - 5, 2) -
           std::pow(x[4] + 2.5, 2) - std::pow(x[5] - 10, 2) -
           std::pow(x[6] - 15, 2) - std::pow(x[7] + 10, 2) -
           std::pow(x[8] + 15, 2) - std::pow(x[9] - 0.5, 2);
  };
  algoritmogeneticsimple ga = algoritmogeneticsimple(
      experimentos.TAMAÑO_POBLACION, experimentos.NUM_GENES,
      experimentos.BITS_POR_GEN, experimentos.LIMITE_SUPERIOR,
      experimentos.LIMITE_INFERIOR, experimentos.PROB_CRUZA,
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES, 2);
  std::vector<std::vector<double>> numeros(4);
  for (int e = 0; e < 100; e++) {
    ga.fit(f, m, MAXIMIZAR);
    numeros[0].push_back(ga.ObtenerMejorObjetivo());
  }
  algoritmogeneticsimple ga2 = algoritmogeneticsimple(
      experimentos.TAMAÑO_POBLACION, experimentos.NUM_GENES,
      experimentos.BITS_POR_GEN, experimentos.LIMITE_SUPERIOR,
      experimentos.LIMITE_INFERIOR, experimentos.PROB_CRUZA,
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES, 4);
  for (int e = 0; e < 100; e++) {
    ga2.fit(f, m, MAXIMIZAR);
    numeros[1].push_back(ga2.ObtenerMejorObjetivo());
  }

  algoritmogeneticsimple ga3 = algoritmogeneticsimple(
      experimentos.TAMAÑO_POBLACION, experimentos.NUM_GENES,
      experimentos.BITS_POR_GEN, experimentos.LIMITE_SUPERIOR,
      experimentos.LIMITE_INFERIOR, experimentos.PROB_CRUZA,
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES, 8);
  for (int e = 0; e < 100; e++) {
    ga3.fit(f, m, MAXIMIZAR);
    numeros[2].push_back(ga3.ObtenerMejorObjetivo());
  }

  algoritmogeneticsimple ga4 = algoritmogeneticsimple(
      experimentos.TAMAÑO_POBLACION, experimentos.NUM_GENES,
      experimentos.BITS_POR_GEN, experimentos.LIMITE_SUPERIOR,
      experimentos.LIMITE_INFERIOR, experimentos.PROB_CRUZA,
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES, 16);
  for (int e = 0; e < 100; e++) {
    ga4.fit(f, m, MAXIMIZAR);
    numeros[3].push_back(ga4.ObtenerMejorObjetivo());
  }
  DataFrame df;
  for (size_t i = 0; i < numeros.size(); i++) {
    df.add_column(std::to_string(i), numeros[i]);
  }

  std::cout << df.rows();
  df.to_csv("data.csv");
}
