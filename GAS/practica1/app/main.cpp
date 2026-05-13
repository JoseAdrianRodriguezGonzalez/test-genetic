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
  metodo m = {"unico", "ruleta"};
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
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES);
  std::vector<double> ruleta, torneo;
  for (int e = 0; e < 100; e++) {
    ga.fit(f, m, MAXIMIZAR);
    ruleta.push_back(ga.ObtenerMejorObjetivo());
  }
  algoritmogeneticsimple ga2 = algoritmogeneticsimple(
      experimentos.TAMAÑO_POBLACION, experimentos.NUM_GENES,
      experimentos.BITS_POR_GEN, experimentos.LIMITE_SUPERIOR,
      experimentos.LIMITE_INFERIOR, experimentos.PROB_CRUZA,
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES);
  m.seleccion = "torneo";
  for (int e = 0; e < 100; e++) {
    ga2.fit(f, m, MAXIMIZAR);
    torneo.push_back(ga2.ObtenerMejorObjetivo());
  }

  DataFrame df;
  df.add_column("ruleta", ruleta);
  df.add_column("torneo", torneo);

  std::cout << df.rows();
  df.to_csv("data.csv");
}
