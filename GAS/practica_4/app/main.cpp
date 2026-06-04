#include "experiments/program.hpp"
#include "functions/benchmark.hpp"
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
  COD_TYPE tc = REAL;
  algoritmogeneticsimple ga = algoritmogeneticsimple(
      tc, experimentos.TAMAÑO_POBLACION, experimentos.NUM_GENES,
      experimentos.BITS_POR_GEN, experimentos.LIMITE_SUPERIOR,
      experimentos.LIMITE_INFERIOR, experimentos.PROB_CRUZA,
      experimentos.PROB_MUTACION, experimentos.MAX_GENERACIONES);
  std::vector<double> objectiveExperiemnts;
  switch (experimentos.functionNumber) {
  case 1:
    for (int e = 0; e < 100; e++) {
      ga.fit(f1, m, MINIMIZAR);
      std::cout << "F1:\t";
      objectiveExperiemnts.push_back(ga.ObtenerMejorObjetivo());
    }
    break;
  case 2:
    for (int e = 0; e < 100; e++) {
      std::cout << "F2:\t";
      ga.fit(f5, m, MINIMIZAR);
      objectiveExperiemnts.push_back(ga.ObtenerMejorObjetivo());
    }
  case 3:
    for (int e = 0; e < 100; e++) {
      std::cout << "F3:\t";
      ga.fit(f13, m, MINIMIZAR);
      objectiveExperiemnts.push_back(ga.ObtenerMejorObjetivo());
    }
  }
  DataFrame df;
  df.add_column(std::to_string(experimentos.functionNumber),
                objectiveExperiemnts);
  std::cout << df.rows();
  df.to_csv("data_" + std::to_string(experimentos.functionNumber) + ".csv");
}
