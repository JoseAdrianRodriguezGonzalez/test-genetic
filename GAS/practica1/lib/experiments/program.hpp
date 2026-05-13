#include <cstring>
#include <iostream>
#include <vector>
using namespace std;
struct Programa {
  unsigned int NUM_GENES;
  unsigned int m;
  vector<unsigned int> BITS_POR_GEN;
  unsigned int TAMAÑO_POBLACION;
  unsigned int MAX_GENERACIONES;
  vector<float> LIMITE_SUPERIOR;
  vector<float> LIMITE_INFERIOR;
  vector<float> LIMITE_SUPERIOR_BASE;
  vector<float> LIMITE_INFERIOR_BASE;
  double PROB_CRUZA;
  double PROB_MUTACION;
};
Programa Asignacion(int argc, char *argv[]);
