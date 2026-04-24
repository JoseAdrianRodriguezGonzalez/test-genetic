#include "lib/gas.hpp"
#include <cstdlib>
#include <ctime>
int main() {
  unsigned int t = 1;
  time_t tx;
  srand((unsigned)std::time(&tx));
  //  unsigned int NUMBITS[10] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  //  float li[10] = {-20, -20, -20, -20, -20, -20, -20, -20, -20, -20};
  //  float ls[10] = {20, 20, 20, 20, 20, 20, 20, 20, 20, 20};
  unsigned int NUMBITS[2] = {16, 16};
  float li[2] = {-500, -500};
  float ls[2] = {500, 500};
  GA gas(60, 2, NUMBITS, li, ls);
  gas.decode_integer();
  gas.decode_real();
  gas.EvaluatePoblation();
  gas.obj_to_fit(MIN);
  while (t <= MAX_GENERCIONES) {
    gas.Roulette();
    gas.cruza1P(0.85);
    gas.muta(0.005);
    gas.Elitismo();
    gas.NextGeneration();
    gas.decode_integer();
    gas.decode_real();
    gas.EvaluatePoblation();
    gas.obj_to_fit(MIN);
    t++;
  }
  std::cout << "Best\n";
  gas.print_individual(gas.getMax());
  return 0;
}
