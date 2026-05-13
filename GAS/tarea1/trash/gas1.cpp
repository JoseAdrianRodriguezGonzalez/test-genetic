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

  return 0;
}
