#include "../lib/algorithms/elitist.hpp"
#include "../lib/benchmark/benchmark.hpp"
#include <random>
typedef struct {
  unsigned int *numbits;
  float *li;
  float *ls;
  unsigned int len;
} gen_init;
typedef struct {
  unsigned int tamp_pob;
  unsigned int num_generaciones;
  float prob_cruza;
  float prob_mut;
} hyperparams;
typedef struct {
  unsigned int pop_min, pop_max;
  unsigned int gen_min, gen_max;
  float cross_min, cross_max;
  float mut_min, mut_max;
} HyperSpace;
#define DIM 30;
hyperparams sample(const HyperSpace &s);
hyperparams *generate_samples(const HyperSpace &s, int N);

void RandomSearch() {}
template <typename T> void fill_array(T *arr, unsigned int len, T value) {
  T *ptr = arr;
  T *end = arr + len;
  while (ptr < end)
    *ptr++ = value;
}
void init_params(gen_init *p, const unsigned int &bits, const float &limit_i,
                 const float &limit_sup, const unsigned int len);
void init_algorithm(const GA *solvers, gen_init *g, unsigned int problems,
                    const float *ls, const float *li,
                    const ObjectiveFunction *functions,
                    const unsigned int &bits, const unsigned int &dim);
int main() {
  gen_init g[3];
  const float LS[3] = {5.12, 30, 50};
  const float LI[3] = {-5.12, -30, -50};
  ObjectiveFunction functions[3] = {f1, f5, f13};
  GA *solver = new GA[3];
  init_algorithm(solver, g, 3, LS, LI, functions, 16, 30);
}
void init_params(gen_init *p, const unsigned int &bits, const float &limit_i,
                 const float &limit_sup, const unsigned int len) {
  p->len = len;
  p->numbits = new unsigned int[len];
  p->li = new float[len];
  p->ls = new float[len];
  fill_array(p->numbits, p->len, bits);
  fill_array(p->li, p->len, limit_i);
  fill_array(p->ls, p->len, limit_sup);
}

void init_algorithm(GA *solvers, gen_init *g, unsigned int problems,
                    const float *ls, const float *li,
                    const ObjectiveFunction *functions,
                    const unsigned int &bits, const unsigned int &dim) {
  for (int i = 0; i < problems; i++) {
    init_params(&g[i], bits, li[i], ls[i], dim);
  }

  for (int i = 0; i < problems; i++) {
    solvers[i].initialization_ga(100, g[i].len, g[i].numbits, g[i].li, g[i].ls,
                                 functions[i]);
  }
}
hyperparams sample(const HyperSpace &s) {
  hyperparams h;
  h.tamp_pob = rand() % (s.pop_max - s.pop_min) + s.pop_min;
  h.num_generaciones = rand() % (s.gen_max - s.gen_min) + s.gen_min;
  h.prob_mut =
      s.cross_min + (float)rand() / RAND_MAX * (s.cross_max - s.cross_min);
  h.prob_cruza = s.mut_min + (float)rand() / RAND_MAX * (s.mut_max - s.mut_min);
  return h;
}
hyperparams *generate_samples(const HyperSpace &s, int N) {
  hyperparams *h = new hyperparams[N];
  for (int i = 0; i < N; i++) {
    *(h + i) = sample(s);
  }
  return h;
}
