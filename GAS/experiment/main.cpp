#include "../lib/algorithms/elitist.hpp"
#include "../lib/benchmark/benchmark.hpp"
#include <cmath>
#include <iostream>
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
typedef struct {
  hyperparams h;
  float mean_error;
} Result;
#define DIM 30;
hyperparams sample(const HyperSpace &s);
hyperparams *generate_samples(const HyperSpace &s, int N);

Result RandomSearch(GA &g, const HyperSpace &s, const unsigned int &guess,
                    const int &runs, gen_init *gi, float &f_best);
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
void hyperparameter_finetuning(Result *r, GA *g, const unsigned int &problems,
                               const HyperSpace &s, const unsigned int &guess,
                               const int &runs, gen_init *gi, float *optimos);
int main() {
  gen_init g[3];
  const float LS[3] = {5.12, 30, 50};
  const float LI[3] = {-5.12, -30, -50};
  ObjectiveFunction functions[3] = {f1, f5, f13};
  float f_best[3] = {0, 0, -1.1428};
  GA *solver = new GA[3];
  init_algorithm(solver, g, 3, LS, LI, functions, 16, 30);
  HyperSpace s = {10, 1000, 10, 1000, 0.01, 1, 0.01, 1};
  Result *r = new Result[3];
  hyperparameter_finetuning(r, solver, 3, s, 1000, 5, g, f_best);
  for (int i = 0; i < 3; i++) {
    std::cout << "Error=" << r[i].mean_error << std::endl;
    std::cout << "Parametros:" << std::endl;
    std::cout << "generaciones: " << r[i].h.num_generaciones;
    std::cout << "\t cruza: " << r[i].h.prob_cruza;
    std::cout << "\t mut: " << r[i].h.prob_mut;
    std::cout << "\t pob: " << r[i].h.tamp_pob << std::endl;
  }
  for (int i = 0; i < 3; i++) {
    delete[] g[i].numbits;
    delete[] g[i].li;
    delete[] g[i].ls;
  }
  delete[] solver;
  delete[] r;
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
Result RandomSearch(GA &g, const HyperSpace &s, const unsigned int &guess,
                    const int &runs, gen_init *gi, float &f_best) {
  hyperparams *h = generate_samples(s, guess);
  Result best;
  best.mean_error = 1e9;
  for (int i = 0; i < guess; i++) {
    float avg_error = 0;
    std::cout << "iteration :" << i << std::endl;
    for (int r = 0; r < runs; r++) {
      std::cout << "Run :" << r << "\t";
      g.free_memory();
      g.initialization_ga(h[i].tamp_pob, gi->len, gi->numbits, gi->li, gi->ls);
      g.run(h[i].num_generaciones, MIN, h[i].prob_mut, h[i].prob_cruza);
      float f_optim = g.getfitness(g.getMax());
      avg_error += std::fabs(f_best - f_optim);
      std::cout << "f(";
      g.printRealValues(g.getMax());
      std::cout << ")= " << f_optim << std::endl;
    }
    avg_error /= runs;
    if (avg_error < best.mean_error) {
      best.mean_error = avg_error;
      best.h = h[i];
    }
  }
  delete[] h;
  return best;
}
void hyperparameter_finetuning(Result *r, GA *g, const unsigned int &problems,
                               const HyperSpace &s, const unsigned int &guess,
                               const int &runs, gen_init *gi, float *optimos) {
  for (int i = 0; i < problems; i++) {
    r[i] = RandomSearch(g[i], s, guess, runs, &gi[i], optimos[i]);
  }
}
