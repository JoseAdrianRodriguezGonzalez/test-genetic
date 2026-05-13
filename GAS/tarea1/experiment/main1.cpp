#include "../lib/algorithms/elitist.hpp"
#include "../lib/benchmark/benchmark.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
void log_result(const std::string &file, float fitness);
bool read_results(const std::string &file, int limit);
void pipeline_process(int val) {
  int func_id = val;
  time_t tx;
  srand((unsigned)std::time(&tx));

  unsigned int dim = 30;
  const float LS[3] = {5.12, 30, 50};
  const float LI[3] = {-5.12, -30, -50};
  ObjectiveFunction functions[3] = {f1, f5, f13};
  unsigned int *bits = new unsigned int[30];
  float *limit_sup = new float[dim];
  float *limit_inf = new float[dim];
  unsigned int pob, genes, generacion;
  float mut, cruza;
  int limit = 100;
  std::string file = "../results/f" + std::to_string(func_id + 1) + ".txt";
  std::cout << "Ga para " << func_id << std::endl;
  switch (func_id) {
  case 0:
    for (int i = 0; i < 30; i++)
      bits[i] = 16;
    for (int i = 0; i < dim; i++)
      limit_inf[i] = LI[0];
    for (int i = 0; i < dim; i++)
      limit_sup[i] = LS[0];
    pob = 100;
    genes = 30;
    generacion = 5000;
    mut = 0.01;
    cruza = 0.2;

    break;
  case 1:
    for (int i = 0; i < 30; i++)
      bits[i] = 16;
    for (int i = 0; i < dim; i++)
      limit_inf[i] = LI[1];
    for (int i = 0; i < dim; i++)
      limit_sup[i] = LS[1];
    pob = 8;
    genes = 30;
    generacion = 300000;
    mut = 0.001;
    cruza = 0.9;
    break;
  case 2:
    for (int i = 0; i < 30; i++)
      bits[i] = 16;
    for (int i = 0; i < dim; i++)
      limit_inf[i] = LI[2];
    for (int i = 0; i < dim; i++)
      limit_sup[i] = LS[2];
    pob = 300;
    genes = 30;
    generacion = 2500;
    mut = 0.01;
    cruza = 0.5;
    break;
  };
  std::cout << "si iniicalizacio";
  for (int e = 0; e < 100; e++) {
    GA g(pob, genes, bits, limit_inf, limit_sup, functions[func_id]);
    g.run(generacion, MIN, mut, cruza);
    float fitness = g.getObjetive(g.getMax());
    log_result(file, fitness);
  }
  if (read_results(file, limit)) {
    std::cout << "Ya no es necesario correr mas";
    return;
  }
}

void read_results(const std::string &file, int quant, float *val);
float media(float *values, int len);
float std_(float *values, int len);
int main(int argc, char **argv) {

  // pipeline_process(atoi(argv[1]));
  float **val = new float *[3];

  const std::string files[3] = {"../results/f1.txt", "../results/f2.txt",
                                "../results/f3.txt"};
  const std::string outfiles[3] = {"../results/f1_est.txt",
                                   "../results/f2_est.txt",
                                   "../results/f3_est.txt"};

  for (int i = 0; i < 3; i++) {
    val[i] = new float[100];
    read_results(files[i], 100, val[i]);
    float mu = media(val[i], 100);
    float sigma = std_(val[i], 100);
    log_result(outfiles[i], mu);
    log_result(outfiles[i], sigma);
  }
  return 0;
  // calcular estradisticos
}
float media(float *values, int len) {
  float sum = 0;

  for (int i = 0; i < len; i++) {
    sum += values[i];
  }
  return sum / len;
}
float std_(float *values, int len) {
  float sum = 0;
  float mu = media(values, len);
  for (int i = 0; i < len; i++) {
    float term = values[i] - mu;
    sum += (term * term);
  }

  return std::sqrt(sum / (float)len);
}
bool read_results(const std::string &file, int limit) {
  std::ifstream in(file);

  if (!in.is_open()) {
    std::cout << "No se pudo abrir el archivo\n";
    return false;
  }
  std::string line;
  int count = 0;
  while (std::getline(in, line)) {
    if (count >= limit) {
      return true;
    }

    std::cout << "Linea " << count << ": " << line << std::endl;
    count++;
  }

  std::cout << "Total leido: " << count << std::endl;
  return false;
}
void log_result(const std::string &file, float fitness) {

  std::ofstream out(file, std::ios::app);
  if (!out.is_open()) {
    std::cout << "ERROR: no se pudo abrir " << file << std::endl;
    return;
  }
  out << fitness << "\n";
  std::cout << "escrito con exito el archiv en " << file;
}
void read_results(const std::string &file, int quant, float *val) {

  std::ifstream in(file);
  if (!in.is_open()) {
    std::cout << "No se pudo abrir";
    return;
  }
  std::string line;
  int count = 0;
  while (std::getline(in, line)) {
    if (count == quant) {
      break;
    }
    val[count] = std::stof(line);
    count++;
  }
  return;
}
/*FUncion 1 bits 16
 * GA g(100, 30, bits, limit_inf, limit_sup, functions[0]);
  g.run(5000, MIN, 0.01, 0.2);

 *
 *
 *
 * Funcion 2
 *8
 *  GA g(8, 30, bits, limit_inf, limit_sup, functions[1]);

  g.run(300000, MIN, 0.001, 0.9);


  Fucnion 3
16 bits
 GA g(300, 30, bits, limit_inf, limit_sup, functions[2]);

  g.run(2500, MIN, 0.01, 0.5);
 * */
