#include "../../lib/algorithms/elitist.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
GA::GA(unsigned int TAM_POBLACION, unsigned int NUM_GENES,
       unsigned int *NumBITsxGEN, float *Linf, float *Lsup,
       ObjectiveFunction f) {
  this->initialization_ga(TAM_POBLACION, NUM_GENES, NumBITsxGEN, Linf, Lsup, f);
}
GA::GA() {}
void GA::initialization_ga(unsigned int TAM_POBLACION, unsigned int NUM_GENES,
                           unsigned int *NumBITsxGEN, float *Linf, float *Lsup,
                           ObjectiveFunction f) {

  this->f = f;
  this->TAM_POBLACION = TAM_POBLACION;
  this->NUM_GENES = NUM_GENES;
  this->POB = new INDIVIDUO[this->TAM_POBLACION];
  this->NewPOB = new INDIVIDUO[this->TAM_POBLACION];
  this->NumBITsxGEN = new unsigned int[this->NUM_GENES];
  for (size_t i = 0; i < this->NUM_GENES; i++) {
    this->NumBITsxGEN[i] = NumBITsxGEN[i];
    std::cout << this->NumBITsxGEN[i] << std::endl;
  }
  this->Limit_Inf = new float[NUM_GENES];
  for (size_t i = 0; i < NUM_GENES; i++) {
    this->Limit_Inf[i] = Linf[i];
  }
  std::cout << " se crea el liminf\n";
  this->Limit_Sup = new float[NUM_GENES];
  for (size_t i = 0; i < NUM_GENES; i++) {
    this->Limit_Sup[i] = Lsup[i];
  }
  std::cout << " se crea el limsup\n";
  this->ChromeSize = 0;
  Chromsize_();
  idMax = 0.0;
  idMin = 0.0;
  initialize_population();
  std::cout << " se inicio\n";
  decode_integer();
  decode_real();
}
void GA::Chromsize_() {
  for (int k = 0; k < NUM_GENES; k++)
    this->ChromeSize += NumBITsxGEN[k];
}

void GA::initialize_population() {
  this->seleccionados = new unsigned int[TAM_POBLACION];
  for (int k = 0; k < this->TAM_POBLACION;
       k++) { // Reservar memoria para cada cadena Binaria de CROMOSOMA
    this->POB[k].Chrom = new BYTE[this->ChromeSize];
    this->POB[k].Vent = new unsigned int[NUM_GENES];

    this->POB[k].Vre = new float[NUM_GENES];

    this->NewPOB[k].Chrom = new BYTE[this->ChromeSize];

    this->NewPOB[k].Vent = new unsigned int[NUM_GENES];

    this->NewPOB[k].Vre = new float[NUM_GENES];

    // Inicializar el cromosoma
    for (int i = 0; i < this->ChromeSize; i++) {
      this->POB[k].Chrom[i] = rand() % 2;
      this->NewPOB[k].Chrom[i] = 0;
    }
    // Inicializar los seleccionados

    for (int i = 0; i < this->TAM_POBLACION; i++) {
      this->seleccionados[i] = 0;
    }
    for (int j = 0; j < NUM_GENES; j++) {
      this->POB[k].Vent[j] = 0;
      this->POB[k].Vre[j] = 0.0;
      this->NewPOB[k].Vent[j] = 0;
      this->NewPOB[k].Vre[j] = 0.0;
    }
    this->POB[k].VObj = 0;
    this->POB[k].Vfit = 0;
    this->NewPOB[k].VObj = 0;
    this->NewPOB[k].Vfit = 0;
  }
}
void GA::cruza1P(float C) {
  unsigned int k, i, Pc;
  unsigned int Limit = ChromeSize - 1;
  unsigned int Padre1, Padre2;
  float r;

  //  for (k = 0; k < TAM_POBLACION; k++)
  //    std::cout << seleccionados[k] << "\n";
  for (k = 0; k < TAM_POBLACION; k += 2) {
    r = (float)rand() / RAND_MAX;
    if (r < C) {

      Padre1 = seleccionados[k];
      Padre2 = seleccionados[k + 1];
      NewPOB[k].Padre1 = Padre1;
      NewPOB[k].Padre2 = Padre2;
      NewPOB[k + 1].Padre1 = Padre2;
      NewPOB[k + 1].Padre2 = Padre1;
      Pc = rand() % (Limit);
      for (i = 0; i <= Pc; i++) {
        NewPOB[k].Chrom[i] = POB[Padre1].Chrom[i];
        NewPOB[k + 1].Chrom[i] = POB[Padre2].Chrom[i];
      }
      for (i = Pc + 1; i < ChromeSize; i++) {
        NewPOB[k].Chrom[i] = POB[Padre2].Chrom[i];
        NewPOB[k + 1].Chrom[i] = POB[Padre1].Chrom[i];
      }
    } else {
      Padre1 = seleccionados[k];
      Padre2 = seleccionados[k + 1];
      NewPOB[k].Padre1 = Padre1;
      NewPOB[k].Padre2 = Padre1;
      NewPOB[k + 1].Padre1 = Padre2;
      NewPOB[k + 1].Padre2 = Padre2;
      // NewPOB[k].Padre1 = Padre1;
      // NewPOB[k].Padre2 = Padre2;
      // NewPOB[k + 1].Padre1 = Padre1;
      // NewPOB[k + 1].Padre2 = Padre2;
      for (i = 0; i < ChromeSize; i++) {
        NewPOB[k].Chrom[i] = POB[Padre1].Chrom[i];
        NewPOB[k + 1].Chrom[i] = POB[Padre2].Chrom[i];
      }
    }
  }
}
void GA::muta(float prob) {
  float r;
  for (int k = 0; k < TAM_POBLACION; k++) {
    for (int i = 0; i < ChromeSize; i++) {
      r = (float)rand() / RAND_MAX;
      if (r < prob) {
        NewPOB[k].Chrom[i] = 1 - NewPOB[k].Chrom[i];
      }
    }
  }
}
void GA::decode_integer() {
  for (int k = 0; k < this->TAM_POBLACION; k++) {
    unsigned int Acumulado = 0, g = 0, aux = 0;
    Acumulado += NumBITsxGEN[g];
    for (int i = 0, j = 0; i < ChromeSize; i++, j++) {
      aux |= POB[k].Chrom[i] << j;
      if (i == (Acumulado - 1)) {
        POB[k].Vent[g] = aux;
        aux = 0;
        g++;
        Acumulado += NumBITsxGEN[g];
        j = -1;
      }
    }
  }
}
void GA::decode_real() {
  for (int k = 0; k < TAM_POBLACION; k++) {
    float rango = 0;
    unsigned int Den;
    for (int j = 0; j < NUM_GENES; j++) {
      rango = Limit_Sup[j] - Limit_Inf[j];
      Den = (1ULL << NumBITsxGEN[j]) - 1;
      POB[k].Vre[j] = (((float)POB[k].Vent[j] / Den) * rango) + Limit_Inf[j];
    }
  }
}
void GA::Roulette() {
  unsigned int k, seleccionado = 0;
  float Suma = 0, pelota;
  float PAcum[TAM_POBLACION];
  PAcum[0] = POB[0].Vfit / SumFit;
  for (k = 0; k < TAM_POBLACION; k++) {
    Suma += POB[k].Vfit / SumFit;
    PAcum[k] = Suma;
  }
  for (k = 0; k < TAM_POBLACION; k++) {
    pelota = (double)rand() / RAND_MAX;
    seleccionado = 0;
    while (pelota > PAcum[seleccionado])
      seleccionado++;
    seleccionados[k] = seleccionado;
  }
}

void GA::EvaluatePoblation() {
  idMax = 0;
  idMin = 0;
  SumObj = 0;
  for (int k = 0; k < this->TAM_POBLACION; k++) {
    POB[k].VObj = f(POB[k].Vre, NUM_GENES);
    if (POB[k].VObj > POB[idMax].VObj) {
      idMax = k;
    }
    if (POB[k].VObj < POB[idMin].VObj) {
      idMin = k;
    }
    SumObj += POB[k].VObj;
  }

  PromObj = SumObj / TAM_POBLACION;
}

/*
 *f(x)=250-(x-115)**2
 *
 * */
void GA::obj_to_fit(OPT_TYPE Tipo) {
  float rango = POB[this->idMax].VObj - POB[this->idMin].VObj + 0.00001;
  SumFit = 0;
  if (Tipo == MAX) {
    for (int k = 0; k < this->TAM_POBLACION; k++) {
      POB[k].Vfit = 100 * (POB[k].VObj - POB[this->idMin].VObj) / rango;
      SumFit += POB[k].Vfit;
    }
    PromFit = SumFit / TAM_POBLACION;
  }
  if (Tipo == MIN) {
    for (int k = 0; k < this->TAM_POBLACION; k++) {
      POB[k].Vfit = 100 * (POB[this->idMax].VObj - POB[k].VObj) / rango;
      SumFit += POB[k].Vfit;
    }
    PromFit = SumFit / TAM_POBLACION;
    int aux = idMin;
    idMin = idMax;
    idMax = aux;
  }
}
// f(x)=1000-(x-3.15)^2-(y+0.5)^2
// Max(f(x,y))=1000 en (x=+3.15,y=-0.5)
// float GA::objectiveFunction(float x) { return abs((x + 2) * (x - 25)); }
/*
float GA::objectiveFunction(unsigned int id) {
  float obj = 0, aux;
  // obj = 1000 - pow(x - 3.15, 2) - pow(y + 0.5, 2);
    obj = 1000 - pow(POB[id].Vre[0] + 7, 2) - pow(POB[id].Vre[1] + 3, 2) -
          pow(POB[id].Vre[2] - 3, 2) - pow(POB[id].Vre[3] - 5, 2) -
          pow(POB[id].Vre[4] + 2.5, 2) - pow(POB[id].Vre[5] - 10, 2) -
          pow(POB[id].Vre[6] - 15, 2) - pow(POB[id].Vre[7] + 10, 2) -
          pow(POB[id].Vre[8] + 15, 2) - pow(POB[id].Vre[9] - 0.5, 2);
          */
//  float x = POB[id].Vre[0], y = POB[id].Vre[1];
//  FUncion 9
/*aux = 0;
for (int i = 0; i < 2; i++)
  aux += pow(POB[id].Vre[k], 2) - 10 * cos(6.2838185 * POB[id].Vre[k]) + 10;
  obj = aux;
  return obj;

//  obj =// funcion tarea
//      10 * std::exp(-((x + 1) * (x + 1) + (y - 3.14) * (y - 3.14)) / (5 *
//      5)) + cos(2 * x) + sin(2 * y);
// FUncion objetivo schwefel
for (int k = 0; k < this->NUM_GENES; k++)
  obj += POB[id].Vre[k] * sin(sqrt(fabs(POB[id].Vre[k])));

// return 418.9829 * this->NUM_GENES - obj;
return -obj;
}
*/
void GA::print_new() {

  for (int k = 0; k < TAM_POBLACION; k++) {
    int Acumulado = ChromeSize - 1, g = NUM_GENES - 1;
    std::cout << "[" << k << "]";
    // Acumulado+=NumBITsxGEN[g];
    for (int i = ChromeSize - 1; i >= 0; i--) {
      if (i == Acumulado) {
        std::cout << ":";
        Acumulado -= NumBITsxGEN[g];
        g--;
      }
      std::cout << (int)NewPOB[k].Chrom[i];
    }
    std::cout << " Padre 1: " << NewPOB[k].Padre1
              << " Padre 2: " << NewPOB[k].Padre2;
    std::cout << std::endl;
  }
}
void GA::NextGeneration(void) {
  INDIVIDUO *aux;
  aux = this->POB;
  this->POB = this->NewPOB;
  this->NewPOB = aux;
}
void GA::print_population() {

  for (int k = 0; k < TAM_POBLACION; k++) {
    int Acumulado = ChromeSize - 1, g = NUM_GENES - 1;
    std::cout << "[" << k << "]";
    // Acumulado+=NumBITsxGEN[g];
    for (int i = ChromeSize - 1; i >= 0; i--) {
      if (i == Acumulado) {
        std::cout << ":";
        Acumulado -= NumBITsxGEN[g];
        g--;
      }
      std::cout << (int)POB[k].Chrom[i];
    }
    std::cout << " Valor_Entero:";
    for (int j = NUM_GENES - 1; j >= 0; j--)
      std::cout << "," << (int)POB[k].Vent[j];
    std::cout << "\t Valor_Real:";
    for (int j = NUM_GENES - 1; j >= 0; j--)
      std::cout << "," << (float)POB[k].Vre[j];
    std::cout << "\t Valor objetivo:";
    for (int j = NUM_GENES - 1; j >= 0; j--)
      std::cout << "," << (float)POB[k].VObj;
    std::cout << "\t Califcacion:";
    for (int j = NUM_GENES - 1; j >= 0; j--)
      std::cout << "," << (float)POB[k].Vfit;
    std::cout << std::endl;
  }
  std::cout << "EL peor individuo es :" << idMin;
  std::cout << "\nEL mejor individuo es :" << idMax;
  std::cout << "\nSuma de valor objetivo :" << SumObj;
  std::cout << "\nPromedio de valor objetivo :" << PromObj;
  std::cout << "\nSuma de valor de fitness:" << SumFit;
  std::cout << "\nPromedio de valor de fitness  :" << PromFit;
}
void GA::print_individual(unsigned int k) {
  int Acumulado = ChromeSize - 1, g = NUM_GENES - 1;
  std::cout << "[" << k << "]";
  // Acumulado+=NumBITsxGEN[g];
  for (int i = ChromeSize - 1; i >= 0; i--) {
    if (i == Acumulado) {
      std::cout << ":";
      Acumulado -= NumBITsxGEN[g];
      g--;
    }
    std::cout << (int)POB[k].Chrom[i];
  }
  std::cout << " Valor_Entero:";
  for (int j = NUM_GENES - 1; j >= 0; j--)
    std::cout << "," << (int)POB[k].Vent[j];
  std::cout << "\t Valor_Real:";
  for (int j = NUM_GENES - 1; j >= 0; j--)
    std::cout << "," << (float)POB[k].Vre[j];
  std::cout << "\t Valor objetivo:";
  std::cout << "," << (float)POB[k].VObj;
  std::cout << "\t Califcacion:";
  std::cout << "," << (float)POB[k].Vfit;
  std::cout << std::endl;
}
unsigned int GA::getMax() { return this->idMax; }
void GA ::Elitismo(void) {
  for (int i = 0; i < ChromeSize; i++) {
    NewPOB[0].Chrom[i] = POB[this->idMax].Chrom[i];
  }
}
void GA::run(const unsigned int &gen, OPT_TYPE opt, const float &prob_mut,
             const float &prob_cross) {
  unsigned int t = 1;
  this->decode_integer();
  this->decode_real();
  this->EvaluatePoblation();
  this->obj_to_fit(opt);
  while (t <= gen) {
    this->Roulette();
    this->cruza1P(prob_cross);
    this->muta(prob_mut);
    this->Elitismo();
    this->NextGeneration();
    this->decode_integer();
    this->decode_real();
    this->EvaluatePoblation();
    this->obj_to_fit(MIN);
    t++;
  }
  std::cout << "Best\n";
  this->print_individual(this->getMax());
}

GA::~GA() {
  for (int k = 0; k < this->TAM_POBLACION; k++) {
    delete[] POB[k].Chrom;
    delete[] POB[k].Vent;
    delete[] POB[k].Vre;
    delete[] NewPOB[k].Chrom;
    delete[] NewPOB[k].Vent;
    delete[] NewPOB[k].Vre;
  }
  delete[] POB;
  delete[] NewPOB;
  delete[] NumBITsxGEN;
  delete[] Limit_Inf;
  delete[] Limit_Sup;
  delete[] seleccionados;
}
void GA::free_memory() {
  for (int k = 0; k < TAM_POBLACION; k++) {
    delete[] POB[k].Chrom;
    delete[] POB[k].Vent;
    delete[] POB[k].Vre;
    delete[] NewPOB[k].Chrom;
    delete[] NewPOB[k].Vent;
    delete[] NewPOB[k].Vre;
  }
  delete[] POB;
  delete[] NewPOB;
  delete[] seleccionados;
}
void GA::resize_population(unsigned int new_size) {
  free_memory();
  this->TAM_POBLACION = new_size;
  this->POB = new INDIVIDUO[this->TAM_POBLACION];
  this->NewPOB = new INDIVIDUO[this->TAM_POBLACION];
  initialize_population();
}
