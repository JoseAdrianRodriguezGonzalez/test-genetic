/*Simple Genetic Algorithm
  SGA.cpp
  Implementacion del algoritmo
  genetico simple
 */
#include <cmath>
#include <iostream>
#include <random>
//***** Parametros de configuracion del AG ****
typedef unsigned char BYTE;
typedef struct {
  BYTE *Chrom;
  unsigned int *Vent; // valores enteros decodificados
  float *Vre;         // valores reales decodificados
} INDIVIDUO;
using namespace std;
class GA {
private:
  INDIVIDUO *POB;
  unsigned int ChromeSize;
  unsigned int NUM_GENES;
  unsigned int *NumBITsxGEN;
  unsigned int TAM_POBLACION;
  float *Limit_Sup;
  float *Limit_Inf;
  void Chromsize_() {
    for (int k = 0; k < NUM_GENES; k++)
      this->ChromeSize += NumBITsxGEN[k];
  }
  void initialize_population() {
    for (int k = 0; k < TAM_POBLACION;
         k++) { // Reservar memoria para cada cadena Binaria de CROMOSOMA
      this->POB[k].Chrom = new BYTE[this->ChromeSize];
      this->POB[k].Vent = new unsigned int[NUM_GENES];
      this->POB[k].Vre = new float[NUM_GENES];
      // Inicializar el cromosoma
      for (int i = 0; i < this->ChromeSize; i++)
        this->POB[k].Chrom[i] = rand() % 2;
      for (int j = 0; j < NUM_GENES; j++) {
        this->POB[k].Vent[j] = 0;
        this->POB[k].Vre[j] = 0.0;
      }
    }
  }
  void decode_integer() {
    for (int k = 0; k < this->TAM_POBLACION; k++) {
      unsigned int Acumulado = 0, g = 0, aux = 0;
      Acumulado += NumBITsxGEN[g];
      for (int i = 0, j = 0; i < ChromeSize; i++, j++) {
        aux += POB[k].Chrom[i] * pow(2, j);
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
  void decode_real() {
    for (int k = 0; k < TAM_POBLACION; k++) {
      float rango = 0;
      unsigned int Den;
      for (int j = 0; j < NUM_GENES; j++) {
        rango = Limit_Sup[j] - Limit_Inf[j];
        Den = pow(2, NumBITsxGEN[j]) - 1;
        POB[k].Vre[j] = (((float)POB[k].Vent[j] / Den) * rango) + Limit_Inf[j];
      }
    }
  }

public:
  GA(unsigned int TAM_POBLACION, unsigned int NUM_GENES,
     unsigned int *NumBitsxaGen, float *Linf, float *Lsup) {
    cout << "iniciando";
    this->TAM_POBLACION = TAM_POBLACION;
    cout << this->TAM_POBLACION;
    this->NUM_GENES = NUM_GENES;
    this->NumBITsxGEN = new unsigned int[NUM_GENES];
    cout << "Parametro inicializados";
    for (size_t i = 0; i < NUM_GENES; i++) {
      this->NumBITsxGEN[i] = NumBitsxaGen[i];
    }
    this->Limit_Inf = new float[NUM_GENES];
    for (size_t i = 0; i < NUM_GENES; i++) {
      this->Limit_Inf[i] = Linf[i];
    }
    this->Limit_Sup = new float[NUM_GENES];
    for (size_t i = 0; i < NUM_GENES; i++) {
      this->Limit_Sup[i] = Lsup[i];
    }
    cout << "Valores de arreglos ya asigandos";
    Chromsize_();
    this->POB = new INDIVIDUO[this->ChromeSize];
    initialize_population();
    decode_integer();
    decode_real();
  }
  void print() {

    for (int k = 0; k < TAM_POBLACION; k++) {
      int Acumulado = ChromeSize - 1, g = NUM_GENES - 1;
      cout << "[" << k << "]";
      // Acumulado+=NumBITsxGEN[g];
      for (int i = ChromeSize - 1; i >= 0; i--) {
        if (i == Acumulado) {
          cout << ":";
          Acumulado -= NumBITsxGEN[g];
          g--;
        }
        cout << (int)POB[k].Chrom[i];
      }
      cout << " Valor_Entero:";
      for (int j = NUM_GENES - 1; j >= 0; j--)
        cout << "," << (int)POB[k].Vent[j];
      cout << " Valor_Real:";
      for (int j = NUM_GENES - 1; j >= 0; j--)
        cout << "," << (float)POB[k].Vre[j];
      cout << endl;
    }
  }
};

int main() {
  cout << "Hola Algoritmo Genetico" << endl;
  cout << "Generando la poblacion..." << endl;
  // Inicializar la Poblacion de Individuos
  // DECODIFICACION A ENTERO
  unsigned int bits[3] = {2, 2, 2};
  float LI[3] = {0, 0, 0};
  float LS[3] = {1, 1, 1};
  GA algoritmo(10, 3, bits, LI, LS);
  algoritmo.print();
  // MOSTRAR LA POBLACION

  cout << "Fin del programa";
  return 0;
}
