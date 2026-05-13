/*Simple Genetic Algorithm
  SGA.cpp
  Implementacion del algoritmo
  genetico simple
 */
#include <cmath>
#include <iostream>

//***** Parametros de configuracion del AG ****
const int NUM_GENES = 3;
const int NumBITsxGEN[NUM_GENES] = {2, 8, 3};
const int TAM_POBLACION = 5;
typedef unsigned char BYTE;
typedef struct {
  BYTE *Chrom;
  unsigned int *Vent;
} INDIVIDUO;
using namespace std;

int main() {
  INDIVIDUO POB[TAM_POBLACION];
  int ChromeSize = 0;
  cout << "Hola Algoritmo Genetico" << endl;
  cout << "Generando la poblacion..." << endl;

  // Calcular el Tamaño del Cromosoma
  for (int k = 0; k < NUM_GENES; k++)
    ChromeSize += NumBITsxGEN[k];

  // Inicializar la Poblacion de Individuos
  for (int k = 0; k < TAM_POBLACION;
       k++) { // Reservar memoria para cada cadena Binaria de CROMOSOMA
    POB[k].Chrom = new BYTE[ChromeSize];
    POB[k].Vent = new unsigned int[NUM_GENES];
    // Inicializar el cromosoma
    for (int i = 0; i < ChromeSize; i++)
      POB[k].Chrom[i] = rand() % 2;
    for (int j = 0; j < NUM_GENES; j++)
      POB[k].Vent[j] = 0;
  }

  // DECODIFICACION A ENTERO
  for (int k = 0; k < TAM_POBLACION; k++) {
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

  // MOSTRAR LA POBLACION
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
    cout << endl;
  }

  cout << "Fin del programa";
  return 0;
}
