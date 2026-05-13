/* ARCHIVO DE CABECERA
   ALGORITMO GENETICO SIMPLE
   sga.hpp
   Archivo de Cabecera del algoritmo genetico simple
   Dr. Carlos García
   Cómputo Evolutivo
   LIDIA
 */

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
using namespace std;

typedef unsigned char BYTE;
typedef enum { MAXIMIZAR, MINIMIZAR } TipoOptimizacion;
typedef enum { BIN, REAL } COD_TYPE;
typedef struct {
  BYTE *cromosoma; // codificacion binaria
  vector<unsigned int> valores_enteros;
  vector<float> valores_reales; // codificacion real
  float valor_objetivo;
  float valor_aptitud;
  unsigned int padre_1;
  unsigned int padre_2;
} Individuo;
typedef struct {
  std::string cruza;
  std::string seleccion;
} metodo;
class algoritmogeneticsimple {
private:
  Individuo *poblacion;
  Individuo *poblacion_nueva;
  unsigned int *seleccion;
  unsigned int tamaño_cromosoma;
  unsigned int num_genes;
  vector<unsigned int> bits_por_gen;
  unsigned int tamaño_poblacion;
  vector<float> limites_superiores;
  vector<float> limites_inferiores;
  unsigned int id_mejor_objetivo;
  unsigned int id_peor_objetivo;
  float suma_objetivos;
  float promedio_objetivos;
  float suma_aptitud;
  float promedio_aptitud;
  float probabilidad_mutacion;
  float probabilidad_cruza;
  unsigned int max_gen;
  vector<float> X;
  vector<float> Y;
  std::function<float(const std::vector<float> &)> funcion_objetivo;

public:
  algoritmogeneticsimple(unsigned int _tamaño_poblacion,
                         unsigned int _num_genes,
                         const vector<unsigned int> &_bits_por_gen,
                         const vector<float> &_limites_superiores,
                         const vector<float> &_limites_inferiores,
                         const float &probabilidad_cruza,
                         const float &probabilidad_mutacion,
                         const unsigned int &max_gen);

  algoritmogeneticsimple(COD_TYPE TC, unsigned int _tamaño_poblacion,
                         unsigned int _num_genes,
                         const vector<unsigned int> &_bits_por_gen,
                         const vector<float> &_limites_superiores,
                         const vector<float> &_limites_inferiores,
                         const float &probabilidad_cruza,
                         const float &probabilidad_mutacion,
                         const unsigned int &max_gen);

  ~algoritmogeneticsimple();

  void DecodificarEnteros(void);
  void DecodificarReales(void);
  void ImprimirIndividuo(unsigned int _id_individuo);
  void ImprimirPoblacion(void);
  void EvaluarPoblacion(void);
  float FuncionObjetivo(unsigned int _id_individuo);
  void ConvertirObjetivoAptitud(TipoOptimizacion _tipo_optimizacion);
  void SeleccionRuleta(void);
  void CruzamientoPuntoUnico(double _probabilidad_cruza);
  void CruzamientoDoblePunto(double _probabilidad_cruza);
  void Mutar(double _probabilidad_mutacion);
  void GenerarProximaGeneracion(void);
  void Elitismo(void);
  unsigned int ObtenerMejor(void);
  float ObtenerMejorObjetivo(void);
  void fit(std::function<float(const std::vector<float> &)> funcion,
           const metodo &method,
           TipoOptimizacion _tipo_optimizacion = MINIMIZAR);
  void torneo();
}; // FIN DE LA CLASE
