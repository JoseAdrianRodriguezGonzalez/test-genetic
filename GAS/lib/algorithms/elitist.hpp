#include <float.h>

typedef unsigned char BYTE;
typedef enum { MAX, MIN } OPT_TYPE;
typedef float (*ObjectiveFunction)(float *, unsigned int);
typedef struct {
  BYTE *Chrom;
  unsigned int *Vent; // valores enteros decodificados
  float *Vre;         // valores reales decodificados
  float VObj;
  float Vfit;
  unsigned int Padre1;
  unsigned int Padre2;

} INDIVIDUO;
const unsigned int MAX_GENERCIONES = 300;
class GA {
private:
  INDIVIDUO *POB;
  INDIVIDUO *NewPOB;
  unsigned int ChromeSize;
  unsigned int NUM_GENES;
  unsigned int *NumBITsxGEN;
  unsigned int TAM_POBLACION;
  unsigned int *seleccionados;
  float *Limit_Sup;
  float *Limit_Inf;
  int idMax;
  int idMin;
  float SumObj;
  float PromObj;
  float SumFit;
  float PromFit;
  ObjectiveFunction f;

public:
  void Chromsize_();
  void initialize_population();
  void decode_integer();
  void decode_real();
  GA();
  GA(unsigned int TAM_POBLACION, unsigned int NUM_GENES,
     unsigned int *NumBITsxGEN, float *Linf, float *Lsup, ObjectiveFunction f);
  void initialization_ga(unsigned int TAM_POBLACION, unsigned int NUM_GENES,
                         unsigned int *NumBITsxGEN, float *Linf, float *Lsup,
                         ObjectiveFunction f);

  void initialization_ga(unsigned int TAM_POBLACION, unsigned int NUM_GENES,
                         unsigned int *NumBITsxGEN, float *Linf, float *Lsup);
  ~GA();
  void print_population();
  //  float objectiveFunction(unsigned int id);
  void EvaluatePoblation();
  void obj_to_fit(OPT_TYPE tipo);
  void Roulette();
  void cruza1P(float C);
  void print_new();
  void muta(float prob_muta);
  void NextGeneration(void);
  void print_individual(unsigned int k);
  unsigned int getMax();
  void Elitismo();
  void run(const unsigned int &gen, OPT_TYPE opt, const float &prob_mut,
           const float &prob_cross);
  void free_memory();
  void resize_population(unsigned int new_size);
  float getObjetive(unsigned int index);
  float getfitness(unsigned int index);
  float *getRealValues(unsigned int index);
  void printRealValues(unsigned int index);
};
