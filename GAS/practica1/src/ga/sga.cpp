/* CÓDIGO FUENTE
   ALGORITMO GENETICO SIMPLE
   sga.cpp
   Implementación del algoritmo genético simple
   Dr. Carlos García
   Cómputo Evolutivo
   LIDIA
 */

#include "ga/sga.hpp"
#include <cstdlib>
#include <vector>
algoritmogeneticsimple::algoritmogeneticsimple(
    COD_TYPE TC, unsigned int _tamaño_poblacion, unsigned int _num_genes,
    const vector<unsigned int> &_bits_por_gen,
    const vector<float> &_limites_superiores,
    const vector<float> &_limites_inferiores, const float &probabilidad_cruza,
    const float &probabilidad_mutacion, const unsigned int &max_gen) {
  this->tamaño_poblacion = _tamaño_poblacion;
  this->num_genes = _num_genes;
  this->bits_por_gen = _bits_por_gen;
  this->limites_superiores = _limites_superiores;
  this->limites_inferiores = _limites_inferiores;
  this->probabilidad_cruza = probabilidad_cruza;
  this->probabilidad_mutacion = probabilidad_mutacion;
  this->tamaño_cromosoma = 0;
  this->max_gen = max_gen;
  // Calcular el tamaño del cromosoma
  if (TC == BIN)
    for (unsigned int k = 0; k < this->num_genes; k++)
      this->tamaño_cromosoma += this->bits_por_gen[k];
  else
    this->tamaño_cromosoma = _num_genes;
  // Inicializar la población de individuos
  this->poblacion = new Individuo[this->tamaño_poblacion];
  this->poblacion_nueva = new Individuo[this->tamaño_poblacion];
  this->seleccion = new unsigned int[this->tamaño_poblacion];
  if (TC == BIN)
    for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
      this->poblacion[k].cromosoma = NULL;
      //    this->poblacion[k].valores_enteros=NULL;
      this->poblacion[k].valores_reales.resize(this->num_genes);
      this->poblacion_nueva[k].cromosoma = NULL;
      //      this->poblacion_nueva[k].valores_enteros=NULL;
      this->poblacion_nueva[k].valores_reales.resize(this->num_genes);

      // Inicializar el cromosoma aleatoriamente
      for (unsigned int i = 0; i < this->tamaño_cromosoma; i++)
        this->poblacion[k].cromosoma[i] = rand() % 2;

      for (unsigned int j = 0; j < this->num_genes; j++) {
        this->poblacion[k].valores_enteros[j] = 0;
        this->poblacion[k].valores_reales[j] = 0;
      }
    }
  else
    for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
      this->poblacion[k].cromosoma = new BYTE[this->tamaño_cromosoma];
      this->poblacion[k].valores_enteros.resize(this->num_genes);
      this->poblacion[k].valores_reales.resize(this->num_genes);
      this->poblacion_nueva[k].cromosoma = new BYTE[this->tamaño_cromosoma];
      this->poblacion_nueva[k].valores_enteros.resize(this->num_genes);
      this->poblacion_nueva[k].valores_reales.resize(this->num_genes);

      // Inicializar el cromosoma aleatoriamente
      for (unsigned int i = 0; i < this->tamaño_cromosoma; i++) {
        float rango = limites_superiores[i] - limites_inferiores[i];
        this->poblacion[k].valores_reales[i] =
            limites_inferiores[i] + rango * ((double)rand() / RAND_MAX);
      }

      for (unsigned int j = 0; j < this->num_genes; j++) {
        this->poblacion[k].valores_enteros[j] = 0;
        this->poblacion[k].valores_reales[j] = 0;
      }
    }
}

algoritmogeneticsimple::algoritmogeneticsimple(
    unsigned int _tamaño_poblacion, unsigned int _num_genes,
    const vector<unsigned int> &_bits_por_gen,
    const vector<float> &_limites_superiores,
    const vector<float> &_limites_inferiores, const float &probabilidad_cruza,
    const float &probabilidad_mutacion, const unsigned int &max_gen) {
  this->tamaño_poblacion = _tamaño_poblacion;
  this->num_genes = _num_genes;
  this->bits_por_gen = _bits_por_gen;
  this->limites_superiores = _limites_superiores;
  this->limites_inferiores = _limites_inferiores;
  this->probabilidad_cruza = probabilidad_cruza;
  this->probabilidad_mutacion = probabilidad_mutacion;
  this->tamaño_cromosoma = 0;
  this->max_gen = max_gen;
  // Calcular el tamaño del cromosoma
  for (unsigned int k = 0; k < this->num_genes; k++)
    this->tamaño_cromosoma += this->bits_por_gen[k];

  // Inicializar la población de individuos
  this->poblacion = new Individuo[this->tamaño_poblacion];
  this->poblacion_nueva = new Individuo[this->tamaño_poblacion];
  this->seleccion = new unsigned int[this->tamaño_poblacion];

  for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
    this->poblacion[k].cromosoma = new BYTE[this->tamaño_cromosoma];
    this->poblacion[k].valores_enteros.resize(this->num_genes);
    this->poblacion[k].valores_reales.resize(this->num_genes);
    this->poblacion_nueva[k].cromosoma = new BYTE[this->tamaño_cromosoma];
    this->poblacion_nueva[k].valores_enteros.resize(this->num_genes);
    this->poblacion_nueva[k].valores_reales.resize(this->num_genes);

    // Inicializar el cromosoma aleatoriamente
    for (unsigned int i = 0; i < this->tamaño_cromosoma; i++)
      this->poblacion[k].cromosoma[i] = rand() % 2;

    for (unsigned int j = 0; j < this->num_genes; j++) {
      this->poblacion[k].valores_enteros[j] = 0;
      this->poblacion[k].valores_reales[j] = 0;
    }
  }
}

algoritmogeneticsimple::~algoritmogeneticsimple() {
  for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
    delete[] this->poblacion[k].cromosoma;
    delete[] this->poblacion_nueva[k].cromosoma;
  }
  delete[] this->poblacion;
  delete[] this->poblacion_nueva;
  delete[] this->seleccion;
}

void algoritmogeneticsimple::DecodificarEnteros(void) {
  for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
    unsigned int indice_gen = 0;
    unsigned int acumulado = this->bits_por_gen[0];
    unsigned int auxiliar = 0;

    for (unsigned int i = 0, j = 0; i < this->tamaño_cromosoma; i++, j++) {
      auxiliar += this->poblacion[k].cromosoma[i] * (unsigned int)pow(2, j);

      if (i == (acumulado - 1)) {
        this->poblacion[k].valores_enteros[indice_gen] = auxiliar;
        auxiliar = 0;
        indice_gen++;
        if (indice_gen < this->num_genes)
          acumulado += this->bits_por_gen[indice_gen];
        j = -1; // se incrementa a 0 en el for
      }
    }
  }
}

void algoritmogeneticsimple::DecodificarReales(void) {
  for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
    for (unsigned int g = 0; g < this->num_genes; g++) {
      float maxi = this->limites_superiores[g];
      float mini = this->limites_inferiores[g];
      float rango = maxi - mini;
      float denominador = (float)((1 << this->bits_por_gen[g]) - 1.0f);
      float normalizado =
          (float)this->poblacion[k].valores_enteros[g] / denominador;

      this->poblacion[k].valores_reales[g] = mini + normalizado * rango;
    }
  }
}

void algoritmogeneticsimple::ImprimirIndividuo(unsigned int _id_individuo) {
  int acumulado = this->tamaño_cromosoma - 1;
  int indice_gen = this->num_genes - 1;
  cout << "[" << _id_individuo << "]";

  for (int i = this->tamaño_cromosoma - 1; i >= 0; i--) {
    if (i == acumulado) {
      cout << ":";
      acumulado -= this->bits_por_gen[indice_gen];
      indice_gen--;
    }
    cout << (int)this->poblacion[_id_individuo].cromosoma[i];
  }

  cout << " :";
  for (int j = this->num_genes - 1; j >= 0; j--)
    cout << this->poblacion[_id_individuo].valores_enteros[j] << ",";

  cout << " :";
  for (int j = this->num_genes - 1; j >= 0; j--)
    cout << this->poblacion[_id_individuo].valores_reales[j] << ",";

  cout << "Objetivo: " << this->poblacion[_id_individuo].valor_objetivo;
  cout << " Aptitud: " << this->poblacion[_id_individuo].valor_aptitud;
  cout << endl;
}

void algoritmogeneticsimple::ImprimirPoblacion(void) {
  for (unsigned int k = 0; k < this->tamaño_poblacion; k++)
    this->ImprimirIndividuo(k);

  cout << "ID Mejor Objetivo: " << this->id_mejor_objetivo << endl;
  cout << "ID Peor Objetivo: " << this->id_peor_objetivo << endl;
  cout << "Suma Objetivos: " << this->suma_objetivos << endl;
  cout << "Promedio Objetivos: " << this->promedio_objetivos << endl;
  cout << "Suma Aptitud: " << this->suma_aptitud << endl;
  cout << "Promedio Aptitud: " << this->promedio_aptitud << endl;
}

void algoritmogeneticsimple::EvaluarPoblacion(void) {
  this->id_mejor_objetivo = 0;
  this->id_peor_objetivo = 0;
  this->suma_objetivos = 0;

  for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
    this->poblacion[k].valor_objetivo = this->FuncionObjetivo(k);

    // mejor = mayor valor_objetivo (antes de conversión de aptitud)
    if (this->poblacion[k].valor_objetivo >
        this->poblacion[this->id_mejor_objetivo].valor_objetivo)
      this->id_mejor_objetivo = k;

    if (this->poblacion[k].valor_objetivo <
        this->poblacion[this->id_peor_objetivo].valor_objetivo)
      this->id_peor_objetivo = k;

    this->suma_objetivos += this->poblacion[k].valor_objetivo;
  }

  this->promedio_objetivos = this->suma_objetivos / this->tamaño_poblacion;
}

void algoritmogeneticsimple::ConvertirObjetivoAptitud(
    TipoOptimizacion _tipo_optimizacion) {
  unsigned int k;
  float rango;
  this->suma_aptitud = 0;

  if (_tipo_optimizacion == MAXIMIZAR) {

    rango = this->poblacion[this->id_mejor_objetivo].valor_objetivo -
            this->poblacion[this->id_peor_objetivo].valor_objetivo;

    // guard contra división por cero (todos iguales)
    if (rango == 0.0f) {
      for (k = 0; k < this->tamaño_poblacion; k++) {
        this->poblacion[k].valor_aptitud = 100.0f;
        this->suma_aptitud += 100.0f;
      }
      this->promedio_aptitud = 100.0f;
      return;
    }

    for (k = 0; k < this->tamaño_poblacion; k++) {
      this->poblacion[k].valor_aptitud =
          100.0f * ((this->poblacion[k].valor_objetivo -
                     this->poblacion[this->id_peor_objetivo].valor_objetivo) /
                    rango);
      this->suma_aptitud += this->poblacion[k].valor_aptitud;
    }
    this->promedio_aptitud = this->suma_aptitud / this->tamaño_poblacion;

  } else { // MINIMIZAR

    for (k = 0; k < this->tamaño_poblacion; k++)
      this->poblacion[k].valor_aptitud =
          this->poblacion[this->id_mejor_objetivo].valor_objetivo -
          this->poblacion[k].valor_objetivo;

    unsigned int auxiliar = this->id_mejor_objetivo;
    this->id_mejor_objetivo = this->id_peor_objetivo;
    this->id_peor_objetivo = auxiliar;

    // Normalizar aptitud a [0, 100]
    float minimo = this->poblacion[this->id_peor_objetivo].valor_aptitud;
    rango = this->poblacion[this->id_mejor_objetivo].valor_aptitud - minimo;

    // guard contra división por cero
    if (rango == 0.0f) {
      for (k = 0; k < this->tamaño_poblacion; k++) {
        this->poblacion[k].valor_aptitud = 100.0f;
        this->suma_aptitud += 100.0f;
      }
      this->promedio_aptitud = 100.0f;
      return;
    }

    for (k = 0; k < this->tamaño_poblacion; k++) {
      this->poblacion[k].valor_aptitud =
          100.0f * ((this->poblacion[k].valor_aptitud - minimo) / rango);
      this->suma_aptitud += this->poblacion[k].valor_aptitud;
    }
    this->promedio_aptitud = this->suma_aptitud / this->tamaño_poblacion;
  }
}

void algoritmogeneticsimple::SeleccionRuleta(void) {
  unsigned int k, seleccionado;
  float suma_acumulada = 0, bolita_aleatoria;
  vector<float> probabilidad_acumulada(this->tamaño_poblacion);

  for (k = 0; k < this->tamaño_poblacion; k++) {
    suma_acumulada += this->poblacion[k].valor_aptitud / this->suma_aptitud;
    probabilidad_acumulada[k] = suma_acumulada;
  }

  for (k = 0; k < this->tamaño_poblacion; k++) {
    bolita_aleatoria = (float)rand() / RAND_MAX;
    seleccionado = 0;

    while (seleccionado < this->tamaño_poblacion - 1 &&
           probabilidad_acumulada[seleccionado] < bolita_aleatoria)
      seleccionado++;

    this->seleccion[k] = seleccionado;
  }
}

void algoritmogeneticsimple::CruzamientoPuntoUnico(double _probabilidad_cruza) {
  unsigned int i, k, punto_cruza, padre_1, padre_2;
  unsigned int limite = this->tamaño_cromosoma - 1;
  double valor_aleatorio;

  for (k = 0; k < this->tamaño_poblacion; k += 2) {
    valor_aleatorio = (double)rand() / RAND_MAX;

    if (valor_aleatorio < _probabilidad_cruza) {
      padre_1 = this->seleccion[k];
      padre_2 = this->seleccion[k + 1];
      this->poblacion_nueva[k].padre_1 = padre_1;
      this->poblacion_nueva[k].padre_2 = padre_2;
      this->poblacion_nueva[k + 1].padre_1 = padre_2;
      this->poblacion_nueva[k + 1].padre_2 = padre_1;
      punto_cruza = rand() % limite;

      for (i = 0; i <= punto_cruza; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
      }
      for (i = punto_cruza + 1; i < this->tamaño_cromosoma; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
      }
    } else {
      // Sin cruzamiento — copiar padres tal cual
      padre_1 = this->seleccion[k];
      padre_2 = this->seleccion[k + 1];
      this->poblacion_nueva[k].padre_1 = padre_1;
      this->poblacion_nueva[k].padre_2 = padre_1;
      this->poblacion_nueva[k + 1].padre_1 = padre_2;
      this->poblacion_nueva[k + 1].padre_2 = padre_2;

      for (i = 0; i < this->tamaño_cromosoma; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
      }
    }
  }
}

void algoritmogeneticsimple::Mutar(double _probabilidad_mutacion) {
  double valor_aleatorio;

  for (unsigned int k = 0; k < this->tamaño_poblacion; k++) {
    for (unsigned int i = 0; i < this->tamaño_cromosoma; i++) {
      valor_aleatorio = (double)rand() / RAND_MAX;
      if (valor_aleatorio < _probabilidad_mutacion)
        this->poblacion_nueva[k].cromosoma[i] =
            1 - this->poblacion_nueva[k].cromosoma[i];
    }
  }
}

void algoritmogeneticsimple::GenerarProximaGeneracion(void) {
  Individuo *auxiliar = this->poblacion;
  this->poblacion = this->poblacion_nueva;
  this->poblacion_nueva = auxiliar;
}

void algoritmogeneticsimple::Elitismo(void) {
  // el elitismo debe aplicarse ANTES del cruzamiento/mutación,
  // es decir aquí preservamos el mejor al slot 0 de poblacion_nueva.
  for (unsigned int i = 0; i < this->tamaño_cromosoma; i++)
    this->poblacion_nueva[0].cromosoma[i] =
        this->poblacion[this->id_mejor_objetivo].cromosoma[i];
}

unsigned int algoritmogeneticsimple::ObtenerMejor(void) {
  return this->id_mejor_objetivo;
}

float algoritmogeneticsimple::ObtenerMejorObjetivo(void) {
  // después de ConvertirObjetivoAptitud(MINIMIZAR),
  // id_mejor_objetivo ya apunta al individuo con MENOR valor_objetivo.
  return this->poblacion[this->id_mejor_objetivo].valor_objetivo;
}

float algoritmogeneticsimple::FuncionObjetivo(unsigned int _id_individuo) {
  return this->funcion_objetivo(this->poblacion[_id_individuo].valores_reales);
  // Error de tipado y aqui se refleja, la funcion lambda espera const
  // std::vector<float>& Mientras que esta recvibiendo un float*, tipado
  // incompatible
}

void algoritmogeneticsimple::fit(
    std::function<float(const std::vector<float> &)> funcion,
    const metodo &method, TipoOptimizacion _tipo_optimizacion) {
  this->funcion_objetivo = funcion;
  this->DecodificarEnteros();
  this->DecodificarReales();
  this->EvaluarPoblacion();
  this->ImprimirPoblacion();
  this->ConvertirObjetivoAptitud(_tipo_optimizacion);
  for (unsigned int gen = 1; gen <= this->max_gen; gen++) {
    if (method.seleccion == "ruleta")
      this->SeleccionRuleta();
    if (method.seleccion == "torneo")
      this->torneo();
    if (method.cruza == "unico")
      this->CruzamientoPuntoUnico(this->probabilidad_cruza);
    if (method.cruza == "doble")
      this->CruzamientoDoblePunto(this->probabilidad_cruza);
    this->Mutar(this->probabilidad_mutacion);
    this->Elitismo();
    this->GenerarProximaGeneracion();
    this->DecodificarEnteros();
    this->DecodificarReales();
    this->EvaluarPoblacion();
    this->ConvertirObjetivoAptitud(_tipo_optimizacion);
  }
  cout << this->ObtenerMejorObjetivo() << "\n";
}
void algoritmogeneticsimple::CruzamientoDoblePunto(double _probabilidad_cruza) {
  unsigned int i, k, punto_cruza_1, punto_cruza_2, padre_1, padre_2;
  unsigned int limite = this->tamaño_cromosoma - 2;
  double valor_aleatorio;

  for (k = 0; k < this->tamaño_poblacion; k += 2) {
    valor_aleatorio = (double)rand() / RAND_MAX;

    if (valor_aleatorio < _probabilidad_cruza) {
      padre_1 = this->seleccion[k];
      padre_2 = this->seleccion[k + 1];
      this->poblacion_nueva[k].padre_1 = padre_1;
      this->poblacion_nueva[k].padre_2 = padre_2;
      this->poblacion_nueva[k + 1].padre_1 = padre_2;
      this->poblacion_nueva[k + 1].padre_2 = padre_1;
      punto_cruza_1 = rand() % limite;
      punto_cruza_2 = (punto_cruza_1 + 2) +
                      rand() % ((this->tamaño_cromosoma) - (punto_cruza_1 + 2));

      for (i = 0; i <= punto_cruza_1; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
      }
      for (i = punto_cruza_1 + 1; i < punto_cruza_2; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
      }
      for (i = punto_cruza_2; i < this->tamaño_cromosoma; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
      }
    } else {
      // Sin cruzamiento — copiar padres tal cual
      padre_1 = this->seleccion[k];
      padre_2 = this->seleccion[k + 1];
      this->poblacion_nueva[k].padre_1 = padre_1;
      this->poblacion_nueva[k].padre_2 = padre_1;
      this->poblacion_nueva[k + 1].padre_1 = padre_2;
      this->poblacion_nueva[k + 1].padre_2 = padre_2;

      for (i = 0; i < this->tamaño_cromosoma; i++) {
        this->poblacion_nueva[k].cromosoma[i] =
            this->poblacion[padre_1].cromosoma[i];
        this->poblacion_nueva[k + 1].cromosoma[i] =
            this->poblacion[padre_2].cromosoma[i];
      }
    }
  }
}
void algoritmogeneticsimple::torneo() {
  unsigned int k, player1, player2, sel;
  for (int k = 0; k < this->tamaño_poblacion; k++) {
    player1 = (unsigned int)rand() % (this->tamaño_poblacion);
    player2 = (unsigned int)rand() % (this->tamaño_poblacion);
    if (this->poblacion[player1].valor_aptitud <
        this->poblacion[player2].valor_aptitud)
      sel = player1;
    else
      sel = player2;
    this->seleccion[k] = sel;
  }
}
