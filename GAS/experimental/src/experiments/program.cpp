#include "experiments/program.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
Programa Asignacion(int argc, char **argv) {
  Programa config;
  config.NUM_GENES = 10;
  config.TAMAÑO_POBLACION = 100;
  config.MAX_GENERACIONES = 500;
  config.PROB_CRUZA = 0.9;
  config.PROB_MUTACION = 0.01;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    std::cout << "ARG: [" << arg << "]\n";
    if (arg == "-variables" && i + 1 < argc) {
      config.NUM_GENES = atoi(argv[++i]);
      config.BITS_POR_GEN.clear();
      config.LIMITE_SUPERIOR.clear();
      config.LIMITE_INFERIOR.clear();
      config.LIMITE_SUPERIOR_BASE.clear();
      config.LIMITE_INFERIOR_BASE.clear();
    } else if (arg == "-bits" && i + 1 < argc) {
      config.BITS_POR_GEN.clear();
      i++;
      while (i < argc) {
        string next = argv[i];
        if (next[0] == '-' && !isdigit(next[1]))
          break;
        config.BITS_POR_GEN.push_back(atoi(argv[i++]));
      }
      i--;
    } else if (arg == "-poblacion" && i + 1 < argc) {
      config.TAMAÑO_POBLACION = atoi(argv[++i]);
    } else if (arg == "-generaciones" && i + 1 < argc) {
      config.MAX_GENERACIONES = atoi(argv[++i]);
    } else if (arg == "-cruza" && i + 1 < argc) {
      config.PROB_CRUZA = atof(argv[++i]);
    } else if (arg == "-mutacion" && i + 1 < argc) {
      config.PROB_MUTACION = atof(argv[++i]);
    } else if (arg == "-sup" && i + 1 < argc) {
      config.LIMITE_SUPERIOR_BASE.clear();
      i++;
      while (i < argc) {
        string next = argv[i];
        if (next[0] == '-' && !isdigit(next[1]))
          break;
        config.LIMITE_SUPERIOR_BASE.push_back(atof(argv[i++]));
      }
      i--;
    } else if (arg == "-inf" && i + 1 < argc) {
      config.LIMITE_INFERIOR_BASE.clear();
      i++;
      while (i < argc) {
        string next = argv[i];
        if (next[0] == '-' && !isdigit(next[1]))
          break;
        config.LIMITE_INFERIOR_BASE.push_back(atof(argv[i++]));
      }
      i--;
    }
  }

  // Completar vectores repitiendo el último valor ingresado
  if (config.BITS_POR_GEN.size() < config.NUM_GENES) {
    unsigned int ultimo =
        config.BITS_POR_GEN.empty() ? 16u : config.BITS_POR_GEN.back();
    while (config.BITS_POR_GEN.size() < config.NUM_GENES)
      config.BITS_POR_GEN.push_back(ultimo);
  }

  if (config.LIMITE_SUPERIOR_BASE.size() == 0) {
    float sup_default = 1.0f;
    float inf_default = 0.0f;
    config.LIMITE_SUPERIOR.assign(config.NUM_GENES, sup_default);
    config.LIMITE_INFERIOR.assign(config.NUM_GENES, inf_default);
    return config;
  }
  config.LIMITE_INFERIOR.resize(config.NUM_GENES);
  config.LIMITE_SUPERIOR.resize(config.NUM_GENES);
  size_t base = config.LIMITE_SUPERIOR_BASE.size();
  for (size_t t = 0; t < config.NUM_GENES; t++) {
    config.LIMITE_INFERIOR[t] = config.LIMITE_INFERIOR_BASE[t % base];
    config.LIMITE_SUPERIOR[t] = config.LIMITE_SUPERIOR_BASE[t % base];
  }
  return config;
}
