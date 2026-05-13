#include "io/reader.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
DataFrame::DataFrame(const std::vector<std::string> &colnames) {
  addColNames(colnames);
}
void DataFrame::addColNames(const std::vector<std::string> &colnames) {
  for (const auto &name : colnames) {
    column col;
    col.name = name;
    col.tipo = ColumnType::Int; // provisional (se ajusta al leer)
    columns.push_back(col);
  }
}

// Devuelve una referencia al vector contiguo de la columna

void DataFrame::add_row(const std::vector<std::string> &row) {
  if (row.size() != this->columns.size())
    throw std::runtime_error("Row has wrong number of columns.");
  for (size_t i = 0; i < row.size(); i++) {
    auto &col = columns[i];
    const std::string val = row[i];

    try {
      int v = std::stoi(val);
      if (col.tipo == ColumnType::Int) {
        col.ints.push_back(v);
      } else if (col.tipo == ColumnType::Double) {
        col.doubles.push_back(double(v));
      } else {
        throw std::runtime_error("Can't store number into string column.");
      }
      continue;
    } catch (...) {
    }
    try {
      double v = std::stod(val);
      if (col.tipo == ColumnType::Int) {
        col.ints.push_back(v);
      } else if (col.tipo == ColumnType::Double) {
        col.doubles.push_back(double(v));
      } else {
        throw std::runtime_error("Can't store number into string column.");
      }
      continue;
    } catch (...) {
    }
    if (col.tipo != ColumnType::String) {
      col.tipo = ColumnType::String;
      col.strings.reserve(nrows + 1);
      for (int v : col.ints)
        col.strings.push_back(std::to_string(v));
      for (double v : col.doubles)
        col.strings.push_back(std::to_string(v));
      col.ints.clear();
      col.doubles.clear();
    }
    col.strings.push_back(val);
  }
  nrows++;
}
void DataFrame::print_cols() {
  std::cout << "Columnas: \n";
  for (const auto &columna : this->columns)
    std::cout << columna.name << std::endl;
}
DataFrame DataFrame::operator[](const std::string &name) {
  DataFrame new_df;
  for (auto &cols : this->columns) {
    if (cols.name == name) {
      new_df.columns.push_back(cols);
      new_df.nrows = this->nrows;
      return new_df;
    }
  }
  throw std::runtime_error("Column not found: " + name);
}
const DataFrame DataFrame::operator[](const std::string &name) const {
  DataFrame new_df;
  for (auto &cols : this->columns) {
    if (cols.name == name) {
      new_df.columns.push_back(cols);
      new_df.nrows = this->nrows;
      return new_df;
    }
  }
  throw std::runtime_error("Column not found: " + name);
}
DataFrame DataFrame::drop(std::string label) {
  DataFrame df_new;
  for (const auto &col : this->columns) {
    if (col.name != label)
      df_new.columns.push_back(col);
  }
  df_new.nrows = this->nrows;
  return df_new;
}
double DataFrame::min(const column &valor) const {
  if (valor.tipo == ColumnType::Int) {
    return *std::min_element(valor.ints.begin(), valor.ints.end());
  }
  if (valor.tipo == ColumnType::Double) {
    return *std::min_element(valor.doubles.begin(), valor.doubles.end());
  }
  throw std::runtime_error("min() not supportd for string column");
}
double DataFrame::max(const column &valor) const {
  if (valor.tipo == ColumnType::Int) {
    return *std::max_element(valor.ints.begin(), valor.ints.end());
  }
  if (valor.tipo == ColumnType::Double) {
    return *std::max_element(valor.doubles.begin(), valor.doubles.end());
  }
  throw std::runtime_error("max() not supportd for string column");
}
double DataFrame::mean(const column &valor) const {
  if (valor.tipo == ColumnType::Int) {
    double s = 0;
    for (int i : valor.ints)
      s += i;
    return s / valor.ints.size();
  }
  if (valor.tipo == ColumnType::Double) {
    double s = 0;
    for (double i : valor.doubles)
      s += i;
    return s / valor.doubles.size();
  }
  throw std::runtime_error("mean() not supportd for string column");
}
double DataFrame::std(const column &valor) const {
  double m = mean(valor);
  double s = 0;
  if (valor.tipo == ColumnType::Int) {
    for (int i : valor.ints)
      s += (i - m) * (i - m);
    return std::sqrt(s / valor.ints.size());
  }
  if (valor.tipo == ColumnType::Double) {
    for (double i : valor.doubles)
      s += (i - m) * (i - m);
    return std::sqrt(s / valor.doubles.size());
  }
  throw std::runtime_error("mean() not supportd for string column");
}
void DataFrame::info(const column &valor) {
  double minimo = this->min(valor);
  double maximo = this->max(valor);
  double promedio = this->mean(valor);
  double desviacion = this->std(valor);
  std::vector<double> cuartil = this->cuartil(valor);
  std::cout << valor.name << ": "
            << ((valor.tipo == ColumnType::Int) ? "integer" : "double")
            << std::endl;
  std::cout << "minimo: " << minimo << std::endl;
  std::cout << "maximo: " << maximo << std::endl;
  std::cout << "promedio: " << promedio << std::endl;
  std::cout << "desviacion: " << desviacion << std::endl;
  std::cout << "q1: " << cuartil[0] << std::endl;
  std::cout << "q2: " << cuartil[1] << std::endl;
  std::cout << "q3: " << cuartil[2] << std::endl;
}
std::vector<double> DataFrame::cuartil(const column &valor) const {
  std::vector<double> data;
  if (valor.tipo == ColumnType::Int)
    for (int v : valor.ints)
      data.push_back(v);
  else if (valor.tipo == ColumnType::Double)
    data = valor.doubles;
  else
    throw std::runtime_error("quartiles() not supported for string columns.");
  std::sort(data.begin(), data.end());
  size_t n = data.size();
  auto q = [&](double p) {
    double idx = p * (n - 1);
    size_t i = std::floor(idx);
    double frac = idx - i;
    return data[i] + frac * (data[i + 1] - data[i]);
  };
  return {q(0.25), q(0.5), q(0.75)};
}
void DataFrame::info() {
  for (const auto &names : this->columns) {
    if (names.tipo == ColumnType::String) {
      std::cout << names.name << " : string (" << names.strings.size() << ")\n";
      continue;
    }
    this->info(names);
  }
}
/*
const DataFrame::column &DataFrame::operator[](const std::string &name) const {
  for (const auto &col : columns) {
    if (col.name == name) {
      return col;
    }
  }
  throw std::out_of_range("Column does not exist: " + name);
}
*/
std::vector<std::pair<std::string, int>>
DataFrame::value_counts(const DataFrame::column &col) {
  std::unordered_map<std::string, int> freq;
  if (col.tipo == ColumnType::String)
    for (auto &v : col.strings)
      freq[v]++;
  if (col.tipo == ColumnType::Int)
    for (int v : col.ints)
      freq[std::to_string(v)]++;
  if (col.tipo == ColumnType::Double)
    for (double v : col.doubles)
      freq[std::to_string(v)]++;
  std::vector<std::pair<std::string, int>> result(freq.begin(), freq.end());
  std::sort(result.begin(), result.end(),
            [](auto &a, auto &b) { return a.second > b.second; });
  return result;
}
void DataFrame::to_csv(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("No se pudo abrir archivo");
  }
  for (size_t j = 0; j < columns.size(); j++) {
    file << columns[j].name;
    if (j < columns.size() - 1)
      file << ",";
  }
  file << "\n";
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < columns.size(); j++) {
      const auto &col = columns[j];
      if (col.tipo == ColumnType::Int)
        file << col.ints[i];
      else if (col.tipo == ColumnType::Double)
        file << col.doubles[i];
      else
        file << col.strings[i];

      if (j < columns.size() - 1)
        file << ",";
    }
    file << "\n";
  }
}

void DataFrame::add_column(const std::string &name,
                           const std::vector<double> &values) {
  if (nrows != 0 && values.size() != nrows) {
    throw std::runtime_error("Column size mismatch");
  }
  column col;
  col.name = name;
  col.tipo = ColumnType::Double;
  col.doubles = values;

  if (nrows == 0) {
    nrows = values.size();
  }

  columns.push_back(std::move(col));
}
DataFrame read_csv(const std::string &file_path, char separation) {
  std::fstream fs(file_path, std::fstream::in);
  if (!fs.is_open()) {
    throw std::runtime_error("Error al leer el archivo");
  }
  std::string line;
  std::getline(fs, line);

  auto cols = split_fast(line, separation);
  for (auto &c : cols)
    c = rtrim(c);
  DataFrame df(cols);
  size_t count = 0;
  std::streampos pos = fs.tellg();
  while (std::getline(fs, line))
    count++;

  // Regresar al inicio de datos
  fs.clear();
  fs.seekg(pos);

  for (auto &c : df.columns) {
    switch (c.tipo) {
    case ColumnType::Int:
      c.ints.reserve(count);
      break;
    case ColumnType::Double:
      c.doubles.reserve(count);
      break;
    case ColumnType::String:
      c.strings.reserve(count);
      break;
    }
  }
  while (std::getline(fs, line)) {
    std::vector<std::string> row_separated = split_fast(line, separation);
    for (auto &c : cols)
      c = rtrim(c);
    df.add_row(row_separated);
  }
  fs.close();
  return df;
}

std::string rtrim(std::string &s) {
  size_t end = s.find_last_not_of(" \t\r\n");
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
std::vector<std::string> split_fast(std::string line, char separation) {
  std::vector<std::string> out;
  size_t start = 0;
  size_t end = line.find(separation);
  while (end != std::string::npos) {
    out.emplace_back(line.substr(start, end - start));
    start = end + 1;
    end = line.find(separation, start);
  }
  out.emplace_back(line.substr(start));
  return out;
}
