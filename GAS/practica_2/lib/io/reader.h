
#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
enum class ColumnType { Int, Double, String };

class DataFrame {
public:
  struct column {
    std::string name;
    std::vector<int> ints;
    ColumnType tipo;
    std::vector<double> doubles;
    std::vector<std::string> strings;
  };

  std::vector<column> columns;

private:
  size_t nrows = 0;

public:
  DataFrame() = default;
  DataFrame(const std::vector<std::string> &colnames);
  void addColNames(const std::vector<std::string> &colnames);
  size_t rows() const { return nrows; }
  size_t cols() const { return columns.size(); }
  double min(const column &col) const;
  double max(const column &col) const;
  double mean(const column &col) const;
  double std(const column &col) const;
  std::vector<double> cuartil(const column &col) const;
  void add_row(const std::vector<std::string> &row);
  DataFrame operator[](const std::string &name);
  const DataFrame operator[](const std::string &name) const;
  void print() const;
  void print_cols();
  void info();
  void info(const column &valor);
  std::vector<std::pair<std::string, int>> value_counts(const column &col);
  DataFrame drop(std::string label);
  template <typename T>
  std::vector<T> &get_column_values(const std::string &name) {
    for (auto &col : columns) {
      if (col.name == name) {
        if constexpr (std::is_same_v<T, int>)
          return col.ints;
        else if constexpr (std::is_same_v<T, double>)
          return col.doubles;
        else if constexpr (std::is_same_v<T, std::string>)
          return col.strings;
      }
    }
    throw std::runtime_error("Column type mismatch");
  }

  void to_csv(const std::string &out) const;
  void add_column(const std::string &name, const std::vector<double> &values);
};

DataFrame read_csv(const std::string &file_path, char separation = ',');
std::vector<std::string> split_fast(std::string line, char separation);
std::string rtrim(std::string &s);
