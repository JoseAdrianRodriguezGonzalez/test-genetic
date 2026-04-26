#include "../../lib/benchmark/benchmark.hpp"
#include <cmath>
float f1(float *x, unsigned int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sum;
}
float f5(float *x, unsigned int n) {
  if (n < 2)
    return 0;
  float sum = 0;
  for (int i = 0; i < n - 1; i++) {
    float xi = x[i];
    float xi1 = x[i + 1];
    float term1 = xi1 - xi * xi;
    float term2 = xi - 1;
    sum += 100 * (term1 * term1) + term2 * term2;
  }
  return sum;
}
float u(float x, float a, float b, float c) {
  if (x > a) {
    return b * std::pow(x - a, c);
  }
  if (x < -a) {
    return b * std::pow(-x - a, c);
  }
  return 0.0;
}
float f13(float *x, unsigned int n) {

  float sum1 = 0;
  float sum2 = 0;
  const double PI = 3.141592653589793;
  for (int i = 0; i < n - 1; i++) {
    float xi = x[i];
    float xi1 = x[i + 1];
    float term1 = xi - 1;
    float term2 = std::sin(3 * PI * xi1);
    sum1 += (term1 * term1) * (1 + term2 * term2);
  }
  for (int i = 0; i < n; i++) {
    sum2 += u(x[i], 5, 100, 4);
  }

  return 0.1 * (std::sin(3 * PI * x[0]) + sum1 +
                std::pow((x[n - 1] - 1), 2) *
                    (1 + std::pow(2 * PI * x[n - 1], 2))) +
         sum2;
}
