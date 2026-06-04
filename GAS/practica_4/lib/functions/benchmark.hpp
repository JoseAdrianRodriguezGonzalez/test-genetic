#pragma once
#include <vector>
typedef float (*ObjectiveFunction)(float *, unsigned int);

float f1(const std::vector<float> x);
float f5(const std::vector<float> x);
float f13(const std::vector<float> x);
float u(float x, float a, float b, float c);
