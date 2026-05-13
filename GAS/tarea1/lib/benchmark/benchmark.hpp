#pragma once

typedef float (*ObjectiveFunction)(float *, unsigned int);

float f1(float *x, unsigned int n);
float f5(float *x, unsigned int n);
float f13(float *x, unsigned int n);
float u(float x, float a, float b, float c);
