#include <math.h>
#include "utils.h"

float max_abs_diff(const float* a, const float* b, int n) {
  float m = 0.f;
  for (int i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}
