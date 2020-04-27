#include "color.h"
#include <cmath>

namespace color {
void mat_33_33_mult(float mat1[3][3], const float mat2[3][3], float out[3][3]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      out[i][j] = 0;
      for (int k = 0; k < 3; ++k)
        out[i][j] += mat1[i][k] * mat2[k][j];
    }
}

void mat_33_31_mult(const float mat[3][3], const double vec[3], float out[3]) {
  for (int i = 0; i < 3; ++i) {
    out[i] = 0;
    for (int j = 0; j < 3; ++j)
      out[i] += mat[i][j] * vec[j];
  }
}

void mat_inv(float mat[3][3], float out[3][3]) {
  double det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
               mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
               mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);

  double invdet = 1 / det;

  out[0][0] = (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) * invdet;
  out[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * invdet;
  out[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * invdet;
  out[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * invdet;
  out[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * invdet;
  out[1][2] = (mat[1][0] * mat[0][2] - mat[0][0] * mat[1][2]) * invdet;
  out[2][0] = (mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1]) * invdet;
  out[2][1] = (mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1]) * invdet;
  out[2][2] = (mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]) * invdet;
}

void XYZ_to_xyY(float XYZ[3], float xyY[3]) {
  xyY[0] = XYZ[0] / (XYZ[0] + XYZ[1] + XYZ[2]);
  xyY[1] = XYZ[1] / (XYZ[0] + XYZ[1] + XYZ[2]);
  xyY[2] = XYZ[1];
}

/*
  Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
  correlated colour temperature using Roberston (1968) method.

  Parameters
  ----------
  CCT
  D_uv

  Returns
  -------
  ndarray
      *CIE UCS* colourspace *uv* chromaticity coordinates.

  References
  ----------
  .. [7]  Wyszecki, G., & Stiles, W. S. (2000). DISTRIBUTION TEMPERATURE,
          COLOR TEMPERATURE, AND CORRELATED COLOR TEMPERATURE. In Color
          Science: Concepts and Methods, Quantitative Data and Formulae
          (pp. 224–229). Wiley. ISBN:978-0471399186
  .. [8]  Adobe Systems. (2013). Adobe DNG Software Development Kit (SDK) -
          1.3.0.0 - dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp::
          dng_temperature::xy_coord. Retrieved from
          https://www.adobe.com/support/downloads/dng/dng_sdk.html
*/
static const float kTintScale = -3000.f;

struct ruvt {
  float r;
  float u;
  float v;
  float t;
};

static const ruvt kTempTable[] =
    {
        {0.f, 0.18006f, 0.26352f, -0.24341f},
        {10.f, 0.18066f, 0.26589f, -0.25479f},
        {20.f, 0.18133f, 0.26846f, -0.26876f},
        {30.f, 0.18208f, 0.27119f, -0.28539f},
        {40.f, 0.18293f, 0.27407f, -0.30470f},
        {50.f, 0.18388f, 0.27709f, -0.32675f},
        {60.f, 0.18494f, 0.28021f, -0.35156f},
        {70.f, 0.18611f, 0.28342f, -0.37915f},
        {80.f, 0.18740f, 0.28668f, -0.40955f},
        {90.f, 0.18880f, 0.28997f, -0.44278f},
        {100.f, 0.19032f, 0.29326f, -0.47888f},
        {125.f, 0.19462f, 0.30141f, -0.58204f},
        {150.f, 0.19962f, 0.30921f, -0.70471f},
        {175.f, 0.20525f, 0.31647f, -0.84901f},
        {200.f, 0.21142f, 0.32312f, -1.0182f},
        {225.f, 0.21807f, 0.32909f, -1.2168f},
        {250.f, 0.22511f, 0.33439f, -1.4512f},
        {275.f, 0.23247f, 0.33904f, -1.7298f},
        {300.f, 0.24010f, 0.34308f, -2.0637f},
        {325.f, 0.24792f, 0.34655f, -2.4681f},
        {350.f, 0.25591f, 0.34951f, -2.9641f},
        {375.f, 0.26400f, 0.35200f, -3.5814f},
        {400.f, 0.27218f, 0.35407f, -4.3633f},
        {425.f, 0.28039f, 0.35577f, -5.3762f},
        {450.f, 0.28863f, 0.35714f, -6.7262f},
        {475.f, 0.29685f, 0.35823f, -8.5955f},
        {500.f, 0.30505f, 0.35907f, -11.324f},
        {525.f, 0.31320f, 0.35968f, -15.628f},
        {550.f, 0.32129f, 0.36011f, -23.325f},
        {575.f, 0.32931f, 0.36038f, -40.770f},
        {600.f, 0.33724f, 0.36051f, -116.45f}
    };

void xyY_to_CCT_tint(float xyY[3], float &CCT, float &tint) {
  // Convert to uv space.
  float u = 2.f * xyY[0] / (1.5f - xyY[0] + 6.f * xyY[1]);
  float v = 3.f * xyY[1] / (1.5f - xyY[0] + 6.f * xyY[1]);

  // Search for line pair coordinate is between.
  float last_dt = 0.0f;

  float last_dv = 0.0f;
  float last_du = 0.0f;

  for (unsigned int index = 1; index <= 30; ++index) {
    // Convert slope to delta-u and delta-v, with length 1.
    float du = 1.0f;
    float dv = kTempTable[index].t;

    float len = sqrt(1.0f + dv * dv);
    du /= len;
    dv /= len;

    // Find delta from black body point to test coordinate.
    float uu = u - kTempTable[index].u;
    float vv = v - kTempTable[index].v;

    // Find distance above or below line.
    float dt = -uu * dv + vv * du;

    // If below line, we have found line pair.
    if (dt <= 0.0f || index == 30) {
      // Find fractional weight of two lines.
      if (dt > 0.0f) dt = 0.0f;
      dt = -dt;

      float f;
      if (index == 1) f = 0.0f;
      else f = dt / (last_dt + dt);

      // Interpolate the temperature.
      CCT = 1.0E6f / (kTempTable[index - 1].r * f + kTempTable[index].r * (1.0f - f));

      // Find delta from black body point to test coordinate.
      uu = u - (kTempTable[index - 1].u * f + kTempTable[index].u * (1.0f - f));
      vv = v - (kTempTable[index - 1].v * f + kTempTable[index].v * (1.0f - f));

      // Interpolate vectors along slope.
      du = du * (1.0f - f) + last_du * f;
      dv = dv * (1.0f - f) + last_dv * f;

      len = sqrt(du * du + dv * dv);

      du /= len;
      dv /= len;

      // Find distance along slope.
      tint = (uu * du + vv * dv) * kTintScale;
      break;
    }

    // Try next line pair.
    last_dt = dt;

    last_du = du;
    last_dv = dv;
  }
}

void CCT_to_CM(const float CCT,
               const float Cal_Illum1,
               const float Cal_Illum2,
               double const cm1[3][3],
               double const cm2[3][3],
               float output[3][3]) {
  if (CCT <= Cal_Illum1)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        output[i][j] = cm1[i][j];
  else if (CCT >= Cal_Illum2)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        output[i][j] = cm2[i][j];
  else {
    float g((1.f / CCT - 1.f / Cal_Illum1) / (1.f / Cal_Illum2 - 1.f / Cal_Illum1));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        output[i][j] = (1 - g) * cm1[i][j] + g * cm2[i][j];
  }
}

void neutral_to_xyY(double const neutral[3],
                    float xyY[3],
                    int c1,
                    double const a[3][3],
                    double const d[3][3]) {
  float last[3] = {D50[0], D50[1], D50[2]}; // Guess D50 white point

  for (int pass = 0; pass < 30; pass++) {
    float xyz2cam[3][3]{}, cam2xyz[3][3]{}, last_temp{}, last_tint{};
    xyY_to_CCT_tint(last, last_temp, last_tint);
    if (c1 == 17) {
      CCT_to_CM(last_temp,
                color::illuminant_standard_light_A,
                color::illuminant_D65,
                a,
                d,
                xyz2cam);
    } else {
      CCT_to_CM(last_temp,
                color::illuminant_standard_light_A,
                color::illuminant_D65,
                d,
                a,
                xyz2cam);
    }
    mat_inv(xyz2cam, cam2xyz);

    float XYZ[3]{}, next[3]{};
    mat_33_31_mult(cam2xyz, neutral, XYZ);
    XYZ_to_xyY(XYZ, next);

    if ((fabs(next[0] - last[0]) + fabs(next[1] - last[1])) < 0.0000001) {
      for (int i = 0; i < 3; ++i)
        xyY[i] = next[i];
      return;
    }

    // If we reach the limit without converging, we are most likely
    // in a two value oscillation.  So take the average of the last
    // two estimates and give up.

    if (pass == 29) {
      next[0] = (last[0] + next[0]) * 0.5f;
      next[1] = (last[1] + next[1]) * 0.5f;
    }

    for (int i = 0; i < 3; ++i)
      last[i] = next[i];
  }

  for (int i = 0; i < 3; ++i)
    xyY[i] = last[i];
}
}