//
// Created by Eddie Barton on 4/18/20.
//

#include "image_io.hh"
#include "HalideBuffer.h"

static inline unsigned short swap2(unsigned short val) {
  unsigned short ret;

  auto *buf = reinterpret_cast<unsigned char *>(&ret);

  unsigned short x = val;
  buf[1] = static_cast<unsigned char>(x);
  buf[0] = static_cast<unsigned char>(x >> 8);

  return ret;
}

//
// Decode 12bit integer image into 16bit image
//
void decode12(Halide::Runtime::Buffer<float> &im, const unsigned char *data, int width, int
height) {
  int offsets[2][2] = {{0, 1}, {1, 2}};

  int bit_shifts[2] = {4, 0};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char buf[3];

      // Calculate load address for 12bit pixel(three 8 bit pixels)
      int n = int(y * width + x);

      // 24 = 12bit * 2 pixel, 8bit * 3 pixel
      int n2 = n % 2;           // used for offset & bit shifts
      int addr3 = (n / 2) * 3;  // 8bit pixel pos
      int odd = (addr3 % 2);

      int bit_shift = bit_shifts[n2];

      int offset[2];
      offset[0] = offsets[n2][0];
      offset[1] = offsets[n2][1];

      buf[0] = data[addr3 + 0];
      buf[1] = data[addr3 + 1];
      buf[2] = data[addr3 + 2];

      auto b0 = static_cast<unsigned int>(buf[offset[0]] & 0xff);
      auto b1 = static_cast<unsigned int>(buf[offset[1]] & 0xff);

      unsigned int val = (b0 << 8) | b1;
      val = 0xfff & (val >> bit_shift);

      im(x, y) = (float)val;
    }
  }
}

//
// Decode 14bit integer image into 16bit integer image
//
void decode14(Halide::Runtime::Buffer<float> &im, const unsigned char *data, int width, int
height) {
  int offsets[4][3] = {{0, 0, 1}, {1, 2, 3}, {3, 4, 5}, {5, 5, 6}};

  int bit_shifts[4] = {2, 4, 6, 0};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char buf[7];

      // Calculate load addres for 14bit pixel(three 8 bit pixels)
      int n = int(y * width + x);

      // 56 = 14bit * 4 pixel, 8bit * 7 pixel
      int n4 = n % 4;           // used for offset & bitshifts
      int addr7 = (n / 4) * 7;  // 8bit pixel pos
      int odd = (addr7 % 2);

      int offset[3];
      offset[0] = offsets[n4][0];
      offset[1] = offsets[n4][1];
      offset[2] = offsets[n4][2];

      int bit_shift = bit_shifts[n4];
      memcpy(buf, &data[addr7], 7);
      auto b0 = static_cast<unsigned int>(buf[offset[0]] & 0xff);
      auto b1 = static_cast<unsigned int>(buf[offset[1]] & 0xff);
      auto b2 = static_cast<unsigned int>(buf[offset[2]] & 0xff);

      // unsigned int val = (b0 << 16) | (b1 << 8) | b2;
      // unsigned int val = (b2 << 16) | (b0 << 8) | b0;
      unsigned int val = (b0 << 16) | (b1 << 8) | b2;
      // unsigned int val = b2;
      val = 0x3fff & (val >> bit_shift);

      im(x, y) = (float)val;
    }
  }
}

//
// Decode 16bit integer image
//
void decode16(Halide::Runtime::Buffer<float> &im, const unsigned char *data, int width, int
height) {
  const uint16_t *ptr = const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(data));

#if _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      im(x, y) = (float)ptr[y * width + x];
    }
  }
}

