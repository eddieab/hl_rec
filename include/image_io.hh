//
// Created by Eddie Barton on 4/18/20.
//
#include <vector>
#include <cstring>

#include "HalideBuffer.h"

#ifndef HL_REC_SRC_IMAGE_IO_HH_
#define HL_REC_SRC_IMAGE_IO_HH_

static inline unsigned short swap2(unsigned short);
void decode12(Halide::Runtime::Buffer<float> &, const unsigned char *, int, int);
void decode14(Halide::Runtime::Buffer<float> &, const unsigned char *, int, int);
void decode16(Halide::Runtime::Buffer<float> &, const unsigned char *, int, int);

#endif //HL_REC_SRC_IMAGE_IO_HH_
