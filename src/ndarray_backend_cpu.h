#pragma once
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct AlignedArray {
    AlignedArray(const size_t size);
    ~AlignedArray();
    size_t ptr_as_int();
    scalar_t* ptr;
    size_t size;
};

void Fill(AlignedArray* out, scalar_t val);
void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
            std::vector<int32_t> strides, size_t offset);
void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out);
void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out);

} // namespace cpu
} // namespace needle