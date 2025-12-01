#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath>

namespace needle {
namespace hip {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct HipArray {
  HipArray(const size_t size) {
    hipError_t err = hipMalloc(&ptr, size * ELEM_SIZE);
    if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
    this->size = size;
  }
  ~HipArray() { hipFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct HipDims {
  dim3 block, grid;
};

HipDims HipOneDim(size_t size) {
  /**
   * Utility function to get hip dimensions for 1D call
   */
  HipDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct HipVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

// Transfer STL vector to HipVec struct
HipVec VecToHip(const std::vector<int32_t>& x) {
  HipVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(HipArray* out, scalar_t val) {
  HipDims dim = HipOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t IdxConversion(size_t remaining, HipVec shape, HipVec strides){
    size_t new_idx, dim_idx;
    new_idx = 0;
    for (int dim = shape.size-1; dim >=0 ; dim--)
    {
       dim_idx = remaining % shape.data[dim];
       remaining /= shape.data[dim];
       new_idx += dim_idx * strides.data[dim];
    }
    return new_idx;    
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, HipVec shape,
                              HipVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type HipVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  /// BEGIN SOLUTION  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  
  if(gid >= size) return;
  size_t non_compact_idx = offset;
  // index conversion from compact index to non compact index
  non_compact_idx += IdxConversion(gid,shape,strides);
  
  out[gid] = a[non_compact_idx]; 
  
  /// END SOLUTION
}

void Compact(const HipArray& a, HipArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * nm 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the   array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  HipDims dim = HipOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToHip(shape),
                                         VecToHip(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,HipVec shape,
                  HipVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  
    if(gid >= size) return;
    size_t non_compact_idx = offset;
    // index conversion from compact index to non compact index
    non_compact_idx += IdxConversion(gid,shape,strides);
    
    out[non_compact_idx] = a[gid]; 

}


void EwiseSetitem(const HipArray& a, HipArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  HipDims dim = HipOneDim(out->size);
  // caculate total number of elements in a
  size_t total_size = 1;
  for (size_t dim : shape)
  {
      total_size *= dim;
  }
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, total_size, VecToHip(shape),
                                         VecToHip(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size,HipVec shape,
                  HipVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  
    size_t non_compact_idx = offset;
    if(gid >= size) return;
    // index conversion from compact index to non compact index
    non_compact_idx += IdxConversion(gid,shape,strides);
    
    out[non_compact_idx] = val; 

}


void ScalarSetitem(size_t size, scalar_t val, HipArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  HipDims dim = HipOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToHip(shape),
                                         VecToHip(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const HipArray& a, const HipArray& b, HipArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  HipDims dim = HipOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  HipDims dim = HipOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

////////////////////////////////////////////////////////////////////////////////
// use C++ template to write kernel function
template<typename Op>
__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  // template kernel to use for element-wise operation
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = Op::apply(a[gid],b[gid]);
}

template<typename Op>
__global__ void SingleEwiseOpKernel(const scalar_t* a, scalar_t* out, size_t size){
  // template kernel to use for element-wise operation
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = Op::apply(a[gid]);
}

template<typename Op>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // template kernel to use for scalar operation
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = Op::apply(a[gid],val);
}

////////////////////////////////////////////////////////////////////////////////
// define Ops
struct MulOp {
  // apply is a function member in MulOp
  // __device__, __host__ means this function can be called by both CPU functions and GPU functions
  __device__ __host__ static scalar_t apply(scalar_t a, scalar_t b) { return a * b; }
};

struct DivOp {
  __device__ __host__ static scalar_t apply(scalar_t a, scalar_t b) { return a / b; }
};

struct PowerOp{
  __device__ __host__ static scalar_t apply(scalar_t a, scalar_t b) { return std::pow(a,b); }
};

struct MaxOp{
  __device__ __host__ static scalar_t apply(scalar_t a, scalar_t b) { return (a>b)?a:b; }
};

struct EqOp{
  __device__ __host__ static scalar_t apply(scalar_t a, scalar_t b) { return (a==b)?true:false; }
};

struct GeOp{
  __device__ __host__ static scalar_t apply(scalar_t a, scalar_t b) { return (a>=b)?true:false; }
};

struct LogOp{
  __device__ __host__ static scalar_t apply(scalar_t a) { return std::log(a); }
};

struct ExpOp{
  __device__ __host__ static scalar_t apply(scalar_t a) { return std::exp(a); }
};

struct TanhOp{
  __device__ __host__ static scalar_t apply(scalar_t a) { return std::tanh(a); }
};

////////////////////////////////////////////////////////////////////////////////
// template-like function interface to call kernels

template<typename Op>
void EwiseOpTemplate(const HipArray& a, const HipArray& b, HipArray* out) {
  HipDims dim = HipOneDim(out->size);
  EwiseOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

template<typename Op>
void SingleEwiseOpTemplate(const HipArray& a, HipArray* out) {
  HipDims dim = HipOneDim(out->size);
  SingleEwiseOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size);
}

template<typename Op>
void ScalarOpTemplate(const HipArray& a, scalar_t val, HipArray* out) {
  HipDims dim = HipOneDim(out->size);
  ScalarOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

void EwiseMul(const HipArray& a, const HipArray& b, HipArray* out) {
  /**
   * Multiply together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<MulOp>(a,b,out);
}

void ScalarMul(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Multiply a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<MulOp>(a,val,out);
}

void EwiseDiv(const HipArray& a, const HipArray& b, HipArray* out) {
  /**
   * Divide together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<DivOp>(a,b,out);
}

void ScalarDiv(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Divide a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<DivOp>(a,val,out);
}

void ScalarPower(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Set entries in out to be the power of correspondings entires in a and scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<PowerOp>(a,val,out);
}

void EwiseMaximum(const HipArray& a, const HipArray& b, HipArray* out) {
  /**
   * Set entries in out to find larger elements between a and b in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<MaxOp>(a,b,out);
}

void ScalarMaximum(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Set entries in out to find larger elements between each element of a and scalar val.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<MaxOp>(a,val,out);
}

void EwiseEq(const HipArray& a, const HipArray& b, HipArray* out) {
  /**
   * Set entries in out to verify if a == b in element-wise
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<EqOp>(a,b,out);
}

void ScalarEq(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Set entries in out to verify if each element of a == val.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<EqOp>(a,val,out);
}

void EwiseGe(const HipArray& a, const HipArray& b, HipArray* out) {
  /**
   * Set entries in out to verify if a >= b in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<GeOp>(a,b,out);
}

void ScalarGe(const HipArray& a, scalar_t val, HipArray* out) {
  /**
   * Set entries in out to verify if each element of a >= val.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<GeOp>(a,val,out);
}

void EwiseLog(const HipArray& a, HipArray* out) {
  /**
   * Set entries in out to find log(a) in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   SingleEwiseOpTemplate<LogOp>(a,out);
}

void EwiseExp(const HipArray& a, HipArray* out) {
  /**
   * Set entries in out to find exp(a) in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   SingleEwiseOpTemplate<ExpOp>(a,out);
}

void EwiseTanh(const HipArray& a, HipArray* out) {
  /**
   * Set entries in out to find tanh(a) in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   SingleEwiseOpTemplate<TanhOp>(a,out);
}






////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

const int S=32;
const int L=32;
const int V = TILE;

__global__ void MMKernel(const scalar_t* A, const scalar_t* B, scalar_t* C, uint32_t M, uint32_t N,
            uint32_t P){
    __shared__ scalar_t sA[S][L];
    __shared__ scalar_t sB[S][L];

    // each thread calculate VxV matrix, the result is stored in registers
    scalar_t c[V][V] = {0};
    scalar_t a[V],b[V];
    // current thread block indices
    int block_row = blockIdx.y;  // M dim
    int block_col = blockIdx.x;  // P dim
    
    int thread_row = threadIdx.y;  // row index in block
    int thread_col = threadIdx.x;  // col index in block

    // the location of each VxV matrix calculated by a thread in global output matrix
    int global_row = block_row * L + thread_row * V;
    int global_col = block_col * L + thread_col * V;

    // thread cooperative fetching for each SxL block in A and B
    for (int k0 = 0; k0 < N; k0+=S)
    {
      __syncthreads();
      // row in sA is indeed a column in A's SxL block
      for (int i = thread_row; i < L; i+=blockDim.y)
      {
        for (int j = thread_col; j<S; j+=blockDim.x)
        {
          int load_row_A = block_row * L + i; // dim M
          int load_col_A = k0 + j; // dim N
          if(load_row_A < M && load_col_A < N)
            sA[j][i] = A[load_row_A * N + load_col_A];
          else
            sA[j][i] = 0.0f;
        }
      }
      // row in sB is indeed a row in B's SxL block
      for (int i = thread_row; i < L; i+=blockDim.y)
      {
        for (int j = thread_col; j<S; j+=blockDim.x)
        {
          int load_row_B = k0 + j; // dim N
          int load_col_B = block_col*L + i; // dim P
          if(load_row_B < N && load_col_B < P)
            sB[j][i] = B[load_row_B * P + load_col_B];
          else
            sB[j][i] = 0.0f;
        }
      }
      __syncthreads();

      // calculate a VxV result
      for (int k_inner = 0; k_inner < S && (k0 + k_inner) < N; k_inner++) {
        // load one line of sA and sB to register
        for (int v = 0; v < V; v++) {
            int sA_col = thread_row * V + v;
            int sB_col = thread_col * V + v;
            if (sA_col < L) 
              a[v] = sA[k_inner][sA_col];
            if (sB_col < L) 
              b[v] = sB[k_inner][sB_col];
            // calcualate VxV part by outer product
        }
        // after loading all a and b ,calculate the result
        for (int y = 0; y < V; y++) {
              for (int x = 0; x < V; x++) {
                  c[y][x] += a[y] * b[x];
              }
        }   
      }
    }

    for (int y = 0; y < V; y++) {
        for (int x = 0; x < V; x++) {
            int row = global_row + y;
            int col = global_col + x;
            if (row < M && col < P) {
                C[row * P + col] = c[y][x];
            }
        }
    }
}


void Matmul(const HipArray& a, const HipArray& b, HipArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  // specify thread number for each block
  dim3 blockSize(L/V,L/V);
  dim3 gridSize((P+L-1)/L, (M+L-1)/L);

  MMKernel<<<gridSize,blockSize>>>(a.ptr,b.ptr,out->ptr,M,N,P);
  hipDeviceSynchronize();
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out,size_t out_size,size_t reduce_size){
  // taking maximum over `reduce_size` contiguous blocks.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid>=out_size) return;
  scalar_t max = a[gid * reduce_size]; // locate each last dimension
  // find the maximum value in the last dimemsion
  for (size_t i = 1; i < reduce_size; i++)
  {
     scalar_t this_element = a[gid * reduce_size + i];
     if (this_element > max)
       max = this_element;     
  }
  out[gid] = max;  
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out,size_t out_size,size_t reduce_size){
  // Reduce by taking sum over `reduce_size` contiguous blocks.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid>=out_size) return;
  scalar_t sum = a[gid * reduce_size]; // locate each last dimension
  // find the maximum value in the last dimemsion
  for (size_t i = 1; i < reduce_size; i++)
  {
     sum += a[gid * reduce_size + i];   
  }
  out[gid] = sum;  
}



void ReduceMax(const HipArray& a, HipArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  HipDims dim = HipOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size,reduce_size);
  /// END SOLUTION
}



void ReduceSum(const HipArray& a, HipArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  HipDims dim = HipOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size,reduce_size);
  /// END SOLUTION
}


////////////////////////////////////////////////////////////////////////////////
// new backend operators
////////////////////////////////////////////////////////////////////////////////
void Conv(const HipArray& a, const HipArray& b, HipArray* out,
          uint32_t batch, uint32_t in_channels, uint32_t in_height, uint32_t in_width,
          uint32_t out_channels, uint32_t kernel_h, uint32_t kernel_w,
          uint32_t stride, uint32_t padding) {
    /**
     * Perform 2D convolution using MIOpen library.
     * 
     * Args:
     *   a: input array of shape (batch, in_channels, in_height, in_width)
     *   b: kernel array of shape (out_channels, in_channels, kernel_h, kernel_w)
     *   out: output array
     *   batch: batch size
     *   in_channels: number of input channels
     *   in_height: input height
     *   in_width: input width
     *   out_channels: number of output channels (filters)
     *   kernel_h: kernel height
     *   kernel_w: kernel width
     *   stride: stride for convolution
     *   padding: padding for convolution
     */
    
    miopenHandle_t handle;
    miopenStatus_t status = miopenCreate(&handle);
    if (status != miopenStatusSuccess) {
      throw std::runtime_error("Failed to create MIOpen handle");
    }
    
    // Create tensor descriptors
    miopenTensorDescriptor_t input_desc, output_desc, kernel_desc;
    miopenCreateTensorDescriptor(&input_desc);
    miopenCreateTensorDescriptor(&output_desc);
    miopenCreateTensorDescriptor(&kernel_desc);
    
    // Create convolution descriptor
    miopenConvolutionDescriptor_t conv_desc;
    miopenCreateConvolutionDescriptor(&conv_desc);
    
    // Set input tensor descriptor (NCHW format)
    miopenSet4dTensorDescriptor(input_desc, miopenFloat, batch, in_channels, in_height, in_width);
    
    // Set kernel tensor descriptor (NCHW format)
    miopenSet4dTensorDescriptor(kernel_desc, miopenFloat, out_channels, in_channels, kernel_h, kernel_w);
    
    // Initialize convolution descriptor
    miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, padding, padding, stride, stride, 1, 1);
    
    // Get output dimensions
    int out_n, out_c, out_h, out_w;
    status = miopenGetConvolutionForwardOutputDim(conv_desc, input_desc, kernel_desc, &out_n, &out_c, &out_h, &out_w);
    if (status != miopenStatusSuccess)
    {
      throw std::runtime_error("Fail to get conv output dims");
    }
   
    // Set output tensor descriptor
    miopenSet4dTensorDescriptor(output_desc, miopenFloat, out_n, out_c, out_h, out_w);

    // Sanity: output buffer size must match
    size_t expected = static_cast<size_t>(out_n) * out_c * out_h * out_w;
    if (expected != out->size) {
      std::ostringstream oss;
      oss << "Conv output size mismatch. MIOpen=("
          << out_n << "," << out_c << "," << out_h << "," << out_w
          << ") total=" << expected << " but out->size=" << out->size;
      throw std::runtime_error(oss.str());
    }
    
    // Find the best convolution algorithm
    size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(handle, kernel_desc, input_desc, conv_desc, output_desc, &workspace_size);
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
      hipMalloc(&workspace, workspace_size);
    }
    
    // // Get algorithm
    // miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoGEMM;

    miopenConvAlgoPerf_t perf_results;
    int returned_algo_count = 0;
    miopenFindConvolutionForwardAlgorithm(
        handle, input_desc, a.ptr, kernel_desc, b.ptr,
        conv_desc, output_desc, out->ptr, 
        1, &returned_algo_count, &perf_results, 
        workspace, workspace_size, false);
        
    miopenConvFwdAlgorithm_t algo = perf_results.fwd_algo;
    
    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionForward(handle, &alpha, input_desc, a.ptr, kernel_desc, b.ptr,
                            conv_desc, algo, &beta, output_desc, out->ptr,
                            workspace, workspace_size);
    
    // Ensure conv completes before further ops (e.g., your HIP reduce/sum)
    hipDeviceSynchronize();

    // Cleanup
    if (workspace) hipFree(workspace);
    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(kernel_desc);
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroy(handle);
  }

void BatchNorm2D(const HipArray& input, HipArray* output,
                 const HipArray& scale, const HipArray& bias,
                 HipArray& running_mean, HipArray& running_var,
                 uint32_t batch, uint32_t channels, uint32_t height, uint32_t width,
                 float eps = 1e-5f) {
    /**
     * Perform 2D Batch Normalization using MIOpen library.
     * TODO: Currently this batchnorm2d implementation is inference-only, add train feature if needed
     * Args:
     *   input: input array of shape (batch, channels, height, width)
     *   output: output array
     *   scale: scale parameters (gamma) of shape (channels,)
     *   bias: bias parameters (beta) of shape (channels,)
     *   running_mean: running mean of shape (channels,)
     *   running_var: running variance of shape (channels,)
     *   batch: batch size
     *   channels: number of channels
     *   height: input height
     *   width: input width
     *   momentum: momentum for running statistics update
     *   eps: epsilon for numerical stability
     */
    
    miopenHandle_t handle;
    miopenStatus_t status = miopenCreate(&handle);
    if (status != miopenStatusSuccess) {
      throw std::runtime_error("Failed to create MIOpen handle");
    }

    // Create tensor descriptors
    miopenTensorDescriptor_t input_desc, output_desc, bn_desc;
    miopenCreateTensorDescriptor(&input_desc);
    miopenCreateTensorDescriptor(&output_desc);
    miopenCreateTensorDescriptor(&bn_desc);

    // Set input/output tensor descriptors (NCHW format)
    miopenSet4dTensorDescriptor(input_desc, miopenFloat, batch, channels, height, width);
    miopenSet4dTensorDescriptor(output_desc, miopenFloat, batch, channels, height, width);
    
    // Set batch norm descriptor for per-activation mode (spatial BN)
    miopenSet4dTensorDescriptor(bn_desc, miopenFloat, 1, channels, 1, 1);

    // Choose batch normalization mode
    miopenBatchNormMode_t bn_mode = miopenBNSpatial;  // Spatial Batch Normalization
    
    float alpha = 1.0f, beta = 0.0f;

    // Inference mode: use running statistics
    miopenBatchNormalizationForwardInference(
        handle, bn_mode, &alpha, &beta,
        input_desc, input.ptr,     // input
        output_desc, output->ptr,  // output
        bn_desc, scale.ptr, bias.ptr,           // scale (gamma), bias (beta)
        running_mean.ptr, running_var.ptr,      // running mean & variance
        eps                        // epsilon
    );

    // Cleanup
    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(bn_desc);
    miopenDestroy(handle);
  }

__global__ void ReLUKernel(const scalar_t* input, scalar_t* output, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    output[gid] = fmaxf(0.0f, input[gid]);
  }
}

__global__ void addChannelBiasKernel(const scalar_t* input, const scalar_t* bias, scalar_t* output,
                                 uint32_t batch, uint32_t channels, uint32_t height, uint32_t width) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = static_cast<size_t>(batch) * channels * height * width;
  if (gid < total_size) {
    size_t c = (gid / (height * width)) % channels; // Calculate channel index
    output[gid] = input[gid] + bias[c];
  }
}

void ConvBNReluFused(const HipArray& input, HipArray* output,
                     const HipArray& weight, const HipArray& bias,
                     const HipArray& scale, const HipArray& shift,
                     const HipArray& running_mean, const HipArray& running_var,
                     uint32_t batch, uint32_t in_channels, uint32_t in_height, uint32_t in_width,
                     uint32_t out_channels, uint32_t kernel_h, uint32_t kernel_w,
                     uint32_t stride, uint32_t padding, float eps = 1e-5f) {
  
  miopenHandle_t handle;
  miopenStatus_t status = miopenCreate(&handle);
  if (status != miopenStatusSuccess) {
    throw std::runtime_error("Fail to create MIOpen handle");
  }

  // Input/output descriptors (NCHW)
  miopenTensorDescriptor_t input_desc, output_desc, weight_desc, bn_desc;
  miopenCreateTensorDescriptor(&input_desc);
  miopenCreateTensorDescriptor(&output_desc);
  miopenCreateTensorDescriptor(&weight_desc);
  miopenCreateTensorDescriptor(&bn_desc);


  // Output dims from conv
  miopenConvolutionDescriptor_t conv_desc;
  miopenCreateConvolutionDescriptor(&conv_desc);
  // set descriptors (NCHW)
  miopenSet4dTensorDescriptor(input_desc, miopenFloat, batch, in_channels, in_height, in_width);
  miopenSet4dTensorDescriptor(weight_desc, miopenFloat, out_channels, in_channels, kernel_h, kernel_w);
  miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                  padding, padding, stride, stride, 1, 1);

  int out_n, out_c, out_h, out_w;
  status = miopenGetConvolutionForwardOutputDim(conv_desc, input_desc, weight_desc,
                                                &out_n, &out_c, &out_h, &out_w);

  // std::cout << "Conv output dims: (" << out_n << "," << out_c << "," << out_h << "," << out_w << ")\n";
  if (status != miopenStatusSuccess) {
    throw std::runtime_error("Fail to get conv output dims");
  }
  miopenSet4dTensorDescriptor(output_desc, miopenFloat, out_n, out_c, out_h, out_w);

  // Sanity: output buffer size must match
  size_t total = static_cast<size_t>(out_n) * out_c * out_h * out_w;
  if (total != output->size) {
    std::ostringstream oss;
    oss << "Conv output size mismatch. MIOpen=("
        << out_n << "," << out_c << "," << out_h << "," << out_w
        << ") total=" << total << " but out->size=" << output->size;
    throw std::runtime_error(oss.str());
  }
  
  // Workspace for convolution
  size_t workspace_size = 0;
  miopenConvolutionForwardGetWorkSpaceSize(handle, weight_desc, input_desc, conv_desc, output_desc, &workspace_size);
  
  // Allocate workspace
  void* workspace = nullptr;
  if (workspace_size > 0) {
    hipMalloc(&workspace, workspace_size);
  }
  
  // // Get algorithm
  // miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoGEMM;

  miopenConvAlgoPerf_t perf_results;
  int returned_algo_count = 0;
  miopenFindConvolutionForwardAlgorithm(
      handle, input_desc, input.ptr, weight_desc, weight.ptr,
      conv_desc, output_desc, output->ptr, 
      1, &returned_algo_count, &perf_results, 
      workspace, workspace_size, false);
      
  miopenConvFwdAlgorithm_t algo = perf_results.fwd_algo;
  
  // Perform convolution
  float alpha = 1.0f, beta = 0.0f;
  status = miopenConvolutionForward(handle, &alpha, input_desc, input.ptr, weight_desc, weight.ptr,
                          conv_desc, algo, &beta, output_desc, output->ptr,
                          workspace, workspace_size);
                         
  hipDeviceSynchronize();
  // Ensure conv completes before further ops (e.g., your HIP reduce/sum)
  if(status != miopenStatusSuccess) {
    throw std::runtime_error("Fail to execute convolution forward");
  }

  // Add bias
  HipDims dim = HipOneDim(total);
  addChannelBiasKernel<<<dim.grid, dim.block>>>(output->ptr, bias.ptr, output->ptr,
                                                out_n, out_c, out_h, out_w);
  hipDeviceSynchronize();

  // BatchNorm (inference mode)
  miopenSet4dTensorDescriptor(bn_desc, miopenFloat, 1, out_c, 1, 1);
  status = miopenBatchNormalizationForwardInference(
      handle, miopenBNSpatial, &alpha, &beta,
      output_desc, output->ptr,     // input
      output_desc, output->ptr,  // output
      bn_desc, scale.ptr, shift.ptr,           // scale (gamma), bias (beta)
      running_mean.ptr, running_var.ptr,      // running mean & variance
      eps                        // epsilon
  );
  if (status != miopenStatusSuccess) {
    throw std::runtime_error("Fail to execute batchnorm forward inference");
  }

  // ReLU activation
  ReLUKernel<<<dim.grid, dim.block>>>(output->ptr, output->ptr, total);
  hipDeviceSynchronize();

  // Cleanup
  if (workspace) hipFree(workspace);
  miopenDestroyConvolutionDescriptor(conv_desc);
  miopenDestroyTensorDescriptor(bn_desc);
  miopenDestroyTensorDescriptor(weight_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(input_desc);
  miopenDestroy(handle);
}

__global__ void max_pool2d(const scalar_t* input, scalar_t* output,int batch_size, int channels,
                            int input_h, int input_w, int output_h, int output_w,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w){

    // index and param relates to each block is processing
    const int out_x_base = blockIdx.x * blockDim.x;
    const int out_y_base = blockIdx.y * blockDim.y;
    const int c = blockIdx.z; 
    const int n = c / channels; // 第n个样本
    const int channel = c % channels;

    // 该block对应输入图片起始位置
    const int input_x_start = out_x_base * stride_w - pad_w;
    const int input_y_start = out_y_base * stride_h - pad_h;
    
    // share memory大小
    const int shared_w = (blockDim.x -1) * stride_w + kernel_w;
    const int shared_h = (blockDim.y -1) * stride_h + kernel_h;
    extern __shared__ float shared_input[];

    // 协作加载input data到share memory
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < shared_h * shared_w; idx+= blockDim.x * blockDim.y)
    {
        // idx在share memory的x,y坐标
        const int local_y = idx / shared_w;
        const int local_x = idx % shared_w;
        // idx转换到整张图片的输入位置
        const int global_y = input_y_start + local_y;
        const int global_x = input_x_start + local_x;
        // 转换的global索引边界检查
        if (global_y >=0 && global_y < input_h && global_x >=0 && global_x < input_w)
        {
            const int global_idx = ((n * channels + channel) * input_h + global_y) * input_w + global_x;
            shared_input[local_y * shared_w + local_x] = input[global_idx];
        }else{
            shared_input[local_y * shared_w + local_x] = -INFINITY;
        }
        
    }

    __syncthreads();

    // 计算每个线程对应的输出图片的idx
    const int output_x = out_x_base + threadIdx.x;
    const int output_y = out_y_base + threadIdx.y;

    if (output_x < output_w && output_y < output_h )
    {
        // 计算share memory的起始位置
        const int share_start_x = output_x * stride_w - pad_w - input_x_start;
        const int share_start_y = output_y * stride_h - pad_h - input_y_start;

        float max_val = -INFINITY;
        // 一个线程会找一个kernel sizexkernel size图片内的最大值
        for (int kh = 0; kh < kernel_h; kh++)
        {
            for (int kw = 0; kw < kernel_w; kw++)
            {
                const int shared_x = share_start_x + kw;
                const int shared_y = share_start_y + kh;
                if (shared_x >= 0 && shared_x < shared_w && shared_y >=0 && shared_y < shared_h)
                {
                    max_val = fmaxf(max_val,shared_input[shared_y * shared_w + shared_x]);
                }
            }   
        } 
        // LOAD TO OUTPUT 
        if(max_val > -INFINITY){
            const int output_idx = ((n*channels + channel) * output_h + output_y) * output_w + output_x;
            output[output_idx] = max_val;
        }
    }
}

void MaxPool2D(const HipArray& input, HipArray* output,
               uint32_t batch_size, uint32_t channels,
               uint32_t input_h, uint32_t input_w,
               uint32_t kernel_h, uint32_t kernel_w,
               uint32_t stride_h, uint32_t stride_w,
               uint32_t pad_h, uint32_t pad_w) {
    /**
     * Perform 2D Max Pooling using custom kernel.
     * 
     * Args:
     *   input: input array of shape (batch_size, channels, input_h, input_w)
     *   output: output array
     *   batch_size: batch size
     *   channels: number of channels
     *   input_h: input height
     *   input_w: input width
     *   kernel_h: kernel height
     *   kernel_w: kernel width
     *   stride_h: stride height
     *   stride_w: stride width
     *   pad_h: padding height
     *   pad_w: padding width
     */

    // Calculate output dimensions
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    dim3 blockDim(16, 16);
    dim3 gridDim((output_w + blockDim.x - 1) / blockDim.x,
                  (output_h + blockDim.y - 1) / blockDim.y,
                  batch_size * channels);

    size_t shared_mem_size = ((blockDim.x -1) * stride_w + kernel_w) *
                             ((blockDim.y -1) * stride_h + kernel_h) *
                             sizeof(scalar_t);

    max_pool2d<<<gridDim, blockDim, shared_mem_size>>>(input.ptr, output->ptr,
                                                          batch_size, channels,
                                                          input_h, input_w,
                                                          output_h, output_w,
                                                          kernel_h, kernel_w,
                                                          stride_h, stride_w,
                                                          pad_h, pad_w);
    hipDeviceSynchronize();
    }

}  // namespace hip
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_hip, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace hip;

  m.attr("__device_name__") = "hip";
  m.attr("__tile_size__") = TILE;

  py::class_<HipArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &HipArray::size)
      .def("ptr", &HipArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const HipArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    hipError_t err = hipMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, hipMemcpyDeviceToHost);
    if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, HipArray* out) {
    hipError_t err =
        hipMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, hipMemcpyHostToDevice);
    if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("conv",Conv);
  m.def("batchnorm2d",BatchNorm2D);
  m.def("convbn2drelu",ConvBNReluFused);
  m.def("maxpool2d",MaxPool2D);
}
