#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

// Transfer STL vector to CudaVec struct
CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
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

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t IdxConversion(size_t remaining, CudaVec shape, CudaVec strides){
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


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
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

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
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
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,CudaVec shape,
                  CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  
    if(gid >= size) return;
    size_t non_compact_idx = offset;
    // index conversion from compact index to non compact index
    non_compact_idx += IdxConversion(gid,shape,strides);
    
    out[non_compact_idx] = a[gid]; 

}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
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
  CudaDims dim = CudaOneDim(out->size);
  // caculate total number of elements in a
  size_t total_size = 1;
  for (size_t dim : shape)
  {
      total_size *= dim;
  }
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, total_size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size,CudaVec shape,
                  CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  
    size_t non_compact_idx = offset;
    if(gid >= size) return;
    // index conversion from compact index to non compact index
    non_compact_idx += IdxConversion(gid,shape,strides);
    
    out[non_compact_idx] = val; 

}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
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
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
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

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

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
void EwiseOpTemplate(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

template<typename Op>
void SingleEwiseOpTemplate(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  SingleEwiseOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size);
}

template<typename Op>
void ScalarOpTemplate(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Multiply together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<MulOp>(a,b,out);
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Multiply a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<MulOp>(a,val,out);
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Divide together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<DivOp>(a,b,out);
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<DivOp>(a,val,out);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Set entries in out to be the power of correspondings entires in a and scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<PowerOp>(a,val,out);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Set entries in out to find larger elements between a and b in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<MaxOp>(a,b,out);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Set entries in out to find larger elements between each element of a and scalar val.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<MaxOp>(a,val,out);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Set entries in out to verify if a == b in element-wise
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<EqOp>(a,b,out);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Set entries in out to verify if each element of a == val.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<EqOp>(a,val,out);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Set entries in out to verify if a >= b in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   EwiseOpTemplate<GeOp>(a,b,out);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Set entries in out to verify if each element of a >= val.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  ScalarOpTemplate<GeOp>(a,val,out);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Set entries in out to find log(a) in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   SingleEwiseOpTemplate<LogOp>(a,out);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Set entries in out to find exp(a) in element-wise.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */

   SingleEwiseOpTemplate<ExpOp>(a,out);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
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


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
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
  cudaDeviceSynchronize();
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



void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
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
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size,reduce_size);
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
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
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size,reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
}
