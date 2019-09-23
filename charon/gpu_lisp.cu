#include <stdint.h>
#include <math.h>

/*
#define MAX_NODE_SIZE
#define MAX_TREE_SIZE
#define MAX_CONST_SIZE
#define USE_DOUBLE
*/
#define PTR_ERROR 0xFFFF

#define TYPE_NULL 0
#define TYPE_HEADER 1
#define TYPE_CHILD 2
#define TYPE_VALUE 3
#define TYPE_CONST 4

#ifdef USE_DOUBLE
typedef double floatX;
#define ONE 1.0
#define ZERO 0.0
#define MONE -1.0
#else
typedef float floatX;
#define ONE 1.0f
#define ZERO 0.0f
#define MONE -1.0f
#endif


#pragma pack(push,1)
typedef struct {
  uint32_t type; /* 1 - header, 2 - child, 3 - value, 4 - const, 0 - NULL */
  union {
    uint32_t h_opcode;
    uint32_t c_ptr; /* used by child */
    floatX v_val; /* used by value */
    uint32_t cv_ptr; /* used by const */
  };
} text;
#pragma pack(pop)

__device__ floatX sign(floatX x)
{
  if(x > 0.) return 1.;
  else if(x < 0.) return -1.;
  else return 0.;
}
__device__ floatX safe_div(floatX x,floatX y)
{
  if(y == 0.) y = 1e-8;
  else if(fabs(y) < 1e-8) y = sign(y)*1e-8;
  return x/y;
}
__device__ floatX eval2atom(text *bc,const floatX *local_const)
{
  floatX val[MAX_NODE_SIZE];
  uint32_t opcode;
  uint32_t valptr;
  uint32_t i;

  valptr = 0;
  if(bc->type == TYPE_VALUE) {
    return bc->v_val;
  }
  else if(bc->type == TYPE_CONST) {
    return local_const[bc->cv_ptr];
  }

  opcode = bc->h_opcode;
  bc++;

  while(bc->type != TYPE_NULL) {
    if(bc->type == TYPE_VALUE) {
      val[valptr++] = bc->v_val;
    }
    else if(bc->type == TYPE_CONST) {
      val[valptr++] = local_const[bc->cv_ptr];
    }
    else {
      assert(0);
      return NAN;
    }
    bc++;
  }
  switch(opcode) {
  case 0: // "nil",
    return 0.;
  case 1: // first
    return val[0];
  case 2: // "+",
    for(i=1;i<valptr;i++) {
      val[0] += val[i];
    }
    return val[0];
  case 3: // "-",
    return val[0] - val[1];
  case 4: // "*",
    for(i=1;i<valptr;i++) {
      val[0] *= val[i];
    }
    return val[0];
  case 5: // "/",
    return safe_div(val[0],val[1]);
  case 6: // fmod
    if(val[1] == 0) return 0.; // numpy return 0
    else return fmod(val[0],val[1]);
  case 7: // >
    if(val[0] > val[1]) return 1.;
    else return 0.;
  case 8: // <
    if(val[0] < val[1]) return 1.;
    else return 0.;
  case 9: // >=
    if(val[0] >= val[1]) return 1.;
    else return 0.;
  case 10: // <=
    if(val[0] <= val[1]) return 1.;
    else return 0.;
  case 11: // =
    if(val[0] == val[1]) return 1.;
    else return 0.;
  case 12: // !=
    if(val[0] != val[1]) return 1.;
    else return 0.;
  case 13: // if
    if(val[0] != 0.) return val[1];
    else return val[2];
  case 14: // and
    if(val[0] != 0. && val[1] != 0.) return 1.;
    else return 0.;
  case 15: // or
    if(val[0] != 0. || val[1] != 0.) return 1.;
    else return 0.;
  case 16: // not
    if(val[0] != 0.) return 0.;
    else return 1.;
  case 17: // ->
    if(val[0] > val[1]) return 1.;
    else return -1.;
  case 18: // -<
    if(val[0] < val[1]) return 1.;
    else return -1.;
  case 19: // ->=
    if(val[0] >= val[1]) return 1.;
    else return -1.;
  case 20: // -<=
    if(val[0] <= val[1]) return 1.;
    else return -1.;
  case 21: // -=
    if(val[0] == val[1]) return 1.;
    else return -1.;
  case 22: // -!=
    if(val[0] != val[1]) return 1.;
    else return -1.;
  case 23: // -if
    if(val[0] > 0.) return val[1];
    else return val[2];
  case 24: // -and
    if(val[0] > 0. && val[1] > 0.) return 1.;
    else return -1.;
  case 25: // -or
    if(val[0] > 0. || val[1] > 0.) return 1.;
    else return -1.;
  case 26: // -not
    if(val[0] > 0.) return -1.;
    else return 1.;
  case 27: // sin
    return sin(val[0]);
  case 28: // cos
    return cos(val[0]);
  case 29: // tan
    return tan(val[0]);
  case 30: // asin
    return asin(fmod(val[0],ONE));
  case 31: // acos
    return acos(fmod(val[0],ONE));
  case 32: // atan
    return atan(val[0]);
  case 33: // sqrt
    return sqrt(fabs(val[0]));
  case 34: // log
    return log(fabs(val[0]));
  case 35: // exp
    return exp(val[0]);
  case 36: // pow
    return pow(val[0],val[1]);
  case 37: // abs
    return fabs(val[0]);
  case 38: // max
    for(i=1;i<valptr;i++) {
      if(val[0] < val[i]) val[0] = val[i];
    }
    return val[0];
  case 39: // min
    for(i=1;i<valptr;i++) {
      if(val[0] > val[i]) val[0] = val[i];
    }
    return val[0];
  case 40: // count
    return valptr;
  case 41: // avg
    for(i=1;i<valptr;i++) {
      val[0] += val[i];     
    }
    val[0] /= (floatX)valptr;
    return val[0];
  case 42: // sign
    return sign(val[0]);
  case 43: // sigmoid
    return ONE/(ONE + exp(val[0]));
  case 44: // sinh
    return sinh(val[0]);
  case 45: // cosh
    return cosh(val[0]);
  case 46: // tanh
    return tanh(val[0]);
  case 47: // asinh
    return asinh(val[0]);
  case 48: // acosh
    return acosh(val[0]);
  case 49: // atanh
    return atanh(val[0]);
  case 50: // relu
    return (val[0] > 0.) ? val[0] : 0.;
  case 51: // softplus
    return log1p(exp(val[0]));
  case 52: // log1p
    return log1p(val[0]);
  case 53: // expm1
    return expm1(val[0]);
  default:
    break;
  }

  assert(0);
  return NAN;
}
__global__ void reduce(floatX *out,text *text_memory,const floatX *const_memory,const int N,const int M)
{
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);
  
  const floatX *local_const;
  text *local_text;
  // evaluate the text at 0
  text *bc_stack[MAX_TREE_SIZE];
  text *bcp_stack[MAX_TREE_SIZE];
  text *current;
  text *currentp;
  uint32_t stackp = 0;
  floatX val;

  if(idx >= N) return;
  local_text = &text_memory[idx*MAX_TREE_SIZE*MAX_NODE_SIZE];
  if(M == 0) {
    local_const = &const_memory[idx*MAX_CONST_SIZE];
  }
  else if(M > 0) {
    local_const = &const_memory[(idx%M)*MAX_CONST_SIZE];
  }
  else if(M < 0) {
    local_const = const_memory;
  }

  currentp = local_text;
  current = currentp;

  for(;;) {
    if(current->type == TYPE_HEADER) {
      // do nothing, it is good
    }
    else if(current->type == TYPE_CHILD) { // go to next node
      bc_stack[stackp] = current;
      bcp_stack[stackp] = currentp;
      stackp++;
      currentp = &local_text[current->c_ptr*MAX_NODE_SIZE];
      current = currentp;
    }
    else if(current->type == TYPE_VALUE || current->type == TYPE_CONST) {
      // do nothing, it is good 
    }
    else if(current->type == TYPE_NULL) {
      if(stackp == 0) break; // reduce complete
      // evaluate value
      val = eval2atom(currentp,local_const);
      stackp--;
      current = bc_stack[stackp];
      currentp = bcp_stack[stackp];
      current->type = TYPE_VALUE;
      current->v_val = val;
    }
    else {
      assert(0);
      out[idx] = NAN;
      return;
    }
    current++;
  }
  out[idx] = eval2atom(local_text,local_const);
}
