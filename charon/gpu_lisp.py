#!/usr/bin/python3
import os
import struct, copy, operator
import numpy as np
from .util import *

__all__ = [
    'gpu_lisp_error',
    'gpu_lisp_text',
    'gpu_lisp_symbol',
    'gpu_lisp_constant',
    'gpu_lisp',
    'TYPE_NULL',
    'TYPE_HEADER',
    'TYPE_CHILD',
    'TYPE_VALUE',
    'TYPE_CONST',
    'OPCODE',
    'OPCODE_INVERSE',
    'ERROR_GENERIC',
    'ERROR_TEXTFULL',
    'ERROR_SYNTAX',
    'ERROR_OPCODENOTFOUND',
    'ERROR_UNKTYPE',
    'ERROR_RUNTIME',
    'HAS_GPU',
]       

HAS_DISABLE_GPU=False
try:
    if int(os.environ['NOGPU']) > 0:
        HAS_DISABLE_GPU=True
except (KeyError,ValueError):
    pass

if HAS_DISABLE_GPU:
    HAS_GPU=False
else:
    try:
        import pycuda.autoinit as cuauto
        import pycuda.driver as cuda
        import pycuda.compiler as cudacc
        import pycuda.gpuarray as gpuarray
        import pycuda.curandom as curandom
        import pycuda.cumath as cumath
        HAS_GPU=True
    except ImportError:
        HAS_GPU=False

OPCODE=[
    "nil", # 0
    "first", # 1
    "+", # 2
    "-", # 3
    "*", # 4
    "/", # 5
    "fmod", # 6
    ">", # 7
    "<", # 8
    ">=", # 9
    "<=", # 10
    "=", # 11
    "!=", # 12
    "if", # 13
    "and", # 14
    "or", # 15
    "not", # 16
    "->", # 17
    "-<", # 18
    "->=", # 19
    "-<=", # 20
    "-=", # 21
    "-!=", # 22
    "-if", # 23
    "-and", # 24
    "-or", # 25
    "-not", # 26
    "sin", # 27
    "cos", # 28
    "tan", # 29
    "asin", # 30
    "acos", # 31
    "atan", # 32
    "sqrt", # 33
    "log", # 34
    "exp", # 35
    "pow", # 36
    "abs", # 37
    "max", # 38
    "min", # 39
    "count", # 40
    "avg", # 41
    "sign", # 42
    "sigmoid", # 43
    "sinh", # 44
    "cosh", # 45
    "tanh", # 46
    "asinh", # 47
    "acosh", # 48
    "atanh", # 49
    "relu", # 50
    "softplus", # 51
    "log1p", # 52
    "expm1", # 53
]

def safe_div(x,y):
    if y == 0.:
        y = 1e-8
    elif np.abs(y) < 1e-8:
        y = 1e-8*np.sign(y)
    return x/y

OPCODE_FUN=[
    lambda v: 0.0,           # nil
    operator.itemgetter(0),  # first
    np.sum,                     # +
    lambda v: v[0]-v[1],     # - 
    np.prod,                 # *
    lambda v: safe_div(v[0],v[1]),     # /
    lambda v: np.fmod(v[0],v[1]), # fmod
    lambda v: 1.0 if v[0] > v[1] else 0.0, # >
    lambda v: 1.0 if v[0] < v[1] else 0.0, # <
    lambda v: 1.0 if v[0] >= v[1] else 0.0, # >=
    lambda v: 1.0 if v[0] <= v[1] else 0.0, # <=
    lambda v: 1.0 if v[0] == v[1] else 0.0, # ==
    lambda v: 1.0 if v[0] != v[1] else 0.0, # !=
    lambda v: v[1] if v[0] != 0 else v[2], # if
    lambda v: 1.0 if v[0] != 0 and v[1] != 0 else 0.0, # and
    lambda v: 1.0 if v[0] != 0 or v[1] != 0 else 0.0, # or
    lambda v: 0.0 if v[0] != 0 else 1.0, # not
    lambda v: 1.0 if v[0] > v[1] else -1.0, # ->
    lambda v: 1.0 if v[0] < v[1] else -1.0, # -<
    lambda v: 1.0 if v[0] >= v[1] else -1.0, # ->=
    lambda v: 1.0 if v[0] <= v[1] else -1.0, # -<=
    lambda v: 1.0 if v[0] == v[1] else -1.0, # -==
    lambda v: 1.0 if v[0] != v[1] else -1.0, # -!=
    lambda v: v[1] if v[0] > 0 else v[2], # -if
    lambda v: 1.0 if v[0] > 0 and v[1] > 0 else -1.0, # -and
    lambda v: 1.0 if v[0] > 0 or v[1] > 0 else -1.0, # -or
    lambda v: -1.0 if v[0] > 0 else 1.0, # -not
    lambda v: np.sin(v[0]), # sin
    lambda v: np.cos(v[0]), # cos
    lambda v: np.tan(v[0]), # tan
    lambda v: np.arcsin(np.fmod(v[0],1.0)), # asin
    lambda v: np.arccos(np.fmod(v[0],1.0)), # acos
    lambda v: np.arctan(v[0]), # atan
    lambda v: np.sqrt(np.abs(v[0])), # sqrt
    lambda v: np.log(np.abs(v[0])), # log
    lambda v: np.exp(v[0]), # exp
    lambda v: np.power(v[0],v[1]), # pow
    lambda v: np.abs(v[0]), # abs
    np.max, # max
    np.min, # min
    len, # count
    np.mean, # avg
    lambda v: np.sign(v[0]), # sign
    lambda v: 1.0/(1.0+np.exp(-v[0])), # sigmoid
    lambda v: np.sinh(v[0]), # sinh
    lambda v: np.cosh(v[0]), # cosh
    lambda v: np.tanh(v[0]), # tanh
    lambda v: np.arcsinh(v[0]), # asinh
    lambda v: np.arccosh(v[0]), # acosh
    lambda v: np.arctanh(v[0]), # atanh
    lambda v: np.maximum(v[0],0.), # relu
    lambda v: np.log1p(np.exp(x)), # softplus
    lambda v: np.log1p(v[0]), # log1p
    lambda v: np.expm1(v[0]), # expm1
]

assert len(OPCODE_FUN) == len(OPCODE), "%d vs %d" % (len(OPCODE_FUN),len(OPCODE))

OPCODE_INVERSE=dict(((x[1],x[0]) for x in enumerate(OPCODE)))

TYPE_NULL=0
TYPE_HEADER=1
TYPE_CHILD=2
TYPE_VALUE=3
TYPE_CONST=4

ERROR_GENERIC=0
ERROR_TEXTFULL=1
ERROR_SYNTAX=2
ERROR_OPCODENOTFOUND=3
ERROR_UNKTYPE=4
ERROR_RUNTIME=5
 
class gpu_lisp_error(Exception):
    def __init__(self,err_type,arg=None):
        self.err = err_type
        if arg is None:
            self.arg = ""
        else:
            self.arg = ": %s" % arg

    def __str__(self):
        return (
            "Generic Error%s",
            "Text Memory Full%s",
            "Variable not found%s",
            "Syntax error%s",
            "Opcode not found%s",
            "Unknown header type%s",
            "Runtime error%s",
            )[self.err] % self.arg

class gpu_lisp_text:
    def __init__(self,max_tree_size=32,max_node_size=8,use_double=False):
        self.max_tree_size = max_tree_size
        self.max_node_size = max_node_size
        self.use_double = use_double

        if use_double:
            self.code_size = 12
        else:
            self.code_size = 8

        self.program_size = self.max_tree_size*self.max_node_size*self.code_size
        self.symbol = None

        self.memory = bytearray(self.program_size)
        self.free_ptr = 0
        
    def clone(self):
        return gpu_lisp_text(self.max_tree_size,self.max_node_size,self.use_double)

    def copy(self):
        n = self.clone()
        n.symbol = self.symbol
        n.memory = copy.copy(self.memory)
        n.free_ptr = self.free_ptr
        return n

    def set_symbol(self,s):
        self.symbol = s

    def zero(self):
        self.memory = bytearray(self.program_size)
        self.free_ptr = 0

    def _compute_treesize(self,bp,nodelen=None):
        l = 0
        s = 0
        p = bp*self.max_node_size
        while True:
            t = self._get_type(p)
            if t == TYPE_NULL:
                break
            elif t == TYPE_CHILD:
                cp = self._get_int(p)
                s += self._compute_treesize(cp,nodelen)
                l += 1
            else:
                l += 1
            p += 1
        if nodelen is not None:
            nodelen[bp] = l
        return s+l

    def tree_size(self,return_map=False):
        if return_map:
            nl = {}
            l = self._compute_treesize(0,nl)
            return (l,nl)
        else:
            return self._compute_treesize(0)
    
    def _get_type(self,p):
        return struct.unpack_from("=L",self.memory,p*self.code_size)[0]

    def _get_int(self,p):
        return struct.unpack_from("=L",self.memory,p*self.code_size+4)[0]

    def _get_float(self,p):
        if self.use_double:
            return struct.unpack_from("=d",self.memory,p*self.code_size+4)[0]
        else:
            return struct.unpack_from("=f",self.memory,p*self.code_size+4)[0]

    def _set_type(self,p,v):
        try:
            struct.pack_into("=L",self.memory,p*self.code_size,v)
        except TypeError:
            print(type(p))
            print(type(self.code_size))
            print(type(v))
            raise
            
    def _set_int(self,p,v):
        struct.pack_into("=L",self.memory,p*self.code_size+4,v)

    def _set_float(self,p,v):
        if self.use_double:
            struct.pack_into("=d",self.memory,p*self.code_size+4,v)
        else:
            struct.pack_into("=f",self.memory,p*self.code_size+4,v)

    def _set_full(self,p,t,v):
        if t == TYPE_VALUE:
            if self.use_double:
                struct.pack_into("=Ld",self.memory,p*self.code_size,t,v)
            else:
                struct.pack_into("=Lf",self.memory,p*self.code_size,t,v)
        else:
            struct.pack_into("=LL",self.memory,p*self.code_size,t,v)

    def is_free(self,p):
        if self._get_type(p*self.max_node_size) == TYPE_NULL:
            return True
        else:
            return False

    def _mark_nonfree(self,p):
        self._set_type(p*self.max_node_size,TYPE_HEADER)

    def _mark_free(self,p):
        self._set_type(p*self.max_node_size,TYPE_NULL)

    def _skip(self,s,p):
        oc = 0
        try:
            while True:
                if s[p] == '(':
                    oc += 1
                elif s[p] == ')':
                    if oc == 0: break
                    oc -= 1
                p += 1
        except IndexError:
            pass
        return p

    def _rlookup(self,p):
        if self.symbol:
            return self.symbol.rlookup(p)
        else:
            return "$%d" % p

    # return pointer of next available object
    def alloc(self):
        p = self.free_ptr
        if self.is_free(p):
            self.free_ptr = (p+1)%self.max_tree_size
            self._mark_nonfree(p)
            return p
        else:
            for i in range(p+1,self.max_tree_size):
                if self.is_free(i):
                    p = i
                    self.free_ptr = (p+1)%self.max_tree_size
                    self._mark_nonfree(p)
                    return p

            for i in range(0,p):
                if self.is_free(i):
                    p = i
                    self.free_ptr = (p+1)%self.max_tree_size
                    self._mark_nonfree(p)
                    return p
            raise gpu_lisp_error(ERROR_TEXTFULL)

    def free(self,p):
        self._mark_free(p)
        if self.free_ptr > p:
            self.free_ptr = p

    def free_node(self,p):
        one = False if self._get_type(p) == TYPE_HEADER else True

        while True:
            t = self._get_type(p)
            if t == TYPE_NULL:
                break
            elif t == TYPE_CHILD:
                self.free_node(self._get_int(p)*self.max_node_size)

            self._set_type(p,TYPE_NULL)

            if one:
                break

            p += 1

        if not one:
            self.free(p // self.max_node_size)

    def copy_node(self,p):
        v = []

        one = False if self._get_type(p) == TYPE_HEADER else True
        
        while True:
            t = self._get_type(p)
            if t == TYPE_NULL:
                break
            elif t == TYPE_HEADER:
                v.append((t,self._get_int(p)))
            elif t == TYPE_CONST:
                v.append((t,self._get_int(p)))
            elif t == TYPE_VALUE:
                v.append((t,self._get_float(p)))
            elif t == TYPE_CHILD:
                v.append((t,self.copy_node(self._get_int(p)*self.max_node_size)))
            else:
                raise gpu_lisp_error(ERROR_UNKHEADER,str(t))

            if one:
                break

            p += 1

        if one:
            return v[0] if len(v) > 0 else None
        else:
            return v

    def replace_node(self,p,data):
        if self._get_type(p) == TYPE_NULL:
            raise gpu_lisp_error(ERROR_GENERIC,"Replacing null node")

        return self._replace_node(p,data)

    def _replace_node(self,p,data):
        t = self._get_type(p)

        if p == 0:
            self.zero()

        if t == TYPE_CHILD:
            self.free_node(self._get_int(p)*self.max_node_size)
        elif t == TYPE_HEADER and (type(data) is list or (type(data) is tuple and data[0] == TYPE_CHILD)):
            assert p == 0, "Replacing a non-root node at TYPE_HEADER, not TYPE_CHILD"

            if type(data) is tuple:
                data = data[1]
                
            bp = self.alloc()
            p = bp*self.max_node_size
            for (i,node) in enumerate(data):
                if i == 0:
                    self._set_full(p,node[0],node[1])
                else:
                    self._replace_node(p,node)
                p += 1
            self._set_type(p,TYPE_NULL)

            return

            # other case should be true header copy to header or pure atom
        if type(data) is list:
            bp = self.alloc()
            self._set_full(p,TYPE_CHILD,bp)
            p = bp*self.max_node_size
            for node in data:
                self._replace_node(p,node)
                p += 1
            self._set_type(p,TYPE_NULL) # make sure there is end
        elif type(data) is tuple:
            (h,v) = data
            if h == TYPE_CHILD:
                self._replace_node(p,v)
            else:
                self._set_full(p,h,v)
        else:
            raise gpu_lisp_error(ERROR_GENERIC,"Unknown data type for replace_node")

    def rparse(self,cbuf,p):
        if cbuf[p] != '(': # treat it as an atom (special case)
            bpptr = self.alloc()
            base = bpptr*self.max_node_size
            if cbuf[p] == '$':
                self._set_full(base,TYPE_CONST,int(cbuf[p+1:]))
            else:
                try:
                    self._set_full(base,TYPE_VALUE,float(cbuf))
                except ValueError:
                    self._set_full(base,TYPE_CONST,self.symbol.lookup(str(cbuf)))
            self._set_type(base+1,TYPE_NULL)
            return

        buf = []
        p += 1
        bpptr = self.alloc()
        bpp = 0
        base = bpptr*self.max_node_size
        eof = False
        while True:
            try:
                ch = cbuf[p]
            except IndexError:
                ch = '\0'

            if bpp >= self.max_node_size:
                raise gpu_lisp_error(ERROR_TEXTFULL,"Maximum node size reached")

            if ch == '(':
                ptr = self.rparse(cbuf,p)
                p = self._skip(cbuf,p+1)
                self._set_full(base + bpp,TYPE_CHILD,ptr)
                bpp += 1
            elif len(buf) > 0 and ch in (' ',')','\n','\0'):
                sbuf = "".join(buf)
                buf = []
                if bpp == 0:
                    try:
                        opcode = OPCODE_INVERSE[sbuf]
                    except KeyError:
                        raise gpu_lisp_error(ERROR_OPCODENOTFOUND,sbuf)
                    self._set_full(base,TYPE_HEADER,opcode)
                else:
                    if sbuf[0] == '$':
                        self._set_full(base + bpp,TYPE_CONST,int(sbuf[1:]))
                    else:
                        try:
                            self._set_full(base + bpp,TYPE_VALUE,float(sbuf))
                        except ValueError:
                            self._set_full(base + bpp,TYPE_CONST,self.symbol.lookup(sbuf))

                bpp += 1
                if ch in (')','\0'):
                    break
            elif ch in (')','\0'):
                break
            else:
                if ch not in (' ','\n'):
                    buf.append(ch)
            p += 1
        self._set_type(base + bpp,TYPE_NULL)
        return bpptr

    def r2text(self,bpp,buf,indent):
        p = bpp
        while True:
            t = self._get_type(p)
            if t == TYPE_NULL:
                break
            elif t == TYPE_HEADER:
                buf.append('(')
                buf.append(OPCODE[self._get_int(p)])
            elif t == TYPE_VALUE:
                buf.append("%g" % self._get_float(p))
            elif t == TYPE_CONST:
                buf.append(self._rlookup(self._get_int(p)))
            elif t == TYPE_CHILD:
                if indent > 0:
                    buf.append("\n")
                    buf.append("  "*indent)
                    self.r2text(self._get_int(p)*self.max_node_size,buf,indent+1)
                else:
                    self.r2text(self._get_int(p)*self.max_node_size,buf,0)
            else:
                raise gpu_lisp_error(ERROR_UNKHEADER,str(t))
            buf.append(' ')
            p += 1

        try:
            buf.pop()
            buf.append(')')
            if indent > 0:
                buf.append("\n")
                buf.append("  "*(indent-2))
        except IndexError:
            pass

    def eval2atom(self,p,const):
        val = []

        bpp = p
        
        t = self._get_type(bpp)
        if t == TYPE_VALUE:
            return self._get_float(bpp)
        elif t == TYPE_CONST:
            return const.data[self._get_int(bpp)]
        else:
            bpp += 1

        while True:
            t = self._get_type(bpp)
            if t == TYPE_NULL:
                break
            elif t == TYPE_VALUE:
                val.append(self._get_float(bpp))
            elif t == TYPE_CONST:
                val.append(const.data[self._get_int(bpp)])
            else:
                raise gpu_lisp_error(ERROR_UNKTYPE,str(t))

            bpp += 1

        opcode = self._get_int(p)
        try:
            fun = OPCODE_FUN[opcode]
        except IndexError:
            raise gpu_lisp_error(ERROR_OPCODENOTFOUND,str(opcode))
    
        try:
            with np.errstate(divide='ignore'):
                return fun(val)
        except IndexError:
            print(self.to_string())
            raise gpu_lisp_error(ERROR_RUNTIME,"Incomplete argument of opcode %d, args=%s" % (opcode,repr(val)))

    def run(self,const):
        btext = self.memory
        self.memory = copy.copy(btext)

        bp_stack = []
        
        base = 0
        current = base
        while True:
            t = self._get_type(current)
            if t == TYPE_CHILD:
                bp_stack.append((base,current))
                base = self._get_int(current) * self.max_node_size
                current = base
            elif t == TYPE_NULL:
                if len(bp_stack) == 0: break
                val = self.eval2atom(base,const)
                (base,current) = bp_stack.pop()
                self._set_full(current,TYPE_VALUE,val)
            elif t in (TYPE_HEADER,TYPE_VALUE,TYPE_CONST):
                pass
            else:
                raise gpu_lisp_error(ERROR_UNKTYPE,str(t))
            current += 1

        val = self.eval2atom(0,const)
        self.memory = btext
        return val

    def to_string(self,indent=False):
        buf = []
        if indent:
            self.r2text(0,buf,1)
        else:
            self.r2text(0,buf,0)
        if buf[0] != '(':
            buf.pop() # pop last )
        return "".join(buf)

    def __str__(self):
        return "<gpu_lisp_text at 0x%x, size=%d, \"%s\">" % (id(self),self.program_size,self.to_string())

class gpu_lisp_symbol:
    def __init__(self,m=None):
        self.var = {}
        self.invvar = {}

        if m is not None:
            for (i,n) in enumerate(m):
                self.assign_name(n,i)

    def __str__(self):
        return "<gpu_lisp_symbol at 0x%x, length=%d>" % (id(self),len(self.var))
    
    def __len__(self):
        return len(self.var)

    def rlookup(self,p):
        try:
            return self.invvar[str(p)]
        except KeyError:
            return "$%d" % p

    def lookup(self,s):
        try:
            return self.var[s]
        except KeyError:
            try:
                return int(s[1:])
            except ValueError:
                raise KeyError

    def reset(self):
        self.var = {}
        self.invvar = {}

    def assign_name(self,name,ptr):
        if name[0] == '$': raise ValueError("All variable name cannot be start with $")
        self.var[name] = ptr
        self.invvar[str(ptr)] = name

    def delete_name(self,name):
        del self.invvar[str(self.var[name])]
        del self.var[name]

class gpu_lisp_constant:
    def __init__(self,max_const_size,use_double):
        self.max_const_size = max_const_size
        self.use_double = use_double

        if self.use_double:
            self.const_size = 8
            self.data = np.empty((self.max_const_size,),dtype=np.float64)
        else:
            self.const_size = 4
            self.data = np.empty((self.max_const_size,),dtype=np.float32)

        self.total_size = max_const_size * self.const_size

        assert self.const_size == self.data.itemsize
        assert self.total_size == self.data.nbytes
        
        self.symbol = None

    def __len__(self):
        return self.max_const_size
    
    def __str__(self):
        return "<gpu_lisp_constant at 0x%x, max_const_size=%d, %s-precision>" % (id(self),self.max_const_size,"double" if self.use_double else "single")

    def set_symbol(self,symbol):
        self.symbol = symbol

    def __setitem__(self,name,val):
        val = float(val)
        try:
            p = int(name)
        except ValueError:
            p = self.symbol.lookup(name)

        self.data[p] = val

    def __getitem__(self,name):
        try:
            p = int(name)
        except ValueError:
            p = self.symbol.lookup(name)

        return self.data[p]

def require_gpu(f):
    if HAS_GPU:
        return f
    else:
        def no_gpu(*a,**ka):
            raise RuntimeError("No GPU")
        return no_gpu

class gpu_lisp:
    def __init__(self,symbol=None,
                 max_tree_size=64,max_node_size=16,max_const_size=16,use_double=False):

        # for max_node_size
        # there is one NULL at the end and there is one OP at the head
        # if max_node_size=4, only (+ $1 $2) is allowed
        # should not be smaller than 3
        assert max_tree_size > 0
        assert max_node_size > 2
        assert max_const_size > 0

        self.max_node_size = max_node_size
        self.max_tree_size = max_tree_size
        self.max_const_size = max_const_size
        self.use_double = use_double

        if use_double:
            self.code_size = 12
            self.atom_size = 8
        else:
            self.code_size = 8
            self.atom_size = 4

        if symbol is None:
            symbol = gpu_lisp_symbol()

        self.symbol = symbol

        self.program_size = self.max_tree_size * self.max_node_size * self.code_size
        self.const_size = self.max_const_size * self.atom_size

        if HAS_GPU:
            self.kernel = None
            self.block_size = 256
            self.compile()
        
    def __str__(self):
        return "<gpu_lisp at 0x%x, max_node_size=%d, max_tree_size=%d, max_const_size=%d, %s-precision>" % (id(self),
                                                                                                self.max_node_size,
                                                                                                self.max_tree_size,
                                                                                                self.max_const_size,
                                                                                                "double" if self.use_double else "single")

    def __getstate__(self):
        tmp = []
        tmp.append(self.max_node_size)
        tmp.append(self.max_tree_size)
        tmp.append(self.max_const_size)
        tmp.append(self.use_double)
        tmp.append(self.symbol)
        return tmp

    def __setstate__(self,state):
        self.max_node_size = state.pop(0)
        self.max_tree_size = state.pop(0)
        self.max_const_size = state.pop(0)
        self.use_double = state.pop(0)
        self.symbol = state.pop(0)

    def create_constant(self,data=None):
        c = gpu_lisp_constant(self.max_const_size,self.use_double)
        c.set_symbol(self.symbol)

        if data is not None:
            if isinstance(data,dict):
                for (k,v) in data.items():
                    c[k] = v
            else:
                c.data[:] = data

        return c

    def parse(self,code,prototype=None):
        if prototype is None:
            text = gpu_lisp_text(self.max_tree_size,self.max_node_size,self.use_double)
        else:
            text = prototype.clone()

        text.set_symbol(self.symbol)
        text.rparse(code,0)
        return text

    @cached
    def smart_split(self,n):
        bs = self.block_size
        grid = int(n/bs)
        if n%bs > 0:
            grid += 1

        return (bs,grid)

    def cpu_map(self,text,const_array):
        dtype = np.float64 if self.use_double else np.float32
        return np.array([text.run(c) for c in const_array],dtype=dtype)

    map=cpu_map

    @require_gpu
    def compile(self):
        if self.kernel is None:
            options = []
            options.append("-DMAX_TREE_SIZE=%d" % self.max_tree_size)
            options.append("-DMAX_NODE_SIZE=%d" % self.max_node_size)
            options.append("-DMAX_CONST_SIZE=%d" % self.max_const_size)
     
            if self.use_double:
                options.append("-DUSE_DOUBLE=1")
     
            (path,dummy) = os.path.split(os.path.realpath(__file__))

            with open(path+"/gpu_lisp.cu","r") as fp:
                code = fp.read()
     
            self.kernel = cudacc.SourceModule(code,options=options)
            gpu_func_map = [
                ("reduce","PPPii"),
            ]
            
            for (gpu_fun,args) in gpu_func_map:
                f = self.kernel.get_function(gpu_fun)
                f.prepare(args)
                setattr(self,"_gpu_" + gpu_fun,f)

    @require_gpu
    def gpu_alloc_constant(self,const_array):
        # FIXME: shall I use np.hstack?
        N = len(const_array)
        cs = self.const_size
        b = bytearray(cs*N)
        for (i,const) in enumerate(const_array):
            b[i*cs:(i+1)*cs] = memoryview(const.data)
        return cuda.to_device(b)

    @require_gpu
    def gpu_alloc_text(self,N):
        return cuda.mem_alloc(self.program_size*N)

    @require_gpu
    def gpu_fill_text(self,ptr,text,N):
        cuda.memcpy_htod(ptr,text.memory*N)

    @require_gpu
    def gpu_fill_text_async(self,ptr,text,N,stream):
        cuda.memcpy_htod_async(ptr,text.memory*N,stream)
            
    @require_gpu
    def gpu_fill_text_array(self,ptr,text_array,N):
        M = len(text_array)
        skip = self.program_size*N
        ptr = int(ptr)
        for k in range(M):
            buf = text_array[k].memory*N
            cuda.memcpy_htod(ptr + k*skip,buf)

    @require_gpu
    def gpu_fill_text_array_async(self,ptr,text_array,N,pagelocked_buf,stream):
        M = len(text_array)
        skip = self.program_size*N
        ptr = int(ptr)
        
        # pagelocked_buf = cuda.pagelocked_empty((N,),dtype="|S%d" % program_size)
        for k in range(M):
            stream.synchronize()
            pagelocked_buf[:] = bytes(text_array[k].memory)
            cuda.memcpy_htod_async(ptr + k*skip,pagelocked_buf,stream)

    @require_gpu
    def gpu_fill_text_array_async2(self,ptr,text_array,N,pagelocked_buf0,stream0,pagelocked_buf1,stream1):
        M = len(text_array)
        skip = self.program_size*N
        ptr = int(ptr)

        assert M%2 == 0
        for k in range(0,M,2):
            stream0.synchronize()
            pagelocked_buf0[:] = bytes(text_array[k].memory)
            cuda.memcpy_htod_async(ptr + k*skip,pagelocked_buf0,stream0)

            k += 1
            
            stream1.synchronize()
            pagelocked_buf1[:] = bytes(text_array[k].memory)
            cuda.memcpy_htod_async(ptr + k*skip,pagelocked_buf1,stream1)
            
    @require_gpu
    def gpu_create_text(self,text,N):
        ptr = self.gpu_alloc_text(N)
        self.gpu_fill_text(ptr,text,N)
        return ptr

    @require_gpu
    def gpu_create_text_array(self,text_array,N):
        M = len(text_array)
        ptr = self.gpu_alloc_text(N*M)
        self.gpu_fill_text_array(ptr,text_array,N)
        return ptr
                
    @require_gpu
    def gpu_alloc_output(self,N):
        if self.use_double:
            out = gpuarray.empty(shape=(N,),dtype=np.float64)
        else:
            out = gpuarray.empty(shape=(N,),dtype=np.float32)
        return out
    
    @require_gpu
    def gpu_raw_map(self,out,text_memory,const_memory,N,M=0):
        (block,grid) = self.smart_split(N)

        self._gpu_reduce.prepared_call(
            (grid,1,1),(block,1,1),
            out.gpudata,
            text_memory,
            const_memory,
            np.int32(N),
            np.int32(M),
        )
            
        return out

    @require_gpu
    def gpu_raw_map_async(self,out,text_memory,const_memory,N,M=0,stream=None):
        (block,grid) = self.smart_split(N)

        self._gpu_reduce.prepared_async_call(
            (grid,1,1),(block,1,1),stream,
            out.gpudata,
            text_memory,
            const_memory,
            np.int32(N),
            np.int32(M),
        )
        
        return out

    @require_gpu
    def gpu_map(self,text,const_array):
        N = len(const_array)
        const_memory = self.gpu_alloc_constant(const_array)
        text_memory = self.gpu_create_text(text,N)
        out = self.gpu_alloc_output(N)
        
        self.gpu_raw_map(out,text_memory,const_memory,N)
        
        ret = out.get()
        const_memory.free()
        text_memory.free()
        del out
        return ret

if __name__ == "__main__":
    pass
