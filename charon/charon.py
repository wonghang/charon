#!/usr/bin/python3
import time
import struct
import copy,operator
import numpy as np

from .gpu_lisp import *
from .util import *

__all__ = ["gp_config","gp_chromosome","gp_pool","batch_generator"]

@singleton
class gp_config:
    def __init__(self):
        self.reset()

    def reset(self):
        self.terminal_node = []
        self.function_node = []

        # the remaining would be reproduction
        self.crossover_prob = 0.95
        self.mutation_prob = 0.0
        self.elite_prob = 0.05
        self.over_select = 8

        self.mutation_depth = 2

    def set_probability(self,cprob,mprob,eprob):
        if cprob+mprob+eprob > 1.0:
            raise ValueError("sum of all probability should not greater than 1")

        self.crossover_prob = cprob
        self.mutation_prob = mprob
        self.elite_prob = eprob

    def set_mutation_depth(self,depth):
        self.mutation_depth = depth+1

    # [(type,value)], type should be
    def add_terminal_const(self,v):
        self.terminal_node.append((TYPE_CONST,v))

    def add_terminal_range(self,r1,r2):
        self.terminal_node.append((TYPE_VALUE,r1,r2))

    def add_terminal_single(self,x):
        self.terminal_node.append((TYPE_VALUE,x))

    def add_function_node(self,funname,min_node,max_node=None):
        opcode = OPCODE_INVERSE[funname]

        if max_node is None:
            max_node = min_node
        self.function_node.append((opcode,min_node,max_node))

    # WHAT IS THAT?
    def log_apriori_probability_of(self,nodelen):
        return -nodelen * np.log(len(self.terminal_node) + len(self.function_node))

    def random_terminal(self):
        l = len(self.terminal_node)
        t = self.terminal_node[np.random.randint(0,l)]
        if t[0] == TYPE_VALUE:
            if len(t) == 2:
                t = (TYPE_VALUE,float(t[1]))
            elif len(t) == 3:
                if isinstance(t[1],int) and isinstance(t[2],int):
                    t = (TYPE_VALUE,float(np.random.randint(t[1],t[2]+1)))
                else:
                    t = (TYPE_VALUE,np.random.uniform(t[1],t[2]))
        return t

    def random_function(self):
        l = len(self.function_node)
        return self.function_node[np.random.randint(0,l)]

    def random_node(self):
        tl = len(self.terminal_node) 
        fl = len(self.function_node)

        n = np.random.randint(0,2)
        if n == 0:
            return (True,self.function_node[np.random.randint(fl)])
        else:
            return (False,self.terminal_node[np.random.randint(tl)])

class gp_chromosome(gpu_lisp_text):
    def __init__(self,*a,**ka):
        super().__init__(*a,**ka)

        self.invalidate_cache()

    def clone(self):
        return gp_chromosome(self.max_tree_size,self.max_node_size,self.use_double)

    def __str__(self):
        return "<gp_chromosome at 0x%x, %s-precision, \"%s\">" % (
            id(self),
            "double" if self.use_double else "single",
            self.to_string()
            )

    def check_config(self):
        cfg = gp_config()

        if len(cfg.terminal_node) == 0 or len(cfg.function_node) == 0:
            raise ValueError("Number of terminal/function cannot be zero")
         
        for node in cfg.function_node:
            if node[1] > node[2]:
                raise ValueError("Function node %s contains invalid node number" % repr(node))

            if node[2] >= self.max_node_size-1:
                raise ValueError("Function node %s has length greater than %d allowed" % (repr(node),self.max_node_size-1))

    def invalidate_cache(self):
        self._fitness = None
        self._complexity = None
        
    def random_grow(self,max_depth,p=0,depth=0):
        cfg = gp_config()

        if depth < max_depth:
            if depth == 0:
                (funp,node) = (True,cfg.random_function())
            else:
                (funp,node) = cfg.random_node()

            if funp:
                try:
                    bp = self.alloc()
                except gpu_lisp_error as e:
                    if e.err == ERROR_TEXTFULL and depth > 0:
                        (funp,node) = (False,cfg.random_terminal())
                    else:
                        raise

            if funp:
                self._set_full(p,TYPE_CHILD,bp)
                p = bp*self.max_node_size
                self._set_full(p,TYPE_HEADER,node[0])
            
                p += 1
                l = np.random.randint(node[1],node[2]+1)
                for i in range(l):
                    self.random_grow(max_depth,p,depth+1)
                    p += 1
            else:
                self._set_full(p,node[0],node[1])
        else:
            node = cfg.random_terminal()
            self._set_full(p,node[0],node[1])

    def random_full(self,max_depth,p=0,depth=0):
        cfg = gp_config()

        T = False
        if depth < max_depth:
            try:
                bp = self.alloc()
                T = True
            except gpu_lisp_error as e:
                if e.err == ERROR_TEXTFULL:
                    T = False

        if T:
            node = cfg.random_function()
            self._set_full(p,TYPE_CHILD,bp)
            p = bp*self.max_node_size
            self._set_full(p,TYPE_HEADER,node[0])
            
            p += 1
            l = np.random.randint(node[1],node[2]+1)
            for i in range(l):
                self.random_full(max_depth,p,depth+1)
                p += 1
        else:
            node = cfg.random_terminal()
            self._set_full(p,node[0],node[1])

    def random_node(self):
        cmpx,nl = self.tree_size(return_map=True)
        
        k = list(nl.keys())

        bp = k[np.random.randint(0,len(k))]
        nl = nl[bp]
        bp = int(bp)

        if bp == 0:
            p = np.random.randint(0,nl)
        else:
            p = np.random.randint(1,nl)

        return bp*self.max_node_size + p

    # return two gp_chromosome with parent self and c
    def crossover(self,c):
        g1 = self.copy()
        g2 = c.copy()

        p1 = g1.random_node()
        p2 = g2.random_node()

        tmp1 = g1.copy_node(p1)
        tmp2 = g2.copy_node(p2)
        try:
            g1.replace_node(p1,tmp2)
        except gpu_lisp_error as e:
            if e.err == ERROR_TEXTFULL:
                g1 = self
            else:
                raise

        try:
            g2.replace_node(p2,tmp1)
        except gpu_lisp_error as e:
            if e.err == ERROR_TEXTFULL:
                g2 = c
            else:
                raise

        g1.invalidate_cache()
        g2.invalidate_cache()
        return (g1,g2)

    def infection(self,c):
        g1 = self.copy()

        p1 = g1.random_node()
        p2 = c.random_node()

        tmp = c.copy_node(p2)
        try:
            g1.replace_node(p1,tmp)
        except gpu_lisp_error as e:
            if e.err == ERROR_TEXTFULL:
                g1 = self
            else:
                raise

        g1.invalidate_cache()
        return g1

    def mutation(self):
        cfg = gp_config()
        r = self.copy()
        r.invalidate_cache()

        p = r.random_node()
        r.free_node(p)
        r.random_grow(cfg.mutation_depth,p,1)
        r.invalidate_cache()
        return r

    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = self._compute_fitness()
        return self._fitness

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = self._compute_complexity()
        return self._complexity
        
    def _compute_fitness(self):
        raise NotImplementedError("You should implement this")

    def _compute_complexity(self):
        return 0.
    
class gp_pool:
    def __init__(self,prototype,cp=0.,M=500):
        self.M = M
        self.cp = cp
        
        self.prototype = prototype
        self.dtype = np.float64 if prototype.use_double else np.float32
        self.pool = [None for x in range(M)]

        self._fitness = None
        self._index = None
        self._complexity = None
        self._dirty = True
        
        self.gcount = 0

        prototype.check_config()

    def __len__(self):
        return self.M
    
    def compute_all(self):
        if self._dirty:
            fitness = np.array([x.fitness for x in self.pool],dtype=self.dtype)
            if self.cp > 0:
                complexity = np.array([x.complexity for x in self.pool],dtype=self.dtype)
                fitness += self.cp*complexity
            else:
                complexity = None
                self._complexity = complexity
            self._fitness = fitness 
            self._index = np.argsort(self._fitness)
            self._dirty = False

    @property
    def all_index(self):
        self.compute_all()
        return self._index
    
    @property
    def all_fitness(self):
        self.compute_all()
        return self._fitness

    @property
    def all_complexity(self):
        self.compute_all()
        return self._complexity

    @property
    def best(self,force=False):
        self.compute_all()
        return self.pool[self._index[0]]

    def invalidate_cache(self):
        for x in self.pool:
            if x is not None:
                x.invalidate_cache()
        self._dirty = True

    # ramped half-and-half method
    def generate_initial(self,max_depth):
        n = 0
        prototype = self.prototype

        pool = self.pool

        for n in range(0,self.M):
            c = prototype.clone()
            depth = n%(max_depth-1)+1
            if n%2 == 0:
                c.random_grow(depth)
            else:
                c.random_full(depth)
            pool[n] = c

        self.gcount = 0

    def generate_offspring(self):
        M = self.M
        cfg = gp_config()
        pool = self.pool

        new_pool = [None]*M
        
        self.compute_all()
        fitness = self._fitness
        index = self._index

        crossover_count = int(M*cfg.crossover_prob)
        crossover_count -= crossover_count%2
        assert crossover_count%2 == 0
        mutation_count = int(M*cfg.mutation_prob)
        elite_count = int(M*cfg.elite_prob)

        reproduction_count = M - crossover_count - mutation_count - elite_count
        assert reproduction_count >= 0
        assert crossover_count >= 0
        assert mutation_count >= 0
        assert elite_count >= 0
        oc = cfg.over_select

        k = 0
        # reproduction
        for i in range(reproduction_count):
            p1 = np.random.randint(0,M)
            p2 = np.random.randint(0,M)

            c1 = pool[p1]
            c2 = pool[p2]

            if fitness[p2] < fitness[p1]:
                c1 = c2

            new_pool[k] = c1
            k += 1

        # elite
        for i in range(elite_count):
            new_pool[k] = pool[index[i]]
            k += 1

        # crossover
        for i in range(crossover_count//2):
            if np.random.randint(0,10) < oc:
                cc1 = np.random.randint(0,crossover_count)
                cc2 = np.random.randint(0,crossover_count)
                p1 = index[cc1]
                p3 = index[cc2]
                c1 = pool[p1]
                c3 = pool[p3]
            else:
                p1 = np.random.randint(0,M)
                p2 = np.random.randint(0,M)
                p3 = np.random.randint(0,M)
                p4 = np.random.randint(0,M)

                c1 = pool[p1]
                c2 = pool[p2]
                c3 = pool[p3]
                c4 = pool[p4]

                if fitness[p2] < fitness[p1]:
                    c1 = c2

                if fitness[p4] < fitness[p3]:
                    c3 = c4

            (c1,c3) = c1.crossover(c3)
            new_pool[k] = c1
            new_pool[k+1] = c3
            k += 2

        # mutation
        for i in range(mutation_count):
            p1 = np.random.randint(0,M)
            c1 = pool[p1]
            new_pool[k] = c1.mutation()
            k += 1

        self.gcount += 1
        self.pool = new_pool
        self._dirty = True

    def dump(self,fp,fitness=True):
        p = self.pool
        if fitness:
            self.compute_all()
            index = self._index
            fitness2 = self._fitness
            for i in range(self.M):
                j = index[i]
                fp.write("[%d] %g %g: %s\n" % (i,fitness2[j],p[j].fitness,p[j].to_string()))
        else:
            for i in range(self.M):
                fp.write("[%d]: %s\n" % (i,p[i].to_string()))

class batch_generator:
    def __init__(self,batch_size,total_size,shuffle=False):
        self.total_size = total_size
        self.batch_size = batch_size
        self.offset = 0
        if shuffle:
            self.shuffle = np.random.choice(total_size,total_size,False)
        else:
            self.shuffle = None

    def generate(self):
        BS = self.batch_size
        batch = np.empty((BS,),dtype=int)
        m = 0
        M = self.total_size
        offset = self.offset
        while m < BS:
            n = min(BS - m,M - offset)
            batch[m:m+n] = range(offset,offset+n)
            offset += n
            m += n
            if offset == M:
                offset = 0
        self.offset = offset
        if self.shuffle is None:
            return batch
        else:
            return self.shuffle[batch]

    def __iter__(self):
        return self

    __next__=generate
    
if __name__ == "__main__":
    pass
