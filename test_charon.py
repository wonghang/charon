import numpy as np
from charon import *

def singleton(cls):
    instance_container = []
    def getinstance():
        if not len(instance_container):
            instance_container.append(cls())
        return instance_container[0]
    return getinstance

@singleton
class area_of_disk_shared:
    def __init__(self):
        symbol = gpu_lisp.gpu_lisp_symbol(["r"])
        lisp = gpu_lisp.gpu_lisp(symbol,max_tree_size=32,max_node_size=8,max_const_size=1,use_double=False)

        data_size = 1000
        
        tmp = np.linspace(-5,5,data_size)
        tmp1 = [lisp.create_constant({"r":tmp[k]}) for k in range(data_size)]
        tmp = np.pi*tmp*tmp
        tmp += np.random.normal(size=tmp.shape)
        dataset = [tmp1,tmp]
            
        self.lisp = lisp
        self.dataset = dataset
        self.symbol = symbol
        
class area_of_disk(gp_chromosome):
    def __init__(self):
        shared = area_of_disk_shared()
        lisp = shared.lisp

        super().__init__(max_tree_size=lisp.max_tree_size,
                         max_node_size=lisp.max_node_size,
                         use_double=lisp.use_double)
        self.set_symbol(shared.symbol)

    def clone(self):
        return area_of_disk()

    def _compute_fitness(self):
        shared = area_of_disk_shared()
        lisp = shared.lisp
        (X,Y) = shared.dataset

        if charon.HAS_GPU:
            y_cup = lisp.gpu_map(self,X)
        else:
            y_cup = lisp.map(self,X)
        err = np.mean(np.abs(y_cup - Y))
        return err

    def _compute_complexity(self):
        return self.tree_size()
    
def main():
    cfg = gp_config()

    cfg.add_terminal_const(0)
    cfg.add_terminal_range(-1.0,1.0)
    cfg.add_function_node("+",2,2)
    cfg.add_function_node("-",2,2)
    cfg.add_function_node("*",2,2)
    cfg.add_function_node("/",2,2)

    cfg.set_probability(0.80,0.15,0.01)
    shared = area_of_disk_shared()
    pool = gp_pool(area_of_disk(),cp=0.001,M=1000)
    pool.generate_initial(5)

    try:
        for n in range(100):
            best = pool.best
            print("[%d] Best: fitness=%g, complexity=%g, code=\"%s\"" % (n,best.fitness,best.complexity,best.to_string()))
            pool.generate_offspring()
    except KeyboardInterrupt:
        print("KeyboardInterrupted. Dumping the pool...")
        fitness = pool.all_fitness
        idx = pool.all_index
        for i in range(len(pool)):
            j = idx[i]
            print("%d/%f: %s" % (i,fitness[j],pool.pool[j].to_string()))
    
if __name__ == "__main__":
    np.random.seed(12345)
    main()
   
