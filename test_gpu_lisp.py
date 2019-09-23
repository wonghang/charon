from charon.gpu_lisp import *

def test_gpu_lisp():
    symbol = gpu_lisp_symbol(["r","n","year","month","day","x","y"])
    l = gpu_lisp(symbol)

    c = l.create_constant({
            "r":10,
            "n":4,
            "year":1986,
            "month":3,
            "day":27,
            "x":5,
            "y":4,
            })

    area = l.parse("(* 3.141592654 (* r r))")
    area2 = l.parse("(+ (* (+ r r) (+ (* (* r 0.559966) (- (+ 0.0930217 0.352407) 0.0850601)) (- r r))) (* (/ r 0.365197) (+ (* (+ (* r (- 0.443465 0.0850601)) (- r r)) (- r r)) r)))")
    stirling = l.parse("(* (sqrt (* 2 3.141592654 n)) (pow (/ n (exp 1)) n) (+ 1 (/ 1 (* 12 n))))")
    pi = l.parse("(* 4 (- (atan (/ 10822096 8641597)) (+ (atan (/ 1 year)) (* month (atan (/ 1 day))))))")
    test_max = l.parse("(sign -1.23)")

    debug = l.parse("(+ x (* (* x x) (sign x)))")

    print(debug.tree_size())
    print(debug.copy_node(2))
    print(debug.to_string())
    debug.replace_node(1,debug.copy_node(2))
    print(debug.to_string())
    print(debug.run(c))

    print(area.to_string(), area.run(c))     # compute area of circle of radius 10
    print(area2.run(c))
    print(stirling.to_string(), stirling.run(c)) # approximate 4!
    print(pi.to_string(), pi.run(c))       # compute pi from my birthday
    print(test_max.to_string(), test_max.run(c) , l.map(test_max,[c]))
    print(debug.to_string(), debug.run(c))

    symbol = gpu_lisp_symbol(["n"])
    l = gpu_lisp(symbol,32,16,1)
    stirling = l.parse("(* (sqrt (* 2 3.141592654 n)) (pow (/ n (exp 1)) n) (+ 1 (/ 1 (* 12 n))))")

    carray = [l.create_constant({"n":n}) for n in range(2,10)]
    ret = l.map(stirling,carray)
    print("CPU",ret)
    for i in range(len(ret)):
        print("%d! ~= %f" % (i+2,ret[i]))

    ret = l.gpu_map(stirling,carray)
    print("GPU",ret)
    for i in range(len(ret)):
        print("%d! ~= %f" % (i+2,ret[i]))

if __name__ == "__main__":
    test_gpu_lisp()
    
