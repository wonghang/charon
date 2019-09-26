# charon
charon: Symbolic regression accelerated with CUDA

It is my implementation of symbolic regression (genetic programming) using pycuda.
It contains a S-expression interpreter written in both python and CUDA.

Although the code runs without a GPU, it will be much faster if you have a GPU and [pycuda](https://documen.tician.de/pycuda/) installed.

To test the interpreter, run

```
$ python3 test_lisp.py
11
(2, [(1, 4), (2, [(1, 4), (4, 5), (4, 5)]), (2, [(1, 42), (4, 5)])])
(+ x (* (* x x) (sign x)))
(+ (* (* x x) (sign x)) (* (* x x) (sign x)))
50.0
(* 3.14159 (* r r)) 314.1592741012573
314.18363189697266
(* (sqrt (* 2 3.14159 n)) (pow (/ n (exp 1)) n) (+ 1 (/ 1 (* 12 n)))) 23.99589109210907
(* 4 (- (atan (/ 1.08221e+07 8.6416e+06)) (+ (atan (/ 1 year)) (* month (atan (/ 1 day)))))) 3.1415927410125732
(sign -1.23) -1.0 [-1.]
(+ (* (* x x) (sign x)) (* (* x x) (sign x))) 50.0
CPU [1.9989630e+00 5.9983282e+00 2.3995892e+01 1.1998620e+02 7.1994073e+02
 5.0396875e+03 4.0318051e+04 3.6286594e+05]
2! ~= 1.998963
3! ~= 5.998328
4! ~= 23.995892
5! ~= 119.986198
6! ~= 719.940735
7! ~= 5039.687500
8! ~= 40318.050781
9! ~= 362865.937500
GPU [1.9989630e+00 5.9983282e+00 2.3995890e+01 1.1998619e+02 7.1994073e+02
 5.0396875e+03 4.0318055e+04 3.6286594e+05]
2! ~= 1.998963
3! ~= 5.998328
4! ~= 23.995890
5! ~= 119.986191
6! ~= 719.940735
7! ~= 5039.687500
8! ~= 40318.054688
9! ~= 362865.937500
```

There is an example file `test_charon.py`, which attempts to find out the formula of a disc (i.e. pi r ^2) by computing 100  generations of S-expression.

```
$ python3 test_charon.py
[0] Best: fitness=4.04277, complexity=45, code="(+ (* (/ (* r -0.357873) (+ r 0.836114)) (/ (- r r) (/ r 0.315958))) (* (+ (/ r 0.352006) (- r r)) (- (+ r -0.507397) (+ 0.364495 -0.318055))))"
[1] Best: fitness=4.04277, complexity=45, code="(+ (* (/ (* r -0.357873) (- r -0.916931)) (/ (- r r) (/ r 0.315958))) (* (+ (/ r 0.352006) (- r r)) (- (+ r -0.507397) (+ 0.364495 -0.318055))))"
[2] Best: fitness=4.04277, complexity=42, code="(+ (* (/ (* r -0.357873) (+ r 0.836114)) (/ (- r r) 0.0750157)) (* (+ (/ r 0.352006) (- r r)) (- (+ r -0.507397) (+ 0.364495 -0.318055))))"
[3] Best: fitness=3.1189, complexity=48, code="(+ (* (/ (* r -0.357873) (+ r 0.836114)) (/ (- r r) (/ r 0.315958))) (* (+ (/ r 0.352006) (- r r)) (- (+ r (- 0.15757 0.471865)) (+ 0.364495 -0.318055))))"
[4] Best: fitness=1.68051, complexity=24, code="(+ (* (+ (- r 0.734239) (+ r r)) (+ (- r -0.744209) (+ -0.138654 -0.864726))) r)"
[5] Best: fitness=1.68051, complexity=24, code="(+ (* (+ (- r 0.734239) (+ r r)) (+ (- r -0.744209) (+ -0.138654 -0.864726))) r)"
[6] Best: fitness=1.2707, complexity=33, code="(- (/ (* r (* (/ r -1) (- (+ (- (* r 0.00560939) (+ r r)) 0.855362) r))) r) (- -0.595815 r))"
[7] Best: fitness=1.2707, complexity=33, code="(- (/ (* r (* (/ r -1) (- (+ (- (* r 0.00560939) (+ r r)) 0.855362) r))) r) (- -0.595815 r))"
[8] Best: fitness=1.2707, complexity=33, code="(- (/ (* r (* (/ r -1) (- (+ (- (* r 0.00560939) (+ r r)) 0.855362) r))) r) (- -0.595815 r))"
[9] Best: fitness=1.25019, complexity=27, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- (- r (+ r -0.964012)) (* 0.262096 r)))"
[10] Best: fitness=1.25019, complexity=27, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- (- r (+ r -0.964012)) (* 0.262096 r)))"
[11] Best: fitness=0.789119, complexity=18, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- r r))"
[12] Best: fitness=0.789119, complexity=18, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- r r))"
[13] Best: fitness=0.789119, complexity=18, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- r r))"
[14] Best: fitness=0.789119, complexity=18, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- r r))"
[15] Best: fitness=0.789119, complexity=18, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- r r))"
[16] Best: fitness=0.789119, complexity=18, code="(+ (* (+ r (+ (/ r 0.882679) r)) r) (- r r))"
[17] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[18] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[19] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[20] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[21] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[22] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[23] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[24] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[25] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[26] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[27] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[28] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[29] Best: fitness=0.789119, complexity=12, code="(* (+ (/ r 0.882679) (+ r r)) r)"
[30] Best: fitness=0.789119, complexity=12, code="(* (+ (/ r 0.882679) (+ r r)) r)"
[31] Best: fitness=0.789119, complexity=12, code="(* (+ (/ r 0.882679) (+ r r)) r)"
[32] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[33] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[34] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[35] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[36] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[37] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[38] Best: fitness=0.789119, complexity=12, code="(* (+ (+ (/ r 0.882679) r) r) r)"
[39] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[40] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[41] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[42] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[43] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[44] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[45] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[46] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[47] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[48] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[49] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[50] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[51] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[52] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[53] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[54] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[55] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[56] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[57] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ r (/ r 0.882679))) r)"
[58] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[59] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[60] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[61] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[62] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[63] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[64] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[65] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[66] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[67] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[68] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[69] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[70] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[71] Best: fitness=0.789119, complexity=12, code="(* (+ (/ r 0.882679) (+ r r)) r)"
[72] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[73] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[74] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[75] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[76] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[77] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ r (/ r 0.882679))) r)"
[78] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[79] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[80] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[81] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[82] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[83] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[84] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[85] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[86] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[87] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[88] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[89] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[90] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[91] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[92] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[93] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[94] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[95] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[96] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[97] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[98] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
[99] Best: fitness=0.789119, complexity=12, code="(* (+ r (+ (/ r 0.882679) r)) r)"
```

After simplification, the formula is 3.1329146835 r^2. 

The following configuration were used in the above example:
```
Operators allowed: +, -, *, /
Terminal nodes  allowed: real numbers uniformly drawn from [-1,1)  and the radius r of the data
Crossover: 0.8
Mutation: 0.15
Elite: 0.01
Reproduction: 0.04
```

Moreover, a penalty term `0.001` on complexity was used. Read the code for the details.

```python
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
```
