import numpy as np

intrinsic_mtx_1 = np.array([[190.26394677, 0., 116.01783456],
                            [0., 189.83058829, 67.42833983],
                            [0., 0., 1.]])

dist_1 = np.array([[-0.01358666, 0.24676366, -0.00165968, -0.00347389, -0.4627277]])

intrinsic_mtx_2 = np.array([[190.88949482, 0., 116.87796692],
                            [0., 190.59717944, 68.38414401],
                            [0., 0., 1.]])

dist_2 = np.array([[-0.02108141, 0.35510424, 0.00096328, -0.00348292, -0.71272242]])

R = np.array([[9.99995260e-01, -7.73324202e-04, 2.98013373e-03],
              [8.04332228e-04, 9.99945410e-01, -1.04177977e-02],
              [-2.97191471e-03, 1.04201453e-02, 9.99941292e-01]])

T = np.array([[-2.70459062],
              [-0.00545933],
              [0.07690288]])

E = np.array([[-4.56307908e-05, -7.69555655e-02, -4.65785364e-03],
              [6.88646995e-02, 2.81227565e-02, 2.70466102e+00],
              [3.28391748e-03, -2.70444720e+00, 2.81921475e-02]])

F = np.array([[-5.93666502e-08, -1.00349426e-04, 5.62028918e-03],
              [8.97318795e-05, 3.67280864e-05, 6.57644900e-01],
              [-5.31373368e-03, -6.63969473e-01, 1.00000000e+00]])