from cmath import log
import timeit
from lib import util
import numpy as np
import time as t
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
def kaczmarz(x_ini, mfolds, num_iter, eps = 0.1):
    logger.info("Starting Kaczmarz projection...")
    # logger.debug(f"Input Length: {x_ini.shape[0]}, # iter: {num_iter}, eps: {eps}")
    t1 = t.time()
    convergence = True
    x_k = x_ini.copy()
    k = 0
    # logger.info(f"manifold shape: {mfolds.m1.y(x_k).shape}")
    # logger.debug(mfolds.m4.y(x_k))
    r = mfolds.y(x_k)
    # print('r:', r)
    
    m = r.shape[0]
    f_ini = -r.copy()
    r_norm = np.linalg.norm(r)
    # print("r_norm: , x_ini: ",r_norm, x_ini['x_1'], x_ini['y_1'])
    curr_iter = 0
    final_x = x_k.copy()
    final_r_norm = r_norm.copy()
    final_r = r.copy()
    timed_residuals = [r.copy().flatten()]
    while ((r_norm > eps) and(curr_iter<num_iter)):
        curr_iter = curr_iter + 1
        # print("r_norm: ", r_norm)
        i = k % m
        # print("i: ",i)
        # print('J of i: ',Js[i,:])
        mfolds.counter = i
        g_i = mfolds.get_sub_j(x_k)
        g_i_norm_sq = np.square(np.linalg.norm(g_i))
        # print("g_i, r_i: ", g_i, r[i], g_i_norm_sq)
        # print(x_k, (r[i][0]/g_i_norm_sq)*g_i)
        x_k = np.round(x_k + ((r[i][0]/g_i_norm_sq)*g_i),2)[0]
        k = k + 1
        r = -mfolds.y(x_k)
        r_norm_old = r_norm
        r_norm = np.linalg.norm(r)
        if r_norm < r_norm_old:
            final_x = x_k.copy()
            final_r_norm = r_norm.copy()
            final_r = r.copy()
        r_norm = np.linalg.norm(r)
        timed_residuals.append(r.copy().flatten())
        # print("k: ", k, r_norm)
    # print("len func: ", len(funcs))
    # print(r_norm)
    if (r_norm>eps):
        convergence = False
    # print('r_norm: %f iter %d' %(r_norm, curr_iter))
    time_taken = t.time()-t1
    logger.info(f"final_r by Kacz: {final_r_norm}")
    return(x_ini, final_x, final_r, f_ini, convergence, time_taken)