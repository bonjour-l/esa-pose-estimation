import cv2
import numpy as np

import scipy
import torch
from scipy.optimize import leastsq
import math
K= [ [572.4114, 0.,25.2611  ] ,
     [  0., 573.57043,242.04899  ],
     [  0. ,   0.  ,   1.     ]   ]










def rotate(point,R):
    point=np.array(point)
    theta=np.linalg.norm(x=R, ord=2)
    r=R/theta
    cos=math.cos(theta)
    sin=math.sin(theta)
    pp0=    cos*point
    pp1=       (1-math.cos(theta))*(np.dot(r,point))*r
    pp2=     math.sin(theta)*np.cross(r,point)
    return pp0+pp1+pp2

def func(T,points_3d):
    return 0

def error(T,points_3d, points_2d):
    error=0
    R=T[0:3]    
    t=T[2:5]
    for i in range(len(points_2d)) :
        pp=rotate(points_3d[i],R) +t
        pp2=[K[0][0]*pp[0]+K[0][2]*pp[2], K[1][1]*pp[1]+K[1][2]*pp[2] ]
        pp2=np.array(pp2)

        error+=np.linalg.norm((np.array(points_2d[i]-pp2)),ord=2)
    return error
def pnp(points_3d, points_2d, camera_matrix,method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    '''
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)

    '''
    _, R_exp, t, inliers = cv2.solvePnPRansac(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                                reprojectionError=5.0,
                               flags=cv2.SOLVEPNP_EPNP)
    #print(len(inliers))
    #, None, None, False, cv2.SOLVEPNP_UPNP)

    '''
    _,R_exp, t, _ = cv2.solvePnPRansac(points_3d,
                                     points_2d,
                                     camera_matrix,
                                     dist_coeffs,
                                     reprojectionError=12)
    '''
    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)


def opt_pnp(p_3d, p_2d):
    pass


if __name__ == '__main__':
    p_2d= np.array(   [[346.20907974,167.12133026],
               [344.96317673,189.74265289],
               [356.21628952,173.68662643],
               [346.52748871,173.284832  ],
               [374.96802902 ,161.65159988],
               [378.33327484, 178.10058594],
               [371.23065948, 158.74113464],
               [361.15768051, 174.64339066],
               [364.5448494 , 172.90166473],
               [373.35230255, 156.06298065],
               [363.8555603 , 168.71496201],
               [366.36207962, 168.34736252],
               [367.14471054, 150.37490082],
               [383.25449371, 166.39247513],
               [358.56852341, 162.73097229],
               [371.12140274, 141.66212845],
               [332.2505188 , 158.64153671],
               [374.36946106, 160.80280304],
               [371.77785492, 131.71759987],
               [370.20325851, 138.51457405],
               [371.81555176, 173.17403793],
               [374.44326401, 185.1792984 ],
               [375.32796097, 162.33521652],
               [376.07334518, 177.63427353],
               [391.91223907, 149.51176071],
               [374.07404709, 146.90738678],
               [382.90458679, 164.87368393],
               [379.25123596, 187.12863159],
               [369.6604805 , 133.28375816]]  )
    p_3d= np.array(  [[-0.0292035  ,-0.033511   ,-0.03553401],
             [ 0.028648   ,-0.0427175  ,-0.0356155 ],
             [ 0.000261   ,-0.016915   ,-0.028158  ],
             [-0.001742   ,-0.03418887 ,-0.02508313],
             [-0.019613   , 0.019865   ,-0.025657  ],
             [ 0.02185908 , 0.02036446 ,-0.02723742],
             [-0.0209338  , 0.01408971 ,-0.01809472],
             [ 0.02006    ,-0.008399   ,-0.013515  ],
             [ 0.02118557 ,-0.00206228 ,-0.00988687],
             [-0.01394245 , 0.01814227 ,-0.00600043],
             [ 0.01894356 ,-0.00215656 ,-0.00206881],
             [ 0.01930961 , 0.00214367 ,-0.00204769],
             [-0.0189505  , 0.00915689 , 0.00541081],
             [ 0.027923   , 0.030979   , 0.004781  ],
             [ 0.0125355  ,-0.0095925  , 0.008302  ],
             [-0.029583   , 0.01823279 , 0.01406689],
             [ 0.00198108 ,-0.05407578 , 0.01763433],
             [ 0.02861866 , 0.01780268 , 0.02149559],
             [-0.02972148 , 0.02175988 , 0.03803323],
             [ 0.01194836 , 0.01662582 , 0.05726228],
             [-0.01167771 , 0.01177814 ,-0.04400902],
             [ 0.01998289 , 0.01185412 ,-0.04508587],
             [-0.01234844 , 0.02019731 ,-0.02026919],
             [ 0.02483033 , 0.016348   ,-0.0222045 ],
             [-0.00393353 , 0.05216144 , 0.01184427],
             [ 0.0206029  , 0.02030482 , 0.04462871],
             [-0.02226678 , 0.03309886 ,-0.0389219 ],
             [ 0.03236124 , 0.01912243 ,-0.03999504],
             [ 0.00925129 , 0.01686898 , 0.06688058]]        )

    Xi = np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])

    Yi = np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])
    opt_pnp(p_3d,p_2d)








































































