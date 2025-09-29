import numpy as np
from scipy import optimize

def lsMatch(F,uvec,tOptFunc,Bi=1E-3):
    """
    Finds the field (B) at each vector in (uvec) such that the transition (t) is at frequency (F) 

    F: The frequency the transition is to match (GHz)
    uvec: The set of unit direction vectors to match 
    tOptFunc: The function to optimise over this should take a series of magnetic fields and return the transition frequencies at each field
    Bi: An initial field guess this is used for all vectors (T)

    """
    B0=Bi*np.ones(uvec.shape[-1])
    func=lambda Bj: np.abs(tOptFunc(Bj*uvec))-F
    rs=optimize.least_squares(func,B0)
    return rs.x
