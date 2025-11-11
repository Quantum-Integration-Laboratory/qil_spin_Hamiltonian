import numpy as np
from scipy import optimize
import qil_SpinHamiltonian.spin_hamiltonian as spin

def lsMatch(F,uvec,tOptFunc,Bi=1E-3,**kwargs):
    """
    Finds the field (B) at each vector in (uvec) such that the transition (t) is at frequency (F) 

    F: The frequency the transition is to match (GHz)
    uvec: The set of unit direction vectors to match 
    tOptFunc: The function to optimise over this should take a series of magnetic fields and return the transition frequencies at each field
    Bi: An initial field guess this is used for all vectors (T)

    """
    B0=Bi*np.ones(uvec.shape[-1])
    func=lambda Bj: np.abs(tOptFunc(Bj*uvec))-F
    rs=optimize.least_squares(func,B0,**kwargs)
    return rs.x

def transitionOptimiseFunc(S,j,Bs):
    """
    For use with lsMatch, frequency at a set of magnetic field values

        Parameters
        ----------
        S : cSpinHamiltonian
            The system to calculate values on
        j: int
            The index of the transition to be calculated
        Bs: np.ndarray, (3,k)
            The magnetic field values to calculate the frequencies at
        Returns
        -------
        TGs: np.ndarray(1,k)
            The calculated frequencies
    """
    HG = S.dynamicH(Bs)#np.array(S.H)[:,:,np.newaxis]+
    F,V = S.getEigFreq(HG)
    
    TGs=spin.eachElemFunc(F,F)
    return TGs[:,j]

def transitionOptimiseFuncMulti(S,j,Bs):
    """
    For use with lsMatch, frequency at a set of magnetic field values

        Parameters
        ----------
        S : cSpinHamiltonian
            The system to calculate values on
        j: int
            The index of the transition to be calculated
        Bs: np.ndarray, (3,k)
            The magnetic field values to calculate the frequencies at
        Returns
        -------
        TGs: np.ndarray(1,k)
            The calculated frequencies
    """
#    HG = S.dynamicH(Bs)#np.array(S.H)[:,:,np.newaxis]+
    F,TGs = S.getEigFreq(Bs)
    
    # TGs=spin.eachElemFunc(F,F)
    return TGs[:,j]


#returns the index of Zefoz points of a dataset A
def ZEFOZidx(A,ax=0):
    #selects out all zero crossings along the given axis
    idx = zero_crossings(A,0)
    idy = zero_crossings(A,1)
    idz = zero_crossings(A,2)


    Zfx=three_axis_zero(idx)
    Zfy=three_axis_zero(idy)
    Zfz=three_axis_zero(idz)
    ind = np.vstack([Zfx,Zfy,Zfz])
    ind =np.unique(ind[:,0:4],axis=0)
    # idx = idx[np.where(idx[:,4]==0)]
    # idy = idy[np.where(idy[:,4]==1)]
    # idz = idz[np.where(idz[:,4]==2)]
    
    #print(idx.shape,idy.shape)
    #print(idx.shape)
    #ind = np.vstack([idx,idy,idz])
    #print(ind.shape)
    #ind=idz
    
    #groups the ids by the first four indicies, (theta,phi,B,transition),
    #and calculates the sum of the fifth axis (x,y,z), where the +1 accounts for the discrepency of x being missing or a crossing
    #x_u,y_s = group_by(id[:,0:4]).sum(id[:,4]+1)
    
    #print(id)
    #x_u,i_s,y_s=np.unique(ind[:,0:4],axis=0,return_counts=True,return_index=True)
    
    
    #print(id)
    #if the above sum is 6, i.e. zero crossing present in x,y,z, consider this  a ZEFOZ point.
    #zf = np.argwhere(y_s>=3)
    #print(y_s[zf])
    
    #select the remaining indexes where these zefoz point appear
    #ind = x_u[zf[:,0],:]
    return ind.astype(int)

def three_axis_zero(a):
    x_u,i_s,y_s=np.unique(a[:,0:4],axis=0,return_counts=True,return_index=True)
    #if the above sum is 3, i.e. zero crossing present in x,y,z, consider this  a ZEFOZ point.
    zf = np.argwhere(y_s==3)
    #print(y_s[zf].T)
    #select the remaining indexes where these zefoz point appear
    return x_u[zf[:,0],:]


#Locates zero crossings in a along the specified axis ax
def zero_crossings(a,ax):
    ZC=np.argwhere(np.diff(np.sign(a),axis=ax))
    #ZC[:,ax]+=1
    return ZC

def connectedPoints(ZP):
    #get all points that differ from their neighbours by at most one
    spid=np.argwhere(np.abs(np.diff(ZP[:,0:4],axis=0))>1)[:,0].flatten()+1
    spidt=np.argwhere(np.diff(ZP[:,3],axis=0))[:,0].flatten()+1
    #print()
    #print(spid,spidt)
    
    spidx=np.concatenate((spid,spidt))
    #print(spidx)
    CP = np.split(ZP,spidx,axis=0)
    return CP

def connectedRegion(ZP):
    CP=connectedPoints(ZP)
    CR = []

    for r in CP:
        #print(r[:,0],r[:,1],r[:,2])
        xv=([np.min(r[:,0]),np.max(r[:,0])+1])
        yv=([np.min(r[:,1]),np.max(r[:,1])+1])
        zv=([np.min(r[:,2]),np.max(r[:,2])+1])
        CR.append([xv,yv,zv])
    return CR

def nonSymmetricBs(Bmax,pts,iPlane=np.array([1,1,1])):
    """
    Generates a set of points, that span the cube with corners \pm Bmax, 
    but not double counting the inversion symmetry across the plane perpendicular to iPlane
    """
    Bxi=np.linspace(-Bmax,Bmax,pts)
    Byi = np.linspace(-Bmax,Bmax,pts)
    Bzi = np.linspace(-Bmax,Bmax,pts)

    Bsi = np.array(np.meshgrid(Bxi,Byi,Bzi)).reshape(3,-1)
    idx=np.squeeze(np.where(np.dot(iPlane,Bsi)>=0))
    return Bsi[:,idx]