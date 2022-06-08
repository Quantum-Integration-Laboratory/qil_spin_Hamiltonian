import numpy as np

from numpy.core.shape_base import hstack
from scipy.spatial.transform import Rotation


muB = 9.27400968e-24 #Bohr Magneton (Am^2)
muN = 5.05078375e-27 #Nuclear Magneton (Am^2)
hbar = 1.055E-34 #Reduced planck constant (Js)
h = 2*np.pi*hbar #Planck constant (Js)

NoneType=type(None)

#class containing many parameters for a spin hamiltonian. Ideally it contains H a static hamiltoninan
#and functions to calculate dynamic hamiltonian parts based on provided values.
class cSpinHamiltonian:
    def __init__(self,E,I):
        #Sets up electronic and nuclear spin operators and initialises empty static hamiltonian
        self.Edim = int(2*E+1)
        self.Idim = int(2*I+1)

        self.S = spinOperator(E)
        self.I = spinOperator(I)

        self.H =None
    #Calculates the hyperfine interaction hamiltonian IAS, and adds it to the static hamiltonian
    def hyperfineInteraction(self,A):
        #calculate the hyperfine interaction 
        self.HHF = (self.I)@A@(self.S).T
        #reshape the hamiltonian to maintain dimension
        self.HHF = self.HyperfineReshape()

        #if this is the first static parameter we calculate set the hamiltonian to it,
        #otherwise add it to the hamiltonian
        if type(self.H) is NoneType:
            self.H=self.HHF
        else:
            self.H+=self.HHF
        return self.HHF

    #reshape the hamiltonian to maintain to be Edim*Idim by Edim*Idim
    def HyperfineReshape(self,H=None):
        Idim = self.Idim
        if type(H) is NoneType:
            H = self.HHF
        a=H[:,0].reshape((Idim,Idim))
        b=H[:,1].reshape((Idim,Idim))
        c=H[:,2].reshape((Idim,Idim))
        d=H[:,3].reshape((Idim,Idim))
        return np.block([[a,b.T],[c.T,d]])


    #setters for our g paramaters
    def setgE(self,g):
        self.gE= g
    def setgN(self,g):
        self.gN= g

    #Calculate the zeeman interaction of the form muBgS, allowing for the same function to do nuclear and electronic
    def zeemanInteraction(self,mu,B,g,S,dim):
        HZ = mu*B.T@g@S.T
        #reshape to be Idim*Edim square matrix
        HZ = HZ.T.reshape(dim,dim)
        HZ = np.kron(HZ,np.eye((2*self.Idim)//dim))
        return HZ
    
    #calculates the electronic Zeeman, basically just fills in correct paramaters
    def electronicZeeman(self,B,g=None):
        #allows for call if g is set before, without having to pass again
        if type(g) is not NoneType:
            self.setgE(g)
        self.HZE = self.zeemanInteraction(muB,B,self.gE,self.S,self.Edim)
        return self.HZE
    
    #calculates the nuclear Zeeman, basically just fills in correct paramaters
    def nuclearZeeman(self,B,g=None):
        if type(g) is not NoneType:
            self.setgN(g)
        self.HZN = self.zeemanInteraction(muN,B,self.gN,self.I,self.Idim)
        return self.HZN

    #gets eigen energies as frequencies
    def getEigFreq(self,H=None):
        #can pass arbitrary hamiltonian or use static
        if H is None:
            H=self.H      
        E,V = np.linalg.eig(H)
        E = np.sort(np.real(E)) #sort E into increasing values of eigen values
        #VG = VG(:,ind) # arrange the columns in this order
        F = E/(2*np.pi*hbar*1e9)     
        return F

    #calculate the quadrapole interaction
    def quadrapoleInteraction(self,Q):
        self.HQP = (self.I)@Q@(self.I).T
        self.HQP = self.HyperfineReshape(self.HQP)
        if type(self.H) is NoneType:
            self.H=self.HQP
        else:
            self.H+=self.HQP
        return self.HQP



        



#same as above without loops, mainly as an exercise to the Ben, though offers a substantial speedup for large spins.
def spinOperator(J,matricies=False):
    dim = int(2*J+1)

    #generate the 0 to dim array
    a = np.arange(0,dim,dtype = np.csingle)+1
    
    #get each matrix element as vectors
    b = np.tile(a,dim) #tile, repeates the whole array dim times i.e. [0,1]->[0,1,0,1]
    a = np.repeat(a,dim) #repeat, repeates each element dim times in order i.e. [0,1]->[0,0,1,1]
    
    #perform our delta on each array
    k1 = (a==b+1).astype(int)
    k2 = (a+1==b).astype(int)
    k3 = (a==b).astype(int)

    #calculate our matrix elements
    Jx = np.asmatrix((1/2)*(k1+k2)*np.emath.sqrt((J+1)*(a+b-1)-a*b))
    Jy = np.asmatrix((1j/2)*(k1-k2)*np.emath.sqrt((J+1)*(a+b-1)-a*b))
    Jz = np.asmatrix((J+1-a)*k3)

    #convert to individual matricies or the full augmented matrix
    if matricies==True:
        Jx = Jx.reshape((dim,dim))
        Jy = Jy.reshape((dim,dim))
        Jz = Jz.reshape((dim,dim))
        return Jx,Jy,Jz
    else:
        return hstack((Jx.T,Jy.T,Jz.T))

#return the tensor A rotatated by the euler angles in angles
def tensorRotation(A,angles,str='ZYX'):
    R = np.asmatrix(Rotation.from_euler(str,angles).as_matrix())
    return R@A@R.T

#performs a function between each element of A and every element of B, mainly used in getting transition frequencies but kept general
#
def eachElemFunc(A,B,ax=0,func=np.subtract):
    dim = (A.shape[ax])
    A = np.tile(A,dim)
    B = np.repeat(B,dim,axis=ax)
    return func(A,B)

def sphereCart(r,theta,phi):
    return r*np.matrix([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T
