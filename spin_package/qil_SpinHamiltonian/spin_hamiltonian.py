import numpy as np
import matplotlib.pyplot as plt

from numpy.core.shape_base import hstack
from scipy.spatial.transform import Rotation
from scipy.linalg import eig
from matplotlib.colors import Normalize
from numpy_indexed import group_by
import yaml
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import warnings


muB = 9.27400968e-24 #Bohr Magneton (J/T)
muN = 5.05078375e-27 #Nuclear Magneton (J/T)
hbar = 1.055E-34 #Reduced planck constant (Js)
h = 2*np.pi*hbar #Planck constant (Js)

#gaussian distribution a-Amplitude, x-Distribution of x's, m-Center, s- STDEV
gaussian = lambda a,x,m,s: a*np.exp(np.divide(-np.square(x-m),2*np.square(s)))
#normalised gaussian, s.t. integral = 1: m-center,s-STDEV
gaussianNorm = lambda x,m,s: (1/np.sqrt(2*np.pi*np.square(s)))*np.exp(np.divide(-np.square(x-m),2*np.square(s)))


NoneType=type(None)

obsWarning="This function is old and probably ineffecient. It may well never be deleted but you probably should avoid it anyway."

def hamilFromYAML(filename:str,EOveride:float=None,IOveride:float=None,AScale:float=None):
    """
    Creates a hamiltonian from the paramaters contained in a YAML file
    Parameters
    ----------
    filename: str
        The file to get the paramaters from
    EOveride: float
        Allows a different electronic spin to be used, probably less useful
    EOveride: float
        Allows a different nuclear spin to be used, useful for isotopes
    Ascale: float
        Allows a isotropic scaling factor on the hyperfine matrix, useful for isotopes
    Returns
    -------
    spin: cSpinHamiltonian
        The spin hamiltonian based on the file
    """

    with open(filename,'r') as file:
            params=yaml.safe_load(file)
    print(params)
    E=EOveride
    I=IOveride
    if type(EOveride)==NoneType:
        E=eval(params["Spin"]["Espin"])
    if type(IOveride)==NoneType:
        I=eval(params["Spin"]["Ispin"])
    hamil=cSpinHamiltonian(E,I)
    hamil.importYAMLparams(filename)
    if not type(AScale)==NoneType:
        hamil.hyperfineInteraction(hamil.A*AScale)
    dH = hamil.genDerivMatrix()
    hamil.dH=dH
    return hamil

#class containing many parameters for a spin hamiltonian. Ideally it contains H a static hamiltoninan
#and functions to calculate dynamic hamiltonian parts based on provided values.
class cSpinHamiltonian:
    
    def __init__(self,E:float,I:float):
        """
        The class that contains most of the spin hamiltonian paramaters, only the spins are passed on 
        setup as these set fundamental dimensions the rest of the values are passed through setters

        Parameters
        ----------
        E: float
            The Electronic Spin
        I: float
            The nuclear spin

        Returns
        -------
        self: cSpinHamiltonian
            The class
        """
        #Sets up electronic and nuclear spin operators and initialises empty static hamiltonian
        self.Edim = int(2*E+1)
        self.Idim = int(2*I+1)
        self.dim = int(self.Edim*self.Idim)

        self.S = spinOperator(E)
        self.I = spinOperator(I)
        self.Is = spinOperator(I,matricies=True)

        self.H =None
        
        self.M=None
        self.gE=None
        self.gN=None
        self.Hfunc=None
        self.A=None
        self.Q=None
        self.flags=[]
 
    #Calculates the hyperfine interaction hamiltonian IAS, and adds it to the static hamiltonian
    def hyperfineInteraction(self,A:np.matrix)->np.matrix:
        """
        Calculates the hyper fine interaction and adds it to the static Hamiltonian
            IAS
        Parameters
        ----------
        A: np.matrix (3,3)
            The hyperfine tensor
        Returns
        -------
        HHF: np.matrix (dim,dim)
            The calculated hamiltoninan term
        """
        self.A=A
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

    #calculate the quadrapole interaction IQI
    def quadrupoleInteraction(self,Q:np.matrix)->np.matrix:
        """
        Calculates the quadrupole interaction and adds it to the static Hamiltonian
            IQI
        Parameters
        ----------
        Q: np.matrix (3,3)
            The Quadrupole tensor
        Returns
        -------
        HQP: np.matrix (dim,dim)
            The calculated hamiltonian term
        """
        self.Q=Q
        self.HQP = (self.I)@Q@(self.I).T

        self.HQP = self.HyperfineReshape(self.HQP)

        #if we haven't initialised the static hamiltonian set it to the quadrupole
        #otherwise add the quadrupol term
        if type(self.H) is NoneType:
            self.H=self.HQP
        else:
            self.H+=self.HQP
        return self.HQP
    

        # self.HQP=H
        # #if we haven't initialised the static hamiltonian set it to the quadrupole
        # #otherwise add the quadrupol term
        # if type(self.H) is NoneType:
        #     self.H=self.HQP
        # else:
        #     self.H+=self.HQP
        # return self.HQP



    def HyperfineReshape(self,H=None)->np.matrix:
        """
        Reshape the hamiltonian to maintain to be Edim*Idim by Edim*Idim
        
        Parameters
        ----------
        H: np.matrix (Idim,Idim)
            The matrix to be reshaped
        Returns
        -------
        H: np.matrix (dim,dim)
            The reshaped matrix
        """
        Idim = self.Idim
        if type(H) is NoneType:
            H = self.HHF
        a=H[:,0].reshape((Idim,Idim))
        b=H[:,1].reshape((Idim,Idim))
        c=H[:,2].reshape((Idim,Idim))
        d=H[:,3].reshape((Idim,Idim))
        if self.Edim==1:
            return a
        return np.block([[a,b],[c,d]])


    #setters for our g paramaters
    def setgE(self,g:np.matrix)->None:
        """
        Sets the electronic Zeeman, and tells dynamic H it exists
        Parameters
        ----------
        g: (3,3) matrix
            The g tensor
        Returns
        -------
        None
        """
        self.gE= g
        self.flags.append("ZE")
    def setgN(self,g:np.matrix)->None:
        """
        Sets the nuclear Zeeman, for the case where mu_n is required 
        Also tells dynamic H it exists
        Parameters
        ----------
        g: (3,3) matrix
            The g tensor
        Returns
        -------
        None
        """
        self.gN= g
        if not type(g)==type(None):
            self.flags.append("ZN")

    def setM(self,M:np.matrix)->None:
        """
        Sets the nuclear Zeeman, for the case where mu_n is included in the tensor
        Also tells dynamic H it exists
        Parameters
        ----------
        M: (3,3) matrix
            The M tensor
        Returns
        -------
        None
        """

        self.M = M
        if not type(M)==type(None):
           self.flags.append("ZN")

    def importYAMLparams(self,file:str)->None:
        """
        Import the hamiltonian paramaters from a file into the class
        
        Parameters
        ----------
        file : str
            The file to get paramaters from
        Returns
        -------
        None

        """
        with open(file,'r') as file:
            params=yaml.safe_load(file)
        
        self.params=params
        if "Rotation" in self.params:
            rot=self.params["Rotation"]["rot"]
        else:
            rot="ZYZ"
        #Hyperfine
        if "Hyperfine" in self.params and self.Idim>1:
            A=eval(self.params['Hyperfine']["A"])
            A_rot=eval(self.params['Hyperfine']["A_rot"])
            A,RH=tensorRotation(A,A_rot,conv=rot,ret_R=True)
            self.hyperfineInteraction(A)
            self.RH=RH
            self.flags.append("HF")
        #Quadrupole
        if "Quadrupole" in self.params and self.Idim>1:
            Q=eval(self.params['Quadrupole']["Q"])
            Q_rot=eval(self.params['Quadrupole']["Q_rot"])
            Q,RQ=tensorRotation(Q,Q_rot,conv=rot,ret_R=True)
            self.RQ=RQ
            if self.Idim>1:
                self.quadrupoleInteraction(Q)
            self.flags.append("Q")
        #Electronic Zeeman
        if "E_Zeeman" in self.params and self.Edim>1:
            g=eval(self.params['E_Zeeman']["g"])
            g_rot=eval(self.params['E_Zeeman']["g_rot"])
            g,RE=tensorRotation(g,g_rot,conv=rot,ret_R=True)
            self.RE=RE
            self.setgE(g)
        #Nuclear Zeeman
        if "N_Zeeman" in self.params and self.Idim>1:
            if "M" in self.params["N_Zeeman"]:
                M=eval(self.params["N_Zeeman"]["M"])
                M_rot=eval(self.params["N_Zeeman"]["M_rot"])
                M,RN=tensorRotation(M,M_rot,conv=rot,ret_R=True)
                self.RN=RN
                self.setM(M)
                self.setgN(None)
            elif "g" in self.params["N_Zeeman"]:
                g=eval(self.params["N_Zeeman"]["g"])
                g_rot=eval(self.params["N_Zeeman"]["g_rot"])
                g,RN=tensorRotation(g,g_rot,conv=rot,ret_R=True)
                self.RN=RN
                self.setM(None)
                self.setgN(g)
            elif "mu" in self.params["N_Zeeman"]:
                mu=eval(self.params["N_Zeeman"]["mu"])
                self.setM(None)
                self.setgN(mu*np.eye(3))


    #Calculate the zeeman interaction of the form muBgS, allowing for the same function to do nuclear and electronic
    def _zeemanInteraction(self,mu:float,B:np.array,g:np.matrix,S:np.matrix,dim:int)->np.ndarray:
        """
        PRIVATE FUNCTION
        Calculates the nuclear or electronic Zeeman of the form
            um*B.T@g@S.T
        This should be called using electronicZeeman or nuclearZeeman, as this fills in most of the values        

        Parameters
        ----------
        mu : float
            The magneton value
        B : (3,k) array
            The magnetic field values to calculate at
        g: (3,3) matrix
            The tensor to operate on, 
        S: (dim,3) matrix
            The spin matrix to operate on
        dim: int
            (2J+1) with J for either the nuclear or electronic spin, 
            used to calculate the missing dimension such that the return is self.dim x self.dim
        Returns
        -------
        H: (k,self.dim,self.dim)
            The Zeeman Hamiltonian

        """
        HZ = mu*B.T@g@S.T
        #reshape to be Idim*Edim square matrix
            #First converts to a dim x dim, then use the kronecker product to add the missing dimension
        HZ = np.array(HZ).reshape((-1,dim,dim))
        
        I=np.eye((self.dim)//dim)[np.newaxis,:,:]
        return HZ,I
        #HZ = np.kron(HZ,I)
        #enforce axis convention
        #HZ=np.moveaxis(HZ,-1,0)
        return np.squeeze(HZ)
    
    def electronicZeeman(self,B:np.ndarray,g=None)->np.ndarray:
        """
        Calculates the electronic Zeeman, basically just fills in correct paramaters to the generic
        function
        
        Parameters
        ----------
        B : (3,k) array
            The magnetic field values to calculate at
        g: np.matrix(dim,dim) or None
            The tensor to operate on, 
                - If none uses the previously setup one
                - If not none, replaces the previous one
        Returns
        -------
        H: (k,dim,dim)
            The electronic Zeeman Hamiltonian

        """
        #allows for call if g is set before, without having to pass again
        if type(g) is not NoneType:
            self.setgE(g)
        HZ,I=self._zeemanInteraction(muB,B,self.gE,self.S,self.Edim)
        self.HZE = np.squeeze(np.kron(HZ,I))
        return self.HZE
    
    #calculates the nuclear Zeeman, basically just fills in correct paramaters
    def nuclearZeeman(self,B:np.ndarray,g=None)->np.ndarray:
        """
        Calculates the nuclear Zeeman, basically just fills in correct paramaters to the generic
        function
        
        Parameters
        ----------
        B : (3,k) array
            The magnetic field values to calculate at
        g: np.matrix(dim,dim) or None
            The tensor to operate on, 
                - If none it checks whether gN or M was setup and does the correct call based 
                  on needing muN or not
                - If not none, sets the passed value as gN and goes from there
        Returns
        -------
        H: (k,dim,dim)
            The nuclear Zeeman Hamiltonian

        """
        if type(g) is not NoneType:
            self.setgN(g)
        if type(self.M) is not NoneType and type(self.gN) is NoneType:
            #print("Using M")
            HZ,I = self._zeemanInteraction(1,B,self.M,self.I,self.Idim)
        else:    
            HZ,I = self._zeemanInteraction(muN,B,self.gN,self.I,self.Idim)
        self.HZN=np.squeeze(np.kron(I,HZ))
        return self.HZN

    def dynamicH(self,B:np.ndarray,func:callable=None,static:bool=True)->np.ndarray:
        """
        Calculates the dynamic part of the hamiltonian, either from the passed function or by checking what dynamic terms have been passed
        Parameters
        ----------
        B : (3,k) array
            The magnetic field values to call the function on
        func: callable or None
            The function to use it should return a (dim, dim,k) array and take only B as a paramater
            If None it will generate a function from the Zeeman paramaters
        static: bool
            If true add the static hamiltonian to the return
        Returns
        -------
        H: (k,dim,dim)
            The energy hamiltonian in J
            static=True-> The total hamiltonian
            static=False-> Just the dyanmic term at the passed Bs

        """
        if type(func)==type(None):
            HD=np.zeros((B.shape[1],self.dim,self.dim),dtype=np.complex128)
            if "ZE" in self.flags:
                #print("Derivative: Electronic Zeeman")
                HD+=self.electronicZeeman(B)
            if "ZN" in self.flags:
                #print("Derivative: Nuclear Zeeman")
                #Note minus sign
                HD-=self.nuclearZeeman(B)
        else:
            HD=func(B)
        if type(self.H) is NoneType:
            print("Zeeman",HD.shape)
            self.H=np.zeros((self.dim,self.dim))

        
        if static:
            return np.array(self.H)[np.newaxis,...]+HD
            #return np.array(self.H)[...,np.newaxis]+HD
        else:
            return HD
    #gets eigen energies as frequencies
    def getEigFreq(self,H:np.ndarray=None)->tuple[np.ndarray,np.ndarray]:
        """
        Gets the eigen frequencies (spectrum), and eigen vectros of the passed hamiltonian, 
        It is important to note that the eigenvalues are given in frequency units whereas the vectors are in energy units, 
        this is more numerically stable but care must be taken when comparing

        Parameters
        ----------
        H : (k,dim,dim) array
            The hamiltonian to use

        Returns
        -------
        F: (k,dim)
            The spectrum frequencies in (GHz)
        V: (k,dim,dim)
            The spectrum eigenvectors in (GHz)

        """


        #can pass arbitrary hamiltonian or use static
        if H is None:
            H=self.H      
        H=H/(h*1E9)
        E,V = np.linalg.eigh(H)
        #May need sorting but eigh should return everything in sorted order
        E = -1*np.real(E)
        ind = np.argsort(E) #sort E into increasing values of eigen values
        if len(E.shape)!=1:
            V=np.take_along_axis(V,ind[:,np.newaxis,:],axis=-1)
            E=np.take_along_axis(E,ind,axis=-1)
        else:
            V=V[:,ind]
            E=E[ind]
        V=np.squeeze(V)
        E=np.squeeze(E)

        F = E#/(h*1e9)     
        return F,V

    
   
    
    def gradient(self,V:np.ndarray,dH=None)->np.ndarray:
        """
        Calculates the first derivative of the eigenvalues based on the pertubation of the Hamiltonian
        Parameters
        ----------
        V : (k,dim,dim) array
            The eigenvectors at each of the k magnetic field values
        dH: (dim,dim) array or None
            The derivitive of the hamiltonian
            If None it will Use the derivative calculated by the dynamic hamiltonian
        Returns
        -------
        df: (k,dim,3)
            The gradient vector calculated at each field strength for each energy level

        """
        if type(dH)==type(None):
            dH=self.dH

        Dx=matrixElem(V,dH[0,...],V,diag=True)
        Dy=matrixElem(V,dH[1,...],V,diag=True)
        Dz=matrixElem(V,dH[2,...],V,diag=True)

        S1=np.array([Dx,Dy,Dz])
        S1=np.moveaxis(S1,0,-1)
        return np.real(S1)#.T#/(h*1E9)


    def curvature(self,V:np.ndarray,F:np.ndarray,dH=None)->np.ndarray:
        """
        Calculates the matrix of second order partial derivative of the eigenvalues 
        based on the pertubation of the Hamiltonian at each field strength

        Parameters
        ----------
        V : (k,dim,dim) array
            The eigenvectors at each of the k magnetic field values
        F : (k,dim) array
            The Frequencies at each of the k magnetic field values
        dH: (dim,dim) array or None
            The derivitive of the hamiltonian
            If None it will Use the derivative calculated by the dynamic hamiltonian
        Returns
        -------
        df: (k,dim,3,3)
            The Curvature matrix calculated at each field strength for each energy level

        """
        if type(dH)==type(None):
            dH=self.dH
        
        E=F

        pdx=matrixElem(V,dH[0,...],V,diag=False)
        pdy=matrixElem(V,dH[1,...],V,diag=False)
        pdz=matrixElem(V,dH[2,...],V,diag=False)
    
        # we know there will be subtractions between the same energy levels so ignore the divide by zeroes
        with np.errstate(divide='ignore'):
            Tmat=eachElemFunc(E,E,axis=1).reshape(-1,self.dim,self.dim)
            Tmat=1/Tmat
        #Zero out the diagonal, I don't know if there's a better way to do this
        di=np.diag_indices(self.dim)
        Tmat[:,di[0],di[1]]=0

        #Element of S matrix, We do an element wise multiplication with A and Tmat and then matrix multiply this with B
        #Off diagonal terms are going to be mixed and unused so we just take the diagonal
        SE=lambda A,B: np.diagonal(np.matmul(np.multiply(A,Tmat),B),axis1=-2,axis2=-1)

        #get the full matrix
        S=np.array([[SE(pdx,pdx),SE(pdx,pdy),SE(pdx,pdz)],[SE(pdy,pdx),SE(pdy,pdy),SE(pdy,pdz)],[SE(pdz,pdx),SE(pdz,pdy),SE(pdz,pdz)]])

        #reorder the array axis such that the field axis is first
        S=np.rollaxis(S,2,0)
        S=np.rollaxis(S,-1,1)
        S+=S.conj()
        return np.real(S)#/(h*1E9)
    
    def genDerivMatrix(self, func=None)->np.ndarray:
        """
        Generates the derivative matrix in this case assuming only linear (Zeeman) and constant (hyperfine,quadropole terms)
        
        Parameters
        ----------
        func: callable or None
            The function to operate on
            - if None calculates based on the dynamic funcition
            - if Callable calulates across the identiy matrix
        Returns
        -------
        df: (k,dim,3,3)
            The Curvature matrix calculated at each field strength for each energy level
        """
        Bu=np.eye(3)

        if type(func)==type(None):
            dH=self.dynamicH(Bu,static=False)
        else:
            dH= func(Bu)
        dH=dH/(h*1E9)
        self.dHx=dH[0,...]
        self.dHy=dH[1,...]
        self.dHz=dH[2,...]
        self.dH = dH
        return dH 
        
    def spinTransitionStrength(self,V:np.ndarray,O:np.matrix)->np.ndarray:
        """
        Calculates the fermi elements between two energy levels mediated by the passed operator
        
        Parameters
        ----------
        V: np.ndarray
            The eigenvectors calculated at the range of field strengths
        O: np.ndarray
            The operator, in general this will be a pauli matrix. Just pass the 2x2 version the sizing will be handled
        Returns
        -------
        OT: (k,dim**2)
            The array of the oscillator strengths between each energy level of the hamiltonian
        """
        #Get the identity matrix that fixes the dimension, 
        I=np.eye(self.dim//2)
        Od=np.array(np.kron(O,I))
        return np.squeeze(TransitionStrength(V,V,Od,self.dim))
    

class cMultiSpin():
    def __init__(self,spins:list[cSpinHamiltonian],labels=None)->None:
        """
        Basically a wrapper that makes it easier to handle multiple spin systems.
        In general this is useful for handling isotopes

        Parameters
        ----------
        spins: list[cSpinHamiltonian]
            The list of spin hamiltonians to handle

        Returns
        -------
        self: cMultiSpin
            The class
        """
        self.spins=spins
        self.i=[0]+[s.dim for s in spins]
        self.labels=labels
    # def _multiDecorator(func):
    #     """
    #     Didn't end up being used in the end as the functions where slightly too different.
    #     It would be nice to work out a way to use a decorator for things
    #     """
    #     def multiFunc(self,*args,**kwargs):
    #         ret=[]
    #         for i,s in enumerate(self.spins):
    #            R=func(s,*args,**kwargs)
    #            ret.append(R)
    #         return ret
    #     return multiFunc
    def getEigFreq(self,B:np.ndarray,static:bool=True)->tuple[np.ndarray,np.ndarray]:
        """
        Handles the hamiltonian generation and calculation of frequencies and eigenvectors.
        This must be called before any other function. 
        This is the most different from the cSpinHamiltonian implementation as it combines a few functions
            This is only setup to deal with the standard dynamic hamiltonian functions
            We also can't nicely return the eigenvectors so these are stored in the class, and the transitions are returned instead

        Parameters
        ----------
        B : (3,k) array
            The magnetic field values to call the function on
        static: bool
            If true add the static hamiltonian to the return
        Returns
        -------
        Fs: (k,dim) array
            The calculated eigenfrequencies
        Ts: (k,dim**2) array
            The transitions between the calculated eigenfrequencies, Does not take transitions between unlike hamiltonians
        """
        Hs=[]
        Fs=[]
        Vs=[]
        Ts=[]
        for s in self.spins:
            H=s.dynamicH(B,func=None,static=static)
            F,V=s.getEigFreq(H)
            T=eachElemFunc(F,F,axis=1)   
            Hs.append(H)
            Fs.append(F)
            Vs.append(V)
            Ts.append(T)
        self.Hs=Hs
        self.Vs=Vs
        self.Fs=Fs

        Fs=np.hstack(Fs)
        Ts=np.concatenate(Ts,axis=1)

        return Fs,Ts
    
    def gradient(self):
        """
        Calculates the gradient

        Parameters
        ----------
        None
            Relies on paramaters passed into getEigFreq call that first
        Returns
        -------
        Fps: (k,dim1+dim2+...+dimN,3) array
            The calculated gradients
        Tps: (k,dim1**2+dim2**2+...+dimN**2,3) array
            The gradients of the transitions
        """
        ret=[]
        Tp=[]
        for i,s in enumerate(self.spins):
            R=s.gradient(self.Vs[i])
            ret.append(R)
            Tp.append(eachElemFunc(R,R,axis=1))
        return np.concatenate(ret,axis=1),np.concatenate(Tp,axis=1)
    def curvature(self):
        """
        Calculates the Curvature

        Parameters
        ----------
        None
            Relies on paramaters passed into getEigFreq call that first
        Returns
        -------
        Fps: (k,dim1+dim2+...+dimN,3,3) array
            The calculated Curvatures
        Tps: (k,dim1**2+dim2**2+...+dimN**2,3,3) array
            The Curvatures of the transitions
        """
        ret=[]
        Tpp=[]
        for i,s in enumerate(self.spins):
            R=s.curvature(self.Vs[i],self.Fs[i])
            ret.append(R)
            Tpp.append(eachElemFunc(R,R,axis=1))
        return np.concatenate(ret,axis=1), np.concatenate(Tpp,axis=1)
    def spinTransitionStrength(self,Op:np.matrix):
        """
        Calculates the Oscillator strength

        Parameters
        ----------
        Relies on paramaters passed into getEigFreq call that first
        
        Op: np.matrix
            the operator matrix that mediates the transition

        Returns
        -------
        Os: (k,dim1**2+dim2**2+...+dimN**2,3) array
            The calculated transition strength between each energy level
        """
        Os=[]
        for i,s in enumerate(self.spins):
            O=s.spinTransitionStrength(self.Vs[i],Op)
            Os.append(O)
        return np.concatenate(Os,axis=1)
    def interestingTransitions(self,fCav=np.inf):
        Ts=[]
        for i,s in enumerate(self.spins):
            Ts.append(interestingTransitions(s.dim)+self.i[i]**2)
        TOI=np.concatenate(Ts)
        if not np.isinf(fCav):
            _,ZFT=self.getEigFreq(np.zeros((3,2)))
            TOI=TOI[np.where(ZFT[0,TOI]<fCav)]
        return TOI
    def genLabels(self):
        Labels=[]
        for i,s in enumerate(self.spins):
            a,b=tilerepidx(s.dim)
            if self.labels==None:
                idstr="(%s) "%i
            else:
                idstr="(%s) "%self.labels[i]
            lab=[idstr+"%s<->%s"%(a[t],b[t]) for t in range(s.dim**2)]
            Labels.append(lab)

        return np.concatenate(Labels)
#---------------------------------------------------------------------------------------------------------------------------------------
#OLD CLASS FUNCTIONS
#---------------------------------------------------------------------------------------------------------------------------------------
    #transition strength for same hamiltonians
    def TransitionStrength(self,V,O):
        N = np.zeros((self.dim,self.dim),dtype = np.csingle)
        #generate the 0 to dim array
        a = np.arange(0,self.dim,dtype = int)
        #get each matrix element as vectors
        b = np.tile(a,self.dim) #tile, repeates the whole array dim times i.e. [0,1]->[0,1,0,1]
        a = np.repeat(a,self.dim) #repeat, repeates each element dim times in order i.e. [0,1]->[0,0,1,1]
        #if len(V.shape)>=2:
        N=fermiElem(V[...,a],O,V[...,b])
        #N=V[:,a].H@O@V[:,b]
        #N=N.H@N
        # N = np.zeros(self.dim**2,dtype = np.csingle)
        # k=0
        # # for i in range(1,self.dim-1):
        # #     for j in range(i+1,self.dim):
        # for i in range(self.dim):
        #     for j in range(self.dim):
        #         N[k] = fermiElem(V[:,i],O,V[:,j])
        #         k+=1
        return N.T#/(h*1E9)
    
    def initSweep(self,thetas,phis,Bs,fdim = None):
        warnings.warn(obsWarning,DeprecationWarning)

        if type(fdim)==NoneType:
            fdim = [int(self.dim)]
        elif not type(fdim)==list:
            fdim=[fdim]
        return np.zeros((len(thetas),len(phis),len(Bs),*fdim),dtype = np.csingle)
        #transition strength for same hamiltonians
        # This is a little bit hacky to work with higher dimensions but seems to work well enough
    def firstOrderEnergySensitivity(self,V,A):
        warnings.warn(obsWarning,DeprecationWarning)
        #N = np.diag(V.H@A@V)
        #return N.T/(h*1E9)
        N=matrixElem(V,A,V)
        return N.T#/(h*1E9)
    
    def runMultiThreadedSweep(self,Bs):
        #if __name__ == '__main__':
        warnings.warn(obsWarning,DeprecationWarning)
        F=np.zeros((len(Bs),self.dim))
        Fp=np.zeros((len(Bs),self.dim,3),dtype = np.csingle)
        Fpp=np.zeros((len(Bs),self.dim,3,3),dtype = np.csingle)
        #resultList=[]
        # with Pool() as pool:
        #     for i,result in enumerate(pool.map(self.calcBOptParams,Bs)):
        #         F[i,:],Fp[i,:],Fpp[i,:]=result
        #         print(result,flush=True)
        pool=Pool(4)
        result=pool.map(self.calcBOptParams,Bs)
        print(result)
        return eachElemFunc(F,F,axis=1),eachElemFunc(Fp,Fp,axis=1),eachElemFunc(Fpp,Fpp,axis=1)
        
        #Loops over a set of magnetic field vectors returning the calculated frequencies
        #Bs- b field magnitude
        #thetas - theta angles
        #phis -  phi angles
        #dynamic - dynamic component of the B field should be a lambda function of B dependant hamiltonian terms.
    def runBfieldSweep(self,Bs,thetas,phis, dynamic = None):
        warnings.warn(obsWarning,DeprecationWarning)
        #sets the default dynamic term as both electronic and nuclear zeeman
        if type(dynamic)==NoneType:
            dynamic=lambda B: self.electronicZeeman(B)-self.nuclearZeeman(B)
        #sets up our frequency vector
        Freq = np.zeros((len(thetas),len(phis),len(Bs),int(self.dim)),dtype = np.csingle)
        Vecs = np.zeros((len(thetas),len(phis),len(Bs),int(self.dim**2)),dtype = np.csingle)
        
        #our loop
        for i in range(len(thetas)):
            for j in range(len(phis)):
                for k in range(len(Bs)):
                    # #convert spherical Magnetic field to cartesian coords.
                    # B =sphereCart(Bs[k],thetas[i],phis[j])
                    # #Calculate our hamiltonian at this timestep
                    # HTemp =self.H+dynamic(B)
                    # #get the eigen frequencies at this timestep
                    # Freq[i,j,k,:],V = self.getEigFreq(HTemp)
                    # Htran = dynamic(np.matrix([1,0,0]).T)
                    # #print(Htran.shape,V[:,0].shape,type(V))
                    # Vecs[i,j,k,:] = self.TransitionStrength(V,Htran)

                    Freq[i,j,k,:],Vecs[i,j,k,:]=self.calcB(Bs[k],thetas[i],phis[j],dynamic)
        return Freq,Vecs
    
    
    #overload of above with seperated Hx,Hy and Hz elements
    def curvatureCalculationOld(self,Hx,Hy,Hz,V,F,indiv=True,eig=True,transitions=None):
        warnings.warn(obsWarning,DeprecationWarning)
        with np.errstate(divide='ignore'):
            #convert frequency back to energy, alternatively we could convert our eigenvectors to be frequencies but this
            # seems to cause numerical error issues
            E=F*h*1E9
            #V/=h*1E9
            
            #single element of partial derivative wrt B, sort of see, example usage for correct form
            pdBv = lambda i,j,H: np.diag(((V[:,i].H)@H@V[:,j])).reshape((self.dim,self.dim))


            #generate the 0 to dim array
            a = np.arange(0,self.dim,dtype = int)
            #get each matrix element as vectors
            b = np.tile(a,self.dim) #tile, repeates the whole array dim times i.e. [0,1]->[0,1,0,1]
            a = np.repeat(a,self.dim) #repeat, repeates each element dim times in order i.e. [0,1]->[0,0,1,1]
            
            
        
            Nx = pdBv(a,b,Hx)
            Ny = pdBv(a,b,Hy)
            Nz = pdBv(a,b,Hz)
            Sgn =np.nan_to_num(1/(E[a]-E[b]).reshape(self.dim,self.dim),posinf=0,nan=0,neginf=0)
            
            #setup our funciton, each transition will be associated with one of the diagonals of Ni*Nj
            S = lambda A,B,i : (A[i,:]@np.diag(Sgn[:,i])@B[:,i])
            #print(S(Nx,Nx,c))
            
            Es = []
            if transitions==None:
                transitions= range(self.dim)
            for i in transitions:
                SMat= np.matrix([[S(Nx,Nx,i),S(Nx,Ny,i),S(Nx,Nz,i)],[S(Ny,Nx,i),S(Ny,Ny,i),S(Ny,Nz,i)],[S(Nz,Nx,i),S(Nz,Ny,i),S(Nz,Nz,i)]])
                # SMat+=np.conj(SMat)
                if indiv:            
                    #calculate the largest eigen values as a single number proxy to the 
                    E = np.linalg.eigvalsh(SMat)
                    Es.append(E[np.abs(E).argmax()])
                else:
                    Es.append(SMat)
            return np.array(Es).T/(h*1E9)
    #alternative curvature calculations that, calculates the H matricies given, a function A, and a pertubation matrix Bp
    def curvatureCalculationAlt(self,A,Bp,V,F,indiv=False):
        warnings.warn(obsWarning,DeprecationWarning)
        with np.errstate(divide='ignore'):
            #split A into its x,y,z elements, and reshape into a dim x dim matrix
            Hx = A(Bp[:,0])
            Hy = A(Bp[:,1])
            Hz = A(Bp[:,2])
            
            return self.curvatureCalculation(Hx,Hy,Hz,V,F,indiv=indiv)
    def curvatureCalculationNaive(self,Hx,Hy,Hz,V,F,indiv=True,eig=True): 
        warnings.warn(obsWarning,DeprecationWarning)
        #there is known divide by zeros which we ignore
        with np.errstate(divide='ignore'):
            #split A into its x,y,z elements, and reshape into a dim x dim matrix
            #Hx = A(Bp[:,0])
            #Hy = A(Bp[:,1])
            #Hz = A(Bp[:,2])
            
            
            #a = np.arange(0,self.dim,dtype = np.int)
            
            E=F*h*1E9   
            #             /(E[n]-E)
            pdb=lambda A,B,n,m: np.trace(np.nan_to_num(((V[:,m].H)@A@V[:,n])@((V[:,n].H)@B@V[:,m])/(E[n]-E[m]),posinf=0,nan=0,neginf=0))
            #Ed = np.nan_to_num(1/eachElemFunc(E,E).reshape(self.dim,self.dim),posinf=0,nan=0,neginf=0)
            #pdbn=lambda A,B: np.nan_to_num((((V.H)@A@V)@((V.H)@B@V)@Ed),posinf=0,nan=0,neginf=0)
            #print("EMat",Ed)
            #print("E[0]",E[0]-E)

            # pdB = lambda i,j,H : ((V[:,i].H)@H@V[:,j])/(np.sqrt(np.abs(E[i]-E[j])))
            # pdBa = lambda i,j,H: np.diag(np.nan_to_num(pdB(i,j,H),0)).reshape((self.dim,self.dim))
            
            #print(np.diag(np.dot(pdbn(Hx,Hx),Ed)))
           # print(np.sum(pdbn(Hx,Hx),axis=1))

            Es = []
            for n in range(self.dim):
                SMat=np.zeros((3,3),dtype=np.complex128)
                for m in range(self.dim):
                    #print((pdb(Hx,Hx,i))/np.sum(E[i]-E))
                    SMat+= np.matrix([[pdb(Hx,Hx,n,m),pdb(Hx,Hy,n,m),pdb(Hx,Hz,n,m)],[pdb(Hy,Hx,n,m),pdb(Hy,Hy,n,m),pdb(Hy,Hz,n,m)],[pdb(Hz,Hx,n,m),pdb(Hz,Hy,n,m),pdb(Hz,Hz,n,m)]])
                if indiv:
                    #calculate the largest eigen values as a single number proxy to the 
                    Evals = np.linalg.eigvalsh(SMat)
                    Es.append(Evals[np.abs(Evals).argmax()])
                else:
                    Es.append(SMat)
            
            return np.array(Es).T/(h*1E9)        
        #print("pdbxx",pdB(0,1,Hx).shape)
            
            #SMat= np.matrix([[pdb(Hx,Hx),pdb(Hx,Hx),S(Nx,Nz)],[S(Ny,Nx),S(Ny,Ny),S(Ny,Nz)],[S(Nz,Nx),S(Nz,Ny),S(Nz,Nz)]])
    def calcBFOsc(self,B,theta,phi,dynamic=None,Vecs = False):
        warnings.warn(obsWarning,DeprecationWarning)
        #sets the default dynamic term as both electronic and nuclear zeeman
        if type(dynamic)==NoneType:
            dynamic=lambda B: self.electronicZeeman(B)-self.nuclearZeeman(B)
        #convert spherical Magnetic field to cartesian coords.
        B =sphereCart(B,theta,phi)
        #Calculate our hamiltonian at this timestep
        HTemp =self.H+dynamic(B)
        #get the eigen frequencies at this timestep
        Freq,V = self.getEigFreq(HTemp)
        Htran = dynamic(np.matrix([1,0,0]).T)
        #print(Htran.shape,V[:,0].shape,type(V))
        OS = self.TransitionStrength(V,Htran)
        if Vecs==True:
            return Freq,OS,V
        else:
            return Freq,OS
    def calcBOptParams(self,B):
        warnings.warn(obsWarning,DeprecationWarning)
        HG = self.calcH(B)
        FG,VG = self.getEigFreq(HG)
        F=FG*1E3
        
        Fp = []
        for l in range(3):
            v = self.firstOrderEnergySensitivity(VG,self.A[l])
            Fp.append(v)
        Fp = np.array(Fp).T
        Fpp = self.curvatureCalculationNaive(self.A[0],self.A[1],self.A[2],VG,FG,indiv=False).T#*spin.muN
        
        return (F,Fp,Fpp)

#---------------------------------------------------------------------------------------------------------------------------------------

#Fast spin operator calculation without loops, mainly as an exercise to the Ben, though offers a substantial speedup for large spins.
def spinOperator(J:float,matricies:bool=False)->np.matrix:
    """
    Generates the spin operator matricies

    Parameters
    ----------
    J : float 
        The spin value to generate the matricies from
    matricies : bool
        Whether to return the individual matricies or not
    Returns
    -------
    Jmat: np.matrix (3,dim**2)
        The spin matrix elements as a single matrix for easy calculations
    Jx,Jy,Jz: tuple(np.matrix,np.matrix,np.matrix)
        if `matricies==True` returns the individual (dim,dim) matricies
    """
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
def tensorRotation(A:np.matrix,angles:np.array,conv:str='ZYZ',dumb:bool=False,ret_R:bool=False):
    """
    Rotates a tensor by the given euler angles

    Parameters
    ----------
    A : np.matrix (3,3) 
        The matrix to rotate
    angle: np.array (3)
        The Euler angles to rotate by
    conv: str
        The Euler angle convention to use, defines the axis each rotation is about
    dumb : bool
        Mostly a testing flag to make sure things line up with brute forcing
    ret_R:bool
        Whether to return the rotation matrix alongside the vector
    Returns
    -------
    RA: np.matrix (3,3)
        The tensor rotated by the given angles
    R: np.matrix (3,3)
        if `ret_R==True` returns rotation matrix
    """
    if dumb:
        R=np.eye(3)
        Rx = lambda a : np.matrix([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        Ry = lambda a : np.matrix([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        Rz = lambda a : np.matrix([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        for i,a in enumerate(angles):
            if conv[i]=="Z":
                R=R@Rz(a)
            elif conv[i]=="Y":
                R=R@Ry(a)
            elif conv[i]=="X":
                R=R@Rx(a)
    else:
        R = np.asmatrix(Rotation.from_euler(conv,angles).as_matrix())
    if ret_R:
        return R@A@R.T,R
    else:
        return R@A@R.T


#performs a function between each element of A and every element of B, mainly used in getting transition frequencies but kept general
#
def eachElemFunc(A:np.ndarray,B:np.ndarray,axis:int=1,func:callable=np.subtract, nosymm:bool=False)->np.ndarray:
    """
    Performs an operation between each element of A and each element of B across the given axis
    This is mostly used to calculate transition and hence defaults to subtraction across the first axis
    Parameters
    ----------
    A : np.ndarray (...,i,...)
        The first array
    B : np.ndarray (...,j,...)
        The second array
    axis : int
        The axis to operate over
    func: Callable
        The operation to perform should be of the signature `R=func(A,B)`
        Defaults to subtraction
    nosymm : bool
        Removes elemnts operating on the opposite case i.e. only includes func(a,b) not func(b,a)
        This offers no calculation speedup, vectorisation means its generally quicker to post select
    Returns
    -------
    res: np.ndarray (...,i*j,...)
        The result of the operation on each element
    """
    dim = (A.shape[axis])
    
    #tile takes a tuple of axis to tile across, we want to make sure only ax is non-one
    tdim = np.ones(len(B.shape))
    tdim[axis]*=dim
    tdim = tuple(tdim.astype(int))
    
    A = np.tile(A,tdim)
    B = np.repeat(B,dim,axis=axis)
    res= func(A,B)
    if nosymm:
        n=dim
        half=np.concatenate([np.arange(n*i+1+i,n*(i+1)) for i in range(n)])
        return res[half]
    else:
        return res



def matrixElem(Vi:np.ndarray,O:np.matrix,Vj:np.ndarray,diag:bool=False):
    '''
    Calculates all sets of <Vi|O|Vj>, given two eigen vectors, Vi,Vj and a operator O
    Parameters
    ----------
    Vi: np.ndarray (k,dim,dim)
        The First set of eigen vectors The vectors are selected by the final dimension
    O: np.matrix (dim,dim)
        The operator matrix 
    Vj: np.ndarray (k,dim,dim)
        The Second set of eigen vectors
    diag: bool
        Whether to only return the diagonal elements useful for things of the form <Vi|O|Vi>
    Returns
    -------
    N: np.array(k,dim,dim) or (k,dim) if diag==True
        The calculated matrix elements
    '''
    VH=Vi.conj().swapaxes(-2,-1) #ndarrays don't have a .H, but anything larger than 2D can't be a matrix
    N=VH@O@Vj
    if diag:
        N=np.diagonal(N,axis1=-2,axis2=-1)
    return N

def fermiElem(Vi,O,Vj,diag=True):
    '''
    Calculates all sets of |<Vi|O|Vj>|^2, given two eigen vectors, Vi,Vj and a operator O

    Parameters
    ----------
    Vi: np.ndarray (k,dim,dim)
        The First set of eigen vectors The vectors are selected by the final dimension
    O: np.matrix (dim,dim)
        The operator matrix 
    Vj: np.ndarray (k,dim,dim)
        The Second set of eigen vectors
    diag: bool
        Whether to only return the diagonal elements
    Returns
    -------
    N: np.array(k,dim,dim) or (k,dim) if diag==True
        The calculated matrix elements
    '''
    E=matrixElem(Vi,O,Vj,diag)
    
    #E.H@E is probably quicker but really annyoing given the dimensions
    E=np.square(np.abs(E))
    
    if E.shape==1:
        return (E).item()
    else:
        return E
    


def TransitionStrength(V1,V2,O,dim):
    '''
    Calculates The transition strength as per
    |<Vi|O|Vj>|^2, given two eigen vectors, Vi,Vj and a operator O
    
    Parameters
    ----------
    Vi: np.ndarray (k,dim,dim)
        The First set of eigen vectors The vectors are selected by the final dimension
    O: np.matrix (dim,dim)
        The operator matrix 
    Vj: np.ndarray (k,dim,dim)
        The Second set of eigen vectors
    dim: int
        The dim of the given matricies
    Returns
    -------
    N: np.array(k,dim**2)
        The calculated Transitions strengths 
    '''
    alt = fermiElem(V1,O,V2,diag=False)#V1.H@O@V2
    alt = np.squeeze(np.asarray(alt.reshape(-1,dim**2)))
    
    return alt#/(h*1E9)#np.diag(N)#/(h*1E9)

def tilerepidx(dim:int):
    """
    returns two index arrays such that each index from 0 to dim interacts with every other index
    Probably actually just a flat meshgrid
    
    Parameters
    ----------
    dim: int
        The size of the array
    Returns
    -------
    a: np.ndarray (dim**2)
        indicies in the order [0,1,...,dim]*dim
    b np.ndarray (dim**2)
        indices in the order [0,0,0,1,1,1,...,[dim,dim,dim]*dim]
    """
    #generate the 0 to dim array
    a = np.arange(0,dim,dtype = int)
    #get each matrix element as vectors
    b = np.tile(a,dim) #tile, repeates the whole array dim times i.e. [0,1]->[0,1,0,1]
    a = np.repeat(a,dim) #repeat, repeates each element dim times in order i.e. [0,1]->[0,0,1,1]
    return a,b

def transitionPixelPlot(freq,Bs,OS=None,frange = None,width=1/10,plot=True,cmap = 'bone',title="Optical Transitions"):
    """
    Runs through a set of frequencies and optionally transition strengths and gets a weight for each output frequency
    based on the gaussian centered at each true frequency, with amplitude given by the oscilator strengths (1 if these are not provided)
    and with a supplied width defining line thickness.

    Parameters
    ----------
        - freq- the true frequencies
        - Bs -  the magnetic field values
        - OS - [optional] the oscillator strengths, they are assumed to be 1 if not provided
        - width - the gaussian c factor, relates to line width
        - cmap - the colourmap to plot with
        - plot - whether to plot or return the values to plot
        - title -  the title of the plot
    Returns
    -------
        None- if plot==True
        gx,gy,gz- if plot== False, these define the x,y and z/colour information to be plotted

    """
    
    #get the frequncy limits based on true range if nothing is provided
    if type(frange)==NoneType:
        yi = np.linspace(np.min(freq),np.max(freq),Bs.shape[-1])
    else:
        yi=frange

    #initialise our coordinate values
    gx,gy = np.meshgrid(Bs,yi,indexing='xy')
    gz = np.zeros((Bs.shape[-1],yi.shape[-1]),dtype=complex)
    
    #get number of frequecny values
    itmax = freq.shape[-1]

    #if we don't provide Oscillator strengths assume 1
    if type(OS)==NoneType:
        for out in range(0,Bs.shape[-1]):
            for it in range(0,itmax):
                #gz is handled per column but we have to repeat for all true frequencies
                gz[out,:] = gz[out,:]+np.exp(-1/2*np.power((yi-freq[out,it])/(width),2,dtype=complex),dtype=complex)
    #otherwise use oscillator strength as amplitude
    else:
        for out in range(0,Bs.shape[-1]):
            for it in range(0,itmax):
                #gz is handled per column but we have to repeat for all true frequencies, and oscilator strengths
                gz[out,:] = gz[out,:]+OS[out,it]*np.exp(-1/2*np.power((yi-freq[out,it])/(width),2,dtype=complex),dtype=complex)
    
    if plot==True:
        #plotting stuff
        plt.pcolor(gx*1E3,gy,1-np.real(gz).T,shading='auto',cmap=cmap,norm=Normalize())
        plt.xlabel('Magnetic Field strength (mT)')
        plt.ylabel('Detuning (GHz)')
        plt.title(title)
        plt.show()
        plt.close()
    else:
        return gx,gy,1-np.real(gz).T


def hyperquad(A:np.matrix,O:np.matrix,B:np.matrix)->np.matrix:
    """
    Calculates terms of the form A@O@C, returning the correct shape, this covers both hyperfine and quadrupole
    Parameters
    ----------
    A: np.matrix (3,Adim)
        First tensor
    O: np.matrix (3,3)
        Second (operator) tensor
    B: np.matrix (3,Bsdim)
        The Quadrupole tensor

    Returns
    -------
    H: np.matrix (Adim*Bdim,Adim*Bdim)
        The calculated hamiltonian term
    """
    Bdim=int(np.sqrt(B.shape[0]))
    Adim=int(np.sqrt(A.shape[0]))


    OB=np.array(O@B.T)
    OBx=OB[0,:].reshape(Bdim,Bdim)
    OBy=OB[1,:].reshape(Bdim,Bdim)
    OBz=OB[2:,:].reshape(Bdim,Bdim)

    Ax=A[:,0].reshape(Adim,Adim)
    Ay=A[:, 1].reshape(Adim,Adim)
    Az=A[:,2].reshape(Adim,Adim)

    H=np.array(np.kron(Ax,OBx)+np.kron(Ay,OBy)+np.kron(Az,OBz))

def properRotation(j,n):
    """
    Determines the character table element for a proper rotation
    j: The spin
    n: the fraction of rotation coming from the symmetry operation
    """
    return np.divide(np.sin((j+1/2)*2*np.pi/n),np.sin(np.pi/n))


def interestingTransitions(dim:int)->np.array:
    """
    Returns "Interesting" Transitions, this is mostly for cavity matching it assumes that transitions with energy levels from opposite halves 
    of the spectrum will have a positive gradient

    Parameters
    ----------
    dim: int
        The number of energy levels

    Returns
    -------
    TOI: np.array
        The Transitions of interest
    """
    c=np.arange(0,dim//2,1)
    d=np.arange(dim//2,dim,1)

    return eachElemFunc(d,dim*c,axis=0,func=np.add)



#---------------------------------------------------------------------------------------------------------------------------------------
#SPHERICAL COORDINATE FUNCTIONS
#---------------------------------------------------------------------------------------------------------------------------------------

def sphereCart(r,theta,phi):
    '''
    converts spherical to cartesian coordinates
    '''
    #return r*np.matrix([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T
    return r*Rotation.from_euler('yz',[theta,phi]).as_matrix()@np.matrix([0,0,1]).T

def sphereUnit(vals,unit=True,conv='yz'):
    '''
    returns the unit vector from the given spherical coords theta,phi
    '''
    RM = Rotation.from_euler(conv,vals).as_matrix()
    if unit:
        return RM@np.matrix([0,0,1]).T
    else:
        return RM
def eulerToSphere(angles,conv):
    '''
    converts euler angles to spherical coordinate theta,phi. 
    I think this works in all cases but double check
    '''
    R =  Rotation.from_euler(conv,angles).as_euler('ZYZ')
    theta = R[1]
    phi = R[0]
    return theta,phi

def eulerToSphereRobust(angles,conv):
    '''
    converts euler angles to spherical theta,phi. More robust performs rotation on zero intialised vector and caluclates coords 
    '''
    R =  np.asmatrix(Rotation.from_euler(conv,angles).as_matrix)
    u = sphereCart(1,0,0)
    res = R@u
    theta = np.arccos(res[2]/1)
    phi = np.arctan2(res[1],res[0])
    return theta,phi

#---------------------------------------------------------------------------------------------------------------------------------------
#OLD FUNCTIONS: Not neccassarily depreciated but haven't been tested in a while
#---------------------------------------------------------------------------------------------------------------------------------------



#Calculates an absorbtion spectra, by stacking gaussians at each transition
def absorptionSpectra(freq,FWHM,Os,xs):
    #calculate the standard deviation from the broadnening FWHM
    c = FWHM/(2*np.sqrt(2*np.log(2)))
    #setup the y axis as a zero initialsed of the same type as the x axis
    a =  np.zeros_like(xs)
    #run through all our freqs with OS as height f as center, and c as stdev
    for i,f in enumerate(freq):
        a+= gaussian(Os[i],xs,f,c)
    return a

def quickAbsorbtion(S1,S2,B,theta,phi,OS,xs,FWHM,dyn=lambda S,B: S.electronicZeeman(B), print = True):
    """
    Calculates and plots the absorbtion spectra for a single magnetic field strength B at angle, theta, phi
    Parameters
    ----------
    S1- State one, i.e. ground
    S2 - State two, i.e. excited
    B,theta,phi- Magnetic field strength and angle
    OS-Transition matrix
    Xs- X values
    FWHM- inhomogeneous broadinening term.
    dyn -  dynamic term of the hamiltonian

    Returns
    ----------
    """

    fg,Vg,eVg = S1.calcB(B,theta,phi,dynamic=lambda B:dyn(S1,B),Vecs=True)
    fe,Ve,eVe = S2.calcB(B,theta,phi,dynamic=lambda B:dyn(S2,B),Vecs=True)

    f = np.real(eachElemFunc(fe,fg))
    OS = np.real(TransitionStrength(eVg,eVe,OS,S1.dim))
                
    A = absorptionSpectra(f,FWHM,OS,xs)

    if print==True:

        plt.plot(xs,A)
        plt.xlabel('Detuning (GHz)')
        plt.ylabel('Absorption Coefficient (arb.)')
        plt.show()
        fig = plt.gcf()
        plt.close()

        return fig
    else:
        return A

#---------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------