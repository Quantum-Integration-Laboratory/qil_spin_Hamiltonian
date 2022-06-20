import numpy as np
import matplotlib.pyplot as plt

from numpy.core.shape_base import hstack
from scipy.spatial.transform import Rotation
from scipy.linalg import eig
from matplotlib.colors import Normalize

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
        self.dim = self.Edim*self.Idim

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

    #calculate the quadrapole interaction IQI
    def quadrupoleInteraction(self,Q):
        #calculate and reshape our hyperfine term
        self.HQP = (self.I)@Q@(self.I).T
        self.HQP = self.HyperfineReshape(self.HQP)

        #if we haven't initialised the static hamiltonian set it to the quadrupole
        #otherwise add the quadrupol term
        if type(self.H) is NoneType:
            self.H=self.HQP
        else:
            self.H+=self.HQP
        return self.HQP


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
        HZ = np.kron(HZ,np.eye((self.dim)//dim))
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
        E,V = np.linalg.eigh(H)
        E = -1*np.real(E)
        #May need sorting but eigh should return everything in sorted order
            #ind = np.argsort(E) #sort E into increasing values of eigen values
            #V = V[:,ind] # arrange the columns in this order
            #E = E[ind]
        F = E/(h*1e9)     
        return F,V


    #Loops over a set of magnetic field vectors returning the calculated frequencies
        #Bs- b field magnitude
        #thetas - theta angles
        #phis -  phi angles
        #dynamic - dynamic component of the B field should be a lambda function of B dependant hamiltonian terms.
    def runBfieldSweep(self,Bs,thetas,phis, dynamic = None):
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
                    #convert spherical Magnetic field to cartesian coords.
                    B =sphereCart(Bs[k],thetas[i],phis[j])
                    #Calculate our hamiltonian at this timestep
                    HTemp =self.H+dynamic(B)
                    #get the eigen frequencies at this timestep
                    Freq[i,j,k,:],V = self.getEigFreq(HTemp)
                    Htran = dynamic(np.matrix([1,0,0]).T)
                    #print(Htran.shape,V[:,0].shape,type(V))
                    Vecs[i,j,k,:] = self.TransitionStrength(V,Htran)
        return Freq,Vecs
    
    #transition strength for same hamiltonians
    def TransitionStrength(self,V,O):
        N = np.zeros(self.dim**2,dtype = np.csingle)
        k=0
        # for i in range(1,self.dim-1):
        #     for j in range(i+1,self.dim):
        for i in range(self.dim):
            for j in range(self.dim):
                N[k] = fermiElem(V[:,i],O,V[:,j])
                k+=1
        return N.T/(h*1E9)


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

#converts spherical to cartesian coordinates
def sphereCart(r,theta,phi):
    return r*np.matrix([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T

#converts euler angles to spherical coordinate theta,phi. I think this works in all cases but double check
def eulerToSphere(angles,str):
    R =  Rotation.from_euler(str,angles).as_euler('ZYZ')
    theta = R[1]
    phi = R[0]
    return theta,phi

#converts euler angles to spherical theta,phi. More robust performs rotation on zero intialised vector and caluclates coords 
def eulerToSphereRobust(angles,str):
    R =  np.asmatrix(Rotation.from_euler(str,angles).as_matrix)
    u = sphereCart(1,0,0)
    res = R@u
    theta = np.arccos(res[2]/1)
    phi = np.arctan2(res[1],res[0])
    return theta,phi

#calculates |<i|O|j>|^2, given two eigen vectors, Vi,Vj and a hamiltonian H
def fermiElem(Vi,O,Vj):
    E = Vi.H@O@Vj
    return (E.H@E).item()

#transitions strength for differing energy levels and not assiociated with a hamiltonian
def TransitionStrength(V1,V2,O,dim):
    N = np.zeros(int(dim**2),dtype = complex)
    k=0
    for i in range(0,dim):
           for j in range(0,dim):
             N[k] = fermiElem(V1[:,i],O,V2[:,j])
             k+=1
    return N#/(h*1E9)


#Runs through a set of frequencies and optionally transition strengths and gets a weight for each output frequency
#based on the gaussian centered at each true frequency, with amplitude given by the oscilator strengths (1 if these are not provided)
#and with a supplied width defining line thickness.
#Parameters are:
    #- freq- the true frequencies
    #- Bs -  the magnetic field values
    #- OS - [optional] the oscillator strengths, they are assumed to be 1 if not provided
    #- width - the gaussian c factor, relates to line width
    #- cmap - the colourmap to plot with
    #- plot - whether to plot or return the values to plot
    #- title -  the title of the plot
#returns are:
    #None- if plot==True
    #gx,gy,gz- if plot== False, these define the x,y and z/colour information to be plotted

def transitionPixelPlot(freq,Bs,OS=None,frange = None,width=1/10,plot=True,cmap = 'bone',title="Optical Transitions"):
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

