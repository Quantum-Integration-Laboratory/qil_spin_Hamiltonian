# Overview
This contains repo contains the spin hamiltonian package written over the course of my PhD. It mainly focuses on work in the rare-earths but is probably more broadly applicable with a bit of work. Most of the functions focus on the generation of energy level diagrams and transition spectra, with a few interfaces for each. Outside of the basics, the main focus has been on finding ZEFOZ points as well as matching to cavities for ESR style measurements, it has proven very effecient at this but this may mean its a little clunky for other uses. Feel free to fork an make any changes.

- 'qil_SpinHamiltonian' contains the actual package functions
- 'ion_params' contains saved paramaters for a variety of rare-earth ion-host combinations
- 'experiments' contains a few of the final experiments discussed in my thesis (link provided at some point).
# Spin Hamiltonian
This code is designed to implement a model of the spin hamiltonian. This was initially designed with the intention of finding Zero First Order Zeeman (ZEFOZ) points though has expanded in scope somewhat since then.

Two main files are implemented: the main `spin_hamiltonian.py` with most of the functions and `search.py` which is intended for implementing search functions though is a bit limited for now.



# Theory 

Spin Hamiltonians should consist of some combination of the form:
```math
H = \underbrace{I\times A\times S}_{\textrm{hyperfine}}+\underbrace{I\times Q\times I}_{Quadrupole}+\underbrace{\mu_{B}\times B\times g\times S}_{\textrm{Electronic Zeeman}}-\underbrace{\mu_{n}\times B\times g_{n}\times I}_{\textrm{Nuclear Zeeman}}
```
The I and S matrices are of fixed form generated from the electronic and nuclear spin degrees respectively. We track the size of these matrices through the parameters `Edim` and `Idim` respectively (collectively `Jdim`) with size $(2J+1)$. for ease of calculation these are reshaped into a `Jdim x 3` matrix, such that the first column consists of all the elements of the $J^x$ matrix the second $J^y$ ect.

$$ J^{x}_{ij} = \frac{1}{2}(\delta{i,j+1}+\delta_{i+1,j})\sqrt{(J+1)(i+j-1)-ij} $$ 
$$ J^{y}_{ij} = \frac{i}{2}(\delta{i,j+1}-\delta_{i+1,j})\sqrt{(J+1)(i+j-1)-ij} $$ 
$$ J^{z}_{ij} = (J+1-i)\delta{ij} $$

The hyperfine (A), Quadrupole (Q), and Zeeman (g and $g_n$)tensors will be more dependent on the species used and must be entered manually; this will be explained in more detail below. In general particularly for low symmetry systems these will consist of three elements of a diagonal matrix and three rotation elements ($\alpha, \beta, \gamma$), which define a rotation in euler notation, it is also important to note which notation this rotation is done in. This maps the measured data to a more universal lab or crystal frame. This is done through the tensor rotation below
```math
A= R(\alpha,\beta,\gamma)\times A\times R^T(\alpha,\beta,\gamma)
```

We separate the hamiltonian into two parts, the static hamiltonian which is not dependent on external factors, namely the magnetic field. And the dynamic term which is dependent on things like the magnetic field. In general the static term is pre calculated and the dynamic calculated as needed. The final resulting hamiltonian will be a square matrix with side length `(Edim x IDim)`

From here the hamiltonian is converted into energy units, and the eigenvectors and values are easily calculated, though some care is taken to ensure these are in the correct order. An example of the calculated energy levels for Yb:YVO is shown below. 

![png](README_files/README_16_0.png)

In order to then calculate transition frequencies, we can then subtract each energy level from every other energy level in either the same or a different hamiltonian. Transition strengths can be included, calculated by the fermi element of the overlap of the eigenvectors mediated by some matrix. The calculation of these matrices is generally non-trivial and left as an exercise to the reader.
```math
O=\left|\left<\psi_n\right|M\left|\psi_n\right>\right|^2
```
![png](README_files/README_20_1.png)

## Derivatives for ZEFOZ finding
Also of interest is particularly in the case of finding ZEFOZ or clock transitions as the first and second derivatives of the hamiltonian. 

### First order
Coming from perturbation theory the first order sensitivity $\vec{S}_1$ of the transition $f_n$ with respect to magnetic field is given by
```math
\vec{S}_{1} = \frac{\partial f_{n}}{\partial B_{x}}\vec{i}+\frac{\partial f_{n}}{\partial B_{y}}\vec{j}+\frac{\partial f_{n}}{\partial B_{z}}\vec{k}
```
With 
```math
\frac{\partial f_{n}}{\partial B_{i}} = \left<\psi_{n}\right|\frac{\partial H}{\partial B_{i}}\left|\psi_{n}\right>
```
In the case of only zeeman terms in our dynamic hamiltonian the derivative of the hamiltonian will simply be its value at the unit vector along each of the cardinal directions.

The sensitivity of the transition is then:
```math
\frac{\partial f_{nm}}{\partial B_{i}}=\frac{\partial f_{n}}{\partial B_{i}}-\frac{\partial f_{m}}{\partial B_{i}}
```
### Second Order
The second order is slightly complicated but still an extension from perturbation theory given by
```math
\vec{S}_{2} = \begin{pmatrix}\frac{\partial^2 f}{\partial B_{x}\partial B_{x}}&\frac{\partial^2 f}{\partial B_{x}\partial B_{y}}&\frac{\partial^2 f}{\partial B_{x}\partial B_{z}}\\\frac{\partial^2 f}{\partial B_{y}\partial B_{x}}&\frac{\partial^2 f}{\partial B_{y}\partial B_{y}}&\frac{\partial^2 f}{\partial B_{y}\partial B_{z}}\\\frac{\partial^2 f}{\partial B_{z}\partial B_{x}}&\frac{\partial^2 f}{\partial B_{z}\partial B_{y}}&\frac{\partial^2 f}{\partial B_{z}\partial B_{z}}\end{pmatrix}
```

With
```math
\frac{\partial^2 f}{\partial B_{i}\partial B_{j}} =\sum_{m\neq n}\frac{1}{f_{n}-f_{m}}\left<\psi_{m}\right|A_{i}\left|\psi_{n}\right>\left<\psi_{n}\right|A_{j}\left|\psi_{m}\right> 
```
A few tricks are played in code to ease this calculation that are worth highlighting here.
We first calculate all the matrix elements.
```math
\partial_{i}^{nm}=\bra{\psi_{n}}\frac{\partial H}{\partial B_{i}}\ket{\psi_{m}}
```
We then create a matrix $T$ of all transition energies such that
```math
T_{nm}=\frac{1}{E_n-E_{m}}
```
Enforcing that $T_{nn}\equiv 0$. Each element can then be calculated as
```math
\frac{\partial^{2}}{\partial B_{i}\partial B_{j}}f_{n}=\textrm{diag}((\mathbf{\partial}_{i}\odot T)\cdot \mathbf{\partial}_{j})
```
With $\odot$ the hadamard or element wise product and $\cdot$ matrix multiplication.

With this the we can determine the field induced dephasing rate

```math
\frac{1}{\pi T_{2}} = \vec{S}_{1}\cdot\Delta\vec{B}+\Delta\vec{B}\cdot\vec{S}_{2}\cdot\Delta\vec{B}
```
### Further Details
#### Hyperfine shape
The speedup of making the columnular spin matricies makes things a little bit tricky when it comes to doing hyperfine, for the most part our electron spin is spin half resulting in 3 2x2 matrices, or a 4x3 in our form. The nuclear spin will generally be bigger with 3 (2I+1)x(2I+1) matricies (2J+1)^2 x 3.
The direct multiplication will result in a matrix with the correct dimension but not the desired $[2(2I+1)]\times[2(2I+1)]$ shape.
```math
\underbrace{HHF}_{(2J+1)^2 \times 4}=\underbrace{I}_{(2J+1)^2 \times 3}\cdot\underbrace{A}_{3\times3}\cdot\underbrace{S^T}_{3\times4}
```
In actuality what we are aiming for is something resembling them kronecker product of the two spin matricies, simplifying to the z case i.e. `A=np.diag([0,0,1])`
```math
\underbrace{HHF}_{2(2J+1) \times 2(2J+1)}=\underbrace{I_{z}}_{(2J+1) \times (2J+1)}\otimes \underbrace{S_{z}}_{2\times 2}
```
# Installation
The package should be installed with a package manager. 
1. Clone this repo to a local location.
2. Direct a terminal to the `spin_package` directory i.e. `cd SOMEPATH/spin_package`
3. From this terminal run `pip install --editable .`
    - This means when further updates are pulled they are automatically recognised by the package

# spin_hamiltonian.py
In terms of actually using the code. Most of the code is handled by the `cSpinHamiltonian` class; it can be instantiated manually or by passing a YAML file containing all the relevant parameters.

An interactive version of this code is found in [example_usage.ipynb](./example_usage.ipynb)

> [!NOTE]
> - Hamiltonian input Parameters should be in Joules, Magnetic fields in T.
> - Output frequency units will be in GHz.
> - For calculating across `(3,k)` magnetic field values work has been done to a abide by the following axis convention `(k,d,...)` for all outputs
>   - k, is the number of field values
>   - d, is the dimension of number of energy levels
>   - any additional dimensions follow
>   - This is not the most natural ordering python wise but makes it easy to plot and multiply across the final dimension

## Initialising

### Manual Setup
We will use Pr:YSO as an example as it has relatively simple but still pedagogically relevant parameters. As a Kramers ion it has effective electronic spin 0, a nuclear spin of 5/2, an anisotropic nuclear g tensor, and a quadrupole effect. All parameters come from Fraval, 2004

We first define our various parameters
```python
Espin=0
Ispin=5/2

#Natural units of the program are GHz, T and radians so conversions must be made
M=np.matrix(np.diag([2.86,3.05,11.56]))*10*1E6*h
Q=np.matrix(np.diag([-0.5624,0.5624,4.4450]))*1E6*h
alpha,beta,gamma=np.array([-99.8,55.8,-40])*np.pi/180

#No convention for rotation is specified in the data source, but ZYZ is typical
M=spin.tensorRotation(M,[alpha,beta,gamma],conv='ZYZ')
Q=spin.tensorRotation(Q,[alpha,beta,gamma],conv='ZYZ')

#Instantiates the class and sets up dimensions and spin matrices
ground = spin.cSpinHamiltonian(Espin,Ispin)

#pass these matrices to the class
ground.setM(M)
#This will calculate and store the interaction in our static hamiltonian
ground.quadrupoleInteraction(Q)
#Access the static hamiltonian it big so lets just print its shape
print(ground.H.shape) 

```
### Import Parameters
Parameters can also be imported using a set of parameters stored in a YAML, this basically runs through the above depending on what values are in the file. 

> [!CAUTION]
> In order to implement easy conversion and matrix operations, the `eval()` function is called on most passed inputs. Do not run this on files without first checking they aren't doing something weird.

We will use the [Pr_YSO.yml](./ion_params/Pr_YSO.yml), a file that contains the same parameters. The spin values can be overridden on the import for ease of including multiple isotopes.
```python
#Much simpler
groundI=spin.hamilFromYAML('./ion_params/Pr_YSO.yml')
#We can also change the spin values if we want to look at other isotopes
    #Lets pretend Pr has another stable isotope
groundN=spin.hamilFromYAML('./ion_params/Pr_YSO.yml',IOveride=3/2)

```
#### Full import Parameters:
> [!Note] 
> There are multiple conventions used for quadrupole tensors 
> - `mu`, this assumes an isotropic value and $g_{n}=\mu I_{3}$
> - `gn`, this can be anisotropic $g_{n}=g_{n}$
> - `M`, some papers include nuclear magneton in the tensor $g_n=M/\mu_{n}$
> In manual mode we can use `setgN` or `setM` 
> If multiple are defined in the YAML it takes the first in the order `M,gn,mu`

```yaml
Rotation:
    Rot: (str) The Euler rotation convention
Hyperfine:
    A: (eval) The A matrix
    A_rot: (eval) The array of alpha,beta,gamma to rotate by
Quadrupole:
    Q: (eval) The Q matrix
    Q_rot: (eval) The array of alpha,beta,gamma to rotate by
E_zeeman:
    g: (eval) The g matrix
    g_rot: (eval) The array of alpha,beta,gamma to rotate by
N_zeeman:
    M: (eval) The M matrix
    M_rot: (eval) The array of alpha,beta,gamma to rotate by
    or
    g: (eval) The g matrix
    g_rot: (eval) The array of alpha,beta,gamma to rotate by
    or
    mu: (eval) The element mu

```
## Running Code
From here we likely want to evaluate at a range of magnetic field values. These should form a (3,k) array. If we aren't doing anything too fancy we can pass this to the dynamic N setup by the setup process.

We can also define H ourselves, however there is the catch that the static hamiltonian will always be (dim,dim), whereas the dynamic will have the addition of the sweep dimension (k,dim,dim). We can force any static terms to match using `np.array(Hs)[np.newaxis,...]`. We first convert it from a matrix to an ndarray allowing us to exceed the two dimension limit. The new axis slicing then handles the rest.
```python
#generate a set of fields from zero to 100mT
Bz=np.linspace(0,200*1E-3,200)
#Unit vector along z
uvec=np.matrix([0,0,1]).T

#convert it to a vector along the our axis
B=uvec*Bz

#If we are doing simple things dynamicH suffices
    # It automatically adds the static hamiltonian if static=True
H=groundI.dynamicH(B)

#This actually conceals a few important things so to write it out in full
    #ground.H contains our static hamiltonian, 
    #it doesn't know about the dimension of the magnetic field
    #Using np.newaxis we can increase the dimension to match the zeeman data
H=np.array(groundI.H)[np.newaxis,...]-groundI.nuclearZeeman(B)

#get the eigen values F and Vectors
F,V=groundI.getEigFreq(H)

#Dimension convention makes is to aid such plotting
plt.plot(Bz,F)
plt.show()

#Calculate the transitions, noting that our transitions always lie across axis 1
T=spin.eachElemFunc(F,F,ax=1)
plt.plot(Bz,T)
plt.show()
```
We can also calculate gradients and curvatures, though these are more annoying to visualise.
```python
#This uses the default calculated hamiltonian derivative based on dynamicH
df=groundI.firstDerivative(V)

#To be explicit we just calculate the gradient as the value at the Identity
    #This generates the three x,y,z hamiltonian elements that are then separated
dH=-groundI.nuclearZeeman(np.eye(3))
dH/=h*1E9 #Remember to convert to GHz

print("dH Shape: ",dH.shape)

df=groundI.firstDerivative(V,dH=dH)
plt.plot(Bz,df.real*uvec)
plt.show()


ddf=groundI.curvatureCalculation(V,F)
#plot the maximum eigenvalue as a single data point proxy
Cmax=np.max(np.linalg.eig(ddf)[0],axis=-1)
plt.plot(Bz,Cmax.real)
plt.show()
```
# search.py
These functions are intended to take parts of the spin hamiltonian and run some form of optimisation on them

# Major Changelog:
## October 2025

This makes quite a few changes that are likely to not be backwards compatible

- All functions are now vectorised with respect to passed magnetic field
    - Convention is field values as the first dimension, transition as the last
    - Some old functions still exist but have been moved or renamed and a depreciation warning added
        - They will never actually be depreciated
- Added docstrings and type hints to all major functions
    - I did get somewhat lazy but generally if it doesn't have a docstring think twice about using it.
- `tensorRotation` 
    - `str` has been changed to `conv` as in convention as it previously overwrote the built in keyword
- `eachElemFunc`
    - Changed `ax` to `axis` to be closer to numpy
- `genAmatrix`->`genDerivMatrix`
    - Clearer name now that I know what this is actually doing
- `firstOrderEnergySensitivty`-> `gradient`
- `CurvatureCalculation`->`curvature`
- Swapped `QuadrupoleAlt` and `Quadrupole`

