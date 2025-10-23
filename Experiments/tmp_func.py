def task(params,B):
    #print(params,flush=True)
    #global ground
    ground=params[0]
    print(ground,flush=True)
    #convert spherical Magnetic field to cartesian coords.
    #B = np.matrix([X,Y,Z]).T
    #Calculate our hamiltonian at this B Field
    HG = ground.Hfunc(B)#Hcalc(B)#ground.H+ground.nuclearZeeman(B)#/spin.muN

    #get the eigen frequencies and vectors at this B field
    FG,VG = ground.getEigFreq(HG)
    F=FG*1E3
    
    Fp = []
    #OS = np.zeros((ground.dim,3),dtype = np.csingle)
    for l in range(3):
        v = ground.firstOrderEnergySensitivity(VG,A[l])
        Fp.append(v)
    Fp = np.array(Fp).T
    #Fpp = ground.curvatureCalculation(As[0],As[1],As[2],VG,FG,indiv=False).T#*spin.muN
    Fpp = ground.curvatureCalculationNaive(A[0],A[1],A[2],VG,FG,indiv=False).T#*spin.muN
    
    return (F,Fp,Fpp)
