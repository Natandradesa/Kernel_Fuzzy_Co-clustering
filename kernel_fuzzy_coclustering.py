import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform
#from scipy.stats.mstats import gmean

#------------------------------------- Helper functions -------------------------------------#

def gmean(x):
    return (x ** (1/len(x))).prod()

# Helper functions for all algortihms
def random_U(shape, random_state = None):
    
    if type(shape) == int:
        np.random.seed(random_state)
        h = np.random.random(shape)
        return h/h.sum()
    else:
        np.random.seed(random_state)
        h = np.random.random(shape)
        return h/h.sum(axis= 1).reshape(-1,1) 

def initial_prototypes(X, Um, Vn):
    '''
    X is the dataset 
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    '''
    K, H = Um.shape[1], Vn.shape[1]
    G = np.zeros((K,H))
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            wh = (uk * np.full_like(X,1) * vh)
            G[k,h] = np.average(a = X, weights = wh )            
    return G

def gaussian_kernel_array(X, G, sig2):
    '''
    X is the dataset
    G is the prototypes of the co-clusters
    sig2 is the parameters of the gaussian kernel
    '''
    n, p = X.shape
    K, H = G.shape
    const = 1/(2*sig2)
    KMs = np.zeros((K, H, n, p))
    for k in range(K):
        for h in range(H):
            KMs[k,h] = np.exp( -const * (X - G[k,h]) ** 2 )     
    return KMs

def getU(D, U, V, m, n):
    '''
    D is the array of the distances between the dataset and the prototype matrix
    U is the fuzzy matrix of the objects into K cluster on the current iteration
    V is the partiton of variables into H cluster
    m and n are the fuzzyness parameters
    '''

    Vn = V**n
    Vnt = np.transpose(Vn)
    vhn = Vn.sum(axis = 0)
    exponent = (1/(m-1))
    K,N = D.shape[0], D.shape[2]
    for i in range(N):
        Di = D[:,:,i,:] 
        Dvi = ((Vnt*Di).sum(axis = 2)) * (1/vhn)
        Dvi = np.where(np.isinf(Dvi),0,Dvi)
        Dvi = np.where(np.isnan(Dvi),0,Dvi)
        Dik = Dvi.sum(axis = 1)
        inv_dk = (1/Dik) ** exponent
        idx_inf =  np.where(np.isinf(inv_dk))[0]
        n_inf = len(idx_inf)
        if n_inf < K:
            idx = np.where(~np.isinf(inv_dk))[0] # values different of inf
            den = inv_dk[idx].sum()
            if den > 0.0 and den != np.inf:
                if n_inf == 0:
                    U[i] = inv_dk/inv_dk.sum()
                else:
                    inv_dk_new = inv_dk[idx]
                    sum_previous = U[i,idx_inf].sum()
                    const = 1 - sum_previous
                    # Computationally is possible that const < 0, but mathmatically const must be non-negative
                    if const >= 0:
                        U[i,idx] = const * (inv_dk_new/inv_dk_new.sum())
                    else:
                        U[i,idx] = 0
    return U 

def getV(D, U, V, m, n):
    '''
    D is the array of the distances between the dataset and the prototype matrix
    U is the fuzzy matrix of the objects into K cluster 
    V is the partiton of variables into H cluster on the current iteration
    m and n are the fuzzyness parameters
    '''
    Um = U**m
    Umt = np.transpose(Um)
    ukm = Um.sum(axis = 0)
    exponent = (1/(n-1))
    H,P = D.shape[1], D.shape[3]
    Dm = np.moveaxis(D,0,1)
    for j in range(P):
        Dj = Dm[:,:,:,j] 
        Duj = ((Umt*Dj).sum(axis=2)) * (1/ukm)
        Duj = np.where(np.isinf(Duj),0,Duj)
        Duj = np.where(np.isnan(Duj),0,Duj)
        Djh = Duj.sum(axis = 1)
        inv_dh = (1/Djh) ** exponent
        idx_inf =  np.where(np.isinf(inv_dh))[0]
        n_inf = len(idx_inf)
        if n_inf < H:
            idx = np.where(~np.isinf(inv_dh))[0]
            den = inv_dh[idx].sum()
            if den > 0.0 and den != np.inf:
                if n_inf == 0:
                    V[j] = inv_dh/inv_dh.sum()
                else:
                    inv_dk_new = inv_dh[idx]
                    sum_previous = V[j,idx_inf].sum()
                    const = (1 - sum_previous)
                    if const >= 0:
                        V[j,idx] = const * (inv_dk_new/inv_dk_new.sum())
                    else:
                        V[j,idx] = 0.0
    return V

def computeJ(D, Um, Vn):
    '''
    D is the array of the distance between the dataset and the prototype matrix
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    '''
    ukm = Um.sum(axis = 0)
    vhn = Vn.sum(axis = 0)
    K,H = D.shape[:2]
    Jc = 0
    for k in range(K):
        if ukm[k] != 0:
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                if vhn[h] != 0:
                    vh = (Vn[:,h]).reshape((1,-1))
                    const = (1/ukm[k]) * (1/vhn[h])
                    if ~np.isinf(const):
                        Dkh = const * ( (uk * D[k,h] * vh).sum() )
                        Jc += Dkh.sum()
    return Jc

# Helper functions for the Gaussian Kernel Fuzzy Double Kmeans(GKFDK)

def prototypes(X, KMs, Um, Vn, G, ld):
    '''
    X is the dataset 
    KMs is the array of kernel between X and G
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    G is the current prototype matrix
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    K,H = G.shape
    P = X.shape[1]
    
   
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            KMkh = uk * KMs[k,h]* vh 
            if KMkh.sum() > ld:
                G[k,h] = np.average(a = X, weights = KMkh )            
    return G

# Helper functions for the Gaussian Kernel Fuzzy Double Kmeans with local product restriction (GKFDK_LP)

def get_weights_local_prod(D, Um, Vn, W, ld):
    '''
    D is the array of distance between X and the co-closters
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    W is the weights of the variables at previous iteration
    ld is the lowest denominator, to avoid indeterminacy.
    '''
   
    vhn = Vn.sum(axis = 0)
    K,H,N,P = D.shape
    for k in range(K):
        uik = (Um[:,k]).reshape((-1,1))
        DN = np.zeros((H,P))
        for h in range(H):
            vjh = (Vn[:,h])
            DN[h] = (uik * D[k,h]).sum(axis = 0) * (vjh/vhn[h])
        DN = np.where(np.isinf(DN),0,DN)
        DHN = DN.sum(axis = 0)
        jless = np.where(DHN<= ld)[0]
        nless = len(jless)
        if nless == 0:
            W[k] =  gmean(DHN)/DHN
        else:
            jupper = np.where(DHN > ld)[0]
            den = np.prod(W[k][jless])
            if den > ld and nless < P:
                const = (1/den)**(1/(P-nless))
                if  ~np.isnan(const) and ~np.isinf(const):
                    sum_upper = DHN[jupper]
                    W[k][jupper] = (const*gmean(sum_upper)/sum_upper)
   
    return W

# Helper functions for the Gaussian Kernel Fuzzy Double Kmeans with global product restriction (GKFDK_GP)

def get_weights_global_prod(D, Um, Vn, W, ld):
    '''
    D is the array of distance between X and the co-closters
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    W is the weights of the variables at previous iterations
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    vhn = Vn.sum(axis = 0)
    ukm = Um.sum(axis = 0)
    K,H,N,P = D.shape
    DK = np.zeros((K,P))
    for k in range(K):
        uik = (Um[:,k]).reshape((-1,1))
        DN = np.zeros((H,P))
        for h in range(H):
            vjh = (Vn[:,h])
            DN[h] = (uik * D[k,h]).sum(axis = 0) * (vjh/vhn[h])
        DN = np.where(np.isinf(DN),0,DN)
        DK[k] = DN.sum(axis = 0) * (1/ukm[k])
    DK = np.where(np.isinf(DK),0,DK)
    DJ = DK.sum(axis = 0)
    if (DJ > ld).all():
        W =  gmean(DJ)/DJ
    else:
        jless = np.where(DJ<= ld)[0]
        jupper = np.where(DJ > ld)[0]
        nless = len(jless)
        const = (1/np.prod(W[jless]))**(1/(P-nless))
        if const > ld and  ~np.isnan(const) and ~np.isinf(const):
            sum_upper = DJ[jupper]
            #print(sum_upper)
            W[jupper] = (const*gmean(sum_upper)/sum_upper)
    return W

# Helper functions for both GKFDK_LP and GKFDK_GP algorithms

def prototypes_adaptive(X, KMs, W, Um, Vn, G, ld):
    '''
    X is the dataset 
    KMs is the array of kernel between X and G
    W is the weights of the variables
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    G is the current prototype matrix
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    K,H = G.shape
    P = X.shape[1]
    
    if np.ndim(W) == 1:
        W = np.tile(W,K).reshape((K,P))
   
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        wk = W[k].reshape((1,-1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            KMkh = uk * KMs[k,h]* vh * wk
            if KMkh.sum() > ld:
                G[k,h] = np.average(a = X, weights = KMkh )            
    return G

def D_adaptive(D,W):

    if np.ndim(W) == 1:
        return (D * W)
    else:
        Da = np.zeros_like(D)
        K = D.shape[0]
        for k in range(K):
            Da[k] = D[k] * W[k]
        return Da


#------------------------------------- Main functions -------------------------------------#

def GKFDK(X, K, H, m, n, sig2, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):
    '''
    X is a dataset
    K is the number of cluster for the objetcs
    H is the numer of cluster for the variables
    m and n are the fuzzyness pararmeters
    sig2 is the hyperparameter of the gaussian kernel
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    U = random_U((N,K),random_state)
    V = random_U((P,H),random_state)
    Uinit = U.copy()
    Vinit = V.copy()
    Um = U**m
    Vn = V**n
    G = initial_prototypes(X, Um, Vn)
    KM = gaussian_kernel_array(X, G, sig2)
    D = (2-2*KM)
    J = 0
    Jlist = []

    # iterative step
    t = 1
    while True:
    
        U = getU(D, U, V, m, n)
        V = getV(D, U, V, m, n)
        Um = U**m
        Vn = V**n
        G = prototypes(X, KM, Um, Vn, G, lowest_denominator)
        KM = gaussian_kernel_array(X, G, sig2)
        D = (2 - 2*KM)
        Jcurr = J
        J = computeJ(D, Um, Vn)
        if np.isnan(J):
            return "Error: J must be a real value, not a Nan "
        Jlist.append(J)

        if np.abs(Jcurr - J) < epsilon or t > T:
            break
        else:
            t = t + 1

    #The following part of the code checks if there have been any technical problems with the 
    # values of m and n. If these values are too small, so that 1/m-1 or 1/n-1 are computationally 
    # infinite. The matrices U and V are not updated. Therefore, these values are not appropriate.s

    if (Uinit == U).all() and (Vinit == V).all():
        print("U and V didn't change")
        chg = False
    elif (Uinit == U).all():
        print("U didn't change")
        chg = False
    elif (Vinit == V).all():
        print("V didn't change")
        chg = False
    else:
        #print("all right")
        chg = True

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G, 'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t, 'change':chg}

def GKFDK_LP(X, K, H, m, n, sig2, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):
    '''
    X is a dataset
    K is the number of cluster for the objetcs
    H is the numer of cluster for the variables
    m and n are the fuzzyness pararmeters
    sig2 is the hyperparameter of the gaussian kernel
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    U = random_U((N,K),random_state)
    V = random_U((P,H),random_state)
    Uinit = U.copy()
    Vinit = V.copy()
    Um = U**m
    Vn = V**n
    W = np.ones((K,P))
    G = initial_prototypes(X, Um, Vn)
    KM = gaussian_kernel_array(X, G, sig2)
    D = 2-2*KM
    Dj = D
    J = 0
    Jlist = []

    # iterative step
    t = 1
    while True:

        U = getU(Dj, U, V, m, n)
        V = getV(Dj, U, V, m, n)
        Um = (U ** m)
        Vn = (V ** n)
        W = get_weights_local_prod(D, Um, Vn, W, lowest_denominator)
        G = prototypes_adaptive(X, KM, W, Um, Vn, G, lowest_denominator)
        KM = gaussian_kernel_array(X, G, sig2)
        D = (2 - 2*KM)
        Dj = D_adaptive(D, W)
        Jcurr = J
        J = computeJ(Dj, Um, Vn)
        if np.isnan(J):
            return "Error: J must be a real value, not a Nan "
        Jlist.append(J)

        if np.abs(Jcurr - J) < epsilon or t > T:
            break
        else:
            t = t + 1
    
    #The following part of the code checks if there have been any technical problems with the 
    # values of m and n. If these values are too small, so that 1/m-1 or 1/n-1 are computationally 
    # infinite. The matrices U and V are not updated. Therefore, these values are not appropriate.

    if (Uinit == U).all() and (Vinit == V).all():
        print("U and V didn't change")
        chg = False
    elif (Uinit == U).all():
        print("U didn't change")
        chg = False
    elif (Vinit == V).all():
        print("V didn't change")
        chg = False
    else:
        #print("all right")
        chg = True

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    W = pd.DataFrame(W, index = knames )
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G, 'W':W, 'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t, 'change':chg}

def GKFDK_GP(X, K, H, m, n, sig2, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):
    '''
    X is a dataset
    K is the number of cluster for the objetcs
    H is the numer of cluster for the variables
    m and n are the fuzzyness pararmeters
    sig2 is the hyperparameter of the gaussian kernel
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    U = random_U((N,K),random_state)
    V = random_U((P,H),random_state)
    Uinit = U.copy()
    Vinit = V.copy()
    Um = U**m
    Vn = V**n
    W = np.ones(P)
    G = initial_prototypes(X, Um, Vn)
    KM = gaussian_kernel_array(X, G, sig2)
    D = 2-2*KM
    Dj = D
    J = 0
    Jlist = []

    # iterative step
    t = 1
    while True:

        U = getU(Dj, U, V, m, n)
        V = getV(D, U, V, m, n)
        Um = (U ** m)
        Vn = (V ** n)
        W = get_weights_global_prod(D, Um, Vn, W, lowest_denominator)
        G = prototypes_adaptive(X, KM, W, Um, Vn, G, lowest_denominator)
        KM = gaussian_kernel_array(X, G, sig2)
        D = (2 - 2*KM)
        Dj = D_adaptive(D, W)
        Jcurr = J
        J = computeJ(Dj, Um, Vn)

        if np.isnan(J):
            return "Error: J must be a real value, not a Nan "
            
        Jlist.append(J)

        if np.abs(Jcurr - J) < epsilon or t > T:
            break
        else:
            t = t + 1


    #The following part of the code checks if there have been any technical problems with the 
    # values of m and n. If these values are too small, so that 1/m-1 or 1/n-1 are computationally 
    # infinite. The matrices U and V are not updated. Therefore, these values are not appropriate.

    if (Uinit == U).all() and (Vinit == V).all():
        print("U and V didn't change")
        chg = False
    elif (Uinit == U).all():
        print("U didn't change")
        chg = False
    elif (Vinit == V).all():
        print("V didn't change")
        chg = False
    else:
        chg = True

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G, 'W':W, 'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t, 'change': chg}

