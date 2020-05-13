import numpy as np
#Version where y is belief state

def V(Q, y, h, done, p, q, L, A, V_saved, M = 20):
    '''
    Recursive implementation of bellman operator
    '''
    #Convert y from discretized version to real_valued
    y = y/M
    
    #If epidemic is over, return number of infected
    if done or h == 0:
        return y.sum()

    #Check if we already computed this
    name = str(Q)+str(y)+str(done)+str(p)+str(q)
    if name in V_saved:
        return V_saved[name]

        
    #Compute R (# infected people healthy people are connected to)
    R = computeR(Q,y,A)
    
    #Generate list of possible testing strategies for this Q
    tests = generatePossibleTests(Q,L)
    
    #Compute value
    V_saved[name] = np.max([computeQ(test, Q, y, h, R, p, q, L, A, V_saved, M) for test in tests])
    
    return V_saved[name]

def generatePossibleTests(Q, L):
    '''
    Helper function to generate list of all possible tests
    '''
    tests = None
    #Adjust L to be min of number of tests or number of people that we can test
    L = min(L, Q.sum())
    
    #Iteratively build up possible
    for i in range(len(Q)):
        
        if tests is not None:
            base = np.ones(tests.shape[0]).reshape((tests.shape[0],1)).astype(np.bool)
        
        #If node is removed from graph then we can't test it
        if Q[i] == 0:
            if tests is None:
                tests = np.array([False]).reshape((1,1))
            else:
                tests = np.hstack([tests,  ~base])
        #If node is in graph, we can
        else:
            if tests is None:
                tests = np.array([False,True]).reshape((2,1))
            else:
                tests = np.vstack([np.hstack([tests,~base]), np.hstack([tests,base])])
        
        #Only keep valid tests
        tests = tests[tests.sum(axis=1) <= L]
    
    #return valid tests (assume will test all we can)
    return tests[tests.sum(axis=1) == L]        
        
def computeQ(test, Q, y, h, R, p, q, L, A, V_saved, M):
    '''
    Helper function to compute q value for different testing procedures
    '''
    
    #Compute new y and possible new Q's and their probability
    y_n = computeTransY(y, R, p)
    Q_s, Q_pr = computeTransQ(Q, y, test, q)
    
    #initialize constants
    val = 0
    self_prob = 0
    
    for Q_n in zip(Q_s, Q_pr):
        #If we transition to where we currently are, save that prob but dont try to do recursion
        if np.array_equal(y_n, y) and np.array_equal(Q_n[0],Q):
            self_prob += Q_n[1]
            continue
        
        #Add recursion term to bellman operator
        val += Q_n[1]*V(Q_n[0],
                               np.floor(y_n*M).astype(np.int32),
                               h - 1,
                               checkDone(computeR(Q_n[0], y_n, A), y_n, Q_n[0]),
                               p, q, L, A,
                               V_saved,
                               M)
      
    return val/(1-self_prob)

def computeR(Q,y,A):
    '''
    Helper function to compute how many infected people a healthy person is connected to
    '''

    R = np.zeros(len(Q))
    #Do matrix multiplication of DIAG(Q)*A*DIAG(Q)*Y
    R[Q] = np.matmul(A[Q,:][:,Q],y[Q])
    
    return R

def checkDone(R,y,Q, thresh = 0):
    '''
    Helper function to check whether or not an epidemic is over
    '''
    #Epidemic is over if no healthy people are connected to someone sick
    #Or there's nobody health left on the graph
    return max(R) <= thresh or max(1-y[Q]) <= thresh

def computeTransY(y, R, p):
    '''
    Function to compute possible Y vectors (whether or not points are infected) 
    and their relative transition probabilities
    '''
    return np.clip(y + (1-y)*(1 - (1-p)**R),0,1)

def computeTransQ(Q, y, tests, q, thresh = 0):
    '''
    Function to compute possible Q vectors (whether or not points are included graph) 
    and their relative transition probabilities
    '''
    
    arr = None
    pr = None
    p0 = q*y
    p0[tests] = y[tests]
    
    thresh = thresh
    
    #Iteratively build up new q, and their associated prob
    #Warning: Exponential in n :(
    #Can make it exponential in n - n_inf but not sure how much that'll really help
    for i in range(len(Q)):
        if arr is not None:
            base = np.ones(arr.shape[0]).reshape((arr.shape[0],1)).astype(np.bool)

        if Q[i] == 0:
            if arr is None:
                arr = np.array([False]).reshape((1,1))
                pr = np.array([1])
            else:
                arr = np.hstack([arr,  base*0])
        else:
            if arr is None:
                arr = np.array([False,True]).reshape((2,1))
                pr = np.array([p0[i], (1- p0[i])])
            else:
                arr = np.vstack([np.hstack([arr,~base]), np.hstack([arr,base])])
                pr = np.concatenate([pr*p0[i], pr*(1-p0[i])])
        
        arr = arr[pr > thresh,:]
        pr = pr[pr > thresh]
    return arr, pr
