import numpy as np
import random
from copy import copy
from optimal_policy import computeTransY

def generateAdjacenyMatrix(n, p):
    '''
    Helper function to generate a sample adjaceny matrix where the prob of person i,j connecting is p
    '''
    
    #check for valid p
    if p < 0 or p > 1:
        raise Exception('Invalid p')
    
    #Adjusts p to deal with the fact that we're generating a symmetric matrix by doing mat + mat.T
    p_new = 1 - np.sqrt(1-p)
    
    #Compute random matrix with p' and zero out diagonal
    mat = np.random.binomial(1, p = p_new, size = (n,n))
    np.fill_diagonal(mat, 0)
    
    #Return symmetrized matrix
    return np.ceil((mat + mat.T)/2)

# Removed nodes from the graph based on testing and if the node is infected
def removeNodes(y, Q, test, q):
    
    # amounce the infected, announce == 0 if that person decides to announce
    announce = np.zeros(len(Q))
    announce[y.astype(np.bool)] = np.random.binomial(1,q,int(y.sum()))    
    
    # remove the person who has announced from the graph
    Q[announce.astype(np.bool)] = 0
    
    # test result according to the test we selected
    Q[np.multiply(y,test).astype(np.bool)] = 0
    
    return Q


# Infect the next generation of nodes
def infection(yReal, Q, p, A):
    
    R = np.zeros(len(Q))
    R[Q.astype(np.bool)] = np.matmul(A[Q.astype(np.bool),:][:,Q.astype(np.bool)], yReal[Q.astype(np.bool)])

    infection = np.zeros(len(Q))
    infection[R > 0] = np.random.binomial(R[R>0].astype(np.int32), p, (R>0).sum()) > 0

    return np.clip(yReal + infection, 0, 1)


def sample(func, Q, h, p, q, L, A, n, args = {}):
    
    # number of infection on the graph
    numInf = np.zeros(h+1)
    numInf[0] = 1
    
    # belief state
    y = np.ones(n)/n 
    
    # real infection progression
    yReal = np.zeros(n)
    
    # select the starting node
    start = random.randint(0,n-1) 
    yReal[start] = 1    
    
    done = False 
    
    for m in range(h): 
        # done if no healthy ppl connected to someone sick or nobody healthy on graph
        R = np.zeros(len(Q))    
        R[Q.astype(np.bool)] = np.matmul(A[Q.astype(np.bool),:][:,Q.astype(np.bool)], yReal[Q.astype(np.bool)])            
        done = (max(R) == 0) or (min(yReal[Q.astype(np.bool)]) == 1)
        
        # else proceed 
        if not done:
            
            #get infected from the current state
            yReal = infection(yReal, Q, p, A)
            
            # test
            args['h'] = h - m
            test = func(y, Q, A, L, args)
            
            # update Q
            Q = removeNodes(yReal, Q, test, q)
                        
            # update y
            y = computeTransY(y, R, p)
                
        # count the num of infected on the graph
        numInf[m+1] = int(y.sum())
 
    return numInf