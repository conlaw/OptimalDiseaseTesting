import numpy as np
import random
from copy import copy
from simulation_helpers import generateAdjacenyMatrix
from optimal_policy import computeTransY


'''
Baseline Heuristics
-Simple naif heuristics
'''
def selectRandom(y, Q, A, L):
    '''
    Heuristic that tests uninfected people randomly
    '''
    L = min(L, Q.sum())
    
    tests = np.zeros(len(Q))
    tests[np.random.choice(np.nonzero(Q)[0], L, replace = False)] = 1
    
    return tests


'''
Graph-Based Heuristics
-These heuristics test based solely off of graph structure
'''
def highRisk(y, Q, A, L):
    '''
    Heuristic that tests those closest to those removed from the graph
    '''
    L = min(L, Q.sum())
    
    #Computee number of 'removed' people everyone is connected to
    R = np.matmul(A, 1-Q)
    R[1-Q] = 0 #If person is removed set 'removed' connections to 0
    
    tests = np.zeros(len(Q))
    tests[np.argsort(R)[::-1][:L]] = 1 #Test the top L most connected to removed nodes

    return tests


def highConnection(y, Q, A, L):
    '''
    Heuristic that tests those that are most connected on the graph
    '''
    L = min(L, Q.sum())
    
    #Compute number of people everyone is connected to
    R = np.matmul(A, np.ones(len(Q)))
    R[1-Q] = 0 #If person is removed set 'number of connections to 0
    
    tests = np.zeros(len(Q))
    tests[np.argsort(R)[::-1][:L]] = 1 #Test the top L most connected to removed nodes

    return tests


'''
Belief Heuristics
-These heuristics operate by selecting who to test based on our belief state
'''

def highBelief(y, Q, A, L):
    '''
    Heuristic that tests the L individuals with highest beliefs
    '''
    L = min(L, Q.sum())
    
    #Adjust y to remove nodes outside of graph
    y_sample = y.copy()
    y_sample[1-Q] = 0
    
    tests = np.zeros(len(Q))
    tests[np.argsort(y_sample)[::-1][:L]] = 1 #Test the top L nodes with highest beliefs
    return tests


def sampleBelief(y,Q,A,L, softmax = True):
    '''
    Heuristic that samples who to test by building a (softmax) distirbution over nodes
    '''
    L = min(L, Q.sum())
    
    #Adjust y to remove nodes outside of graph
    y_sample = y.copy()
    y_sample[1-Q] = 0
    
    #Build a distribution over nodes based on belief
    dist = np.exp(y_sample)/np.exp(y_sample).sum() if softmax else y/y.sum()
    
    tests = np.zeros(len(Q))
    tests[np.random.choice(np.arange(len(Q)), L, p = dist, replace = False)] = 1 #Test L nodes sampled from distribution
    return tests


'''
Hybrid Heuristics
-These heuristics operate based on a combination of graph and belief data
'''
def highBeliefRisk(y, Q, A, L, consider_removed = False):
    '''
    Heuristic that tests those that are most connected to the people most likely to be sick
    '''
    L = min(L, Q.sum())
    
    #Adjust Y to deal with removed nodes
    y_sample = y.copy()
    y_sample[1-Q] = 1 if consider_removed else 0 #Takes advantage of fact removed nodes = 1

    #Compute number of people everyone is connected to
    R = np.matmul(A+np.eye(A.shape[0]), y_sample) #Add Identity matrix so that we include belief on person themself
    R[1-Q] = 0 #If person is removed set risk to 0
    
    tests = np.zeros(len(Q))
    tests[np.argsort(R)[::-1][:L]] = 1 #Test the top L most connected to removed nodes

    return tests


def sampleBeliefRisk(y, Q, A, L, consider_removed = False, softmax = False):
    '''
    Heuristic that tests those that are most connected to the people most likely to be sick
    '''
    L = min(L, Q.sum())
    
    #Adjust Y to deal with removed nodes
    y_sample = y.copy()
    y_sample[1-Q] = 1 if consider_removed else 0 #Takes advantage of fact removed nodes = 1

    #Compute number of people everyone is connected to
    R = np.matmul(A+np.eye(A.shape[0]), y_sample) #Add Identity matrix so that we include belief on person themself
    R[1-Q] = 0 #If person is removed set risk to 0
    
    dist = np.exp(R)/np.exp(R).sum() if softmax else R/R.sum()

    tests = np.zeros(len(Q))
    tests[np.random.choice(np.arange(len(Q)), L, p = dist, replace = False)] = 1 #Test L nodes sampled from distribution

    return tests


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
    R[Q] = np.matmul(A[Q,:][:,Q], yReal[Q])

    infection = np.zeros(len(Q))
    infection[R > 0] = np.random.binomial(R[R>0].astype(np.int32), p, (R>0).sum()) > 0

    return np.clip(yReal + infection, 0, 1)


def sample(func, Q, h, p, q, L, A, n):
    
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
    y[start] = 1
    
    
    done = False 
    
    for m in range(h): 
        # done if no healthy ppl connected to someone sick or nobody healthy on graph
        R = np.zeros(len(Q))    
        R[Q] = np.matmul(A[Q,:][:,Q], yReal[Q])            
        done = (max(R) == 0) or (min(yReal[Q]) == 1)
        
        # else proceed 
        if not done:
            
            #get infected from the current state
            yReal = infection(yReal, Q, p, A)
            
            # test
            test = func(y, Q, A, L)
            
            # update Q
            Q = removeNodes(yReal, Q, test, q)
                        
            # update y
            y = computeTransY(y, R, p)
                
        # count the num of infected on the graph
        numInf[m+1] = int(y.sum())
 
    return numInf

