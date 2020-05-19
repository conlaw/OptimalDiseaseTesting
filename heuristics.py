import numpy as np
import random
from copy import copy
from simulation_helpers import generateAdjacenyMatrix
from optimal_policy import computeTransY

''' 
Optimal Policy 
-
'''

def optimalPolicy(y, Q, A, L, args = {}):
    '''
    Heuristic that tests uninfected people randomly
    '''
    L = min(L, Q.sum())
    
    if 'actions' not in args:
        raise Exception("Missing Optimal Actions for simulating it")
    
    M = args['M'] if 'M' in args else 20
    
    y_disc = (y*M).astype(np.int32) / M
    
    name = str(Q)+'-'+str(y_disc)+'-'+str(args['h'])
    
    return args['actions'][name]

'''
Baseline Heuristics
-Simple naif heuristics
'''
def selectRandom(y, Q, A, L, args = {}):
    '''
    Heuristic that tests uninfected people randomly
    '''
    L = int(min(L, Q.sum()))
    
    tests = np.zeros(len(Q))
    tests[np.random.choice(np.nonzero(Q)[0], L, replace = False)] = 1
    
    return tests


'''
Graph-Based Heuristics
-These heuristics test based solely off of graph structure
'''
def highRisk(y, Q, A, L, args = {}):
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


def highConnection(y, Q, A, L, args = {}): #LH added in args = {}
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

def highBelief(y, Q, A, L, args = {}):
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


def sampleBelief(y, Q, A, L, args = {}, softmax = False):
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
def highBeliefRisk(y, Q, A, L, args = {}, consider_removed = False):
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


def sampleBeliefRisk(y, Q, A, L, args = {}, consider_removed = False, softmax = False):
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


