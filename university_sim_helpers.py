#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import operator as op
import math
import numpy as np
from collections import Counter

import networkx as nx
from networkx.algorithms import node_classification
import matplotlib.pyplot as plt


# In[ ]:


def AdjacencyUniWMajor(N,C,mu,sd,M,plug,perc):
    '''Helper Function
       generate an adjacency matrix imitating the university social network'''
    #N - number of students
    #C - number of classes
    #mu - avg number of students per class
    #sd - sd dev of students per class
    #M - number of majors. Here in second method, we incorporate the ideas of having majors 
        #and different students have priority taking their major core courses
    #plug - controls the avg percentage of students in one specific major, rest will be divided equally across
    #perc - the percentage of students of each class that are limited to major students

    floorfunc=np.vectorize(math.floor)
    maxfunc=np.vectorize(max)
    Cn=maxfunc(floorfunc(np.random.normal(mu,sd,size=(C,1))),0) #sampling class size N(mu,sd^2),capped at N, min at 0

    mp=np.append(np.repeat((1-plug)/(M-1),M-1),plug) #probability vector for each student to be one of the four major
    MS=np.random.multinomial(N,mp) #number of studetns randomly allocated to the M majors
    cumMS=np.cumsum(MS)
    cumMS=np.insert(cumMS,0,0)

    #create a vector of majors, to be used later when plotting graph/network
    MSmajor=np.repeat(range(M),MS) #each int value denotes a major

    CS=np.random.multinomial(C,mp) #number of classes opened by the M majors respectively
    cumCS=np.cumsum(CS)
    cumCS=np.insert(cumCS,0,0)

    A2temp=np.zeros((N,C))

    for i in range(C): #loop through each class
        for j in range(M):
            if i<cumCS[j+1] and i>=cumCS[j]:
                inMajorS=min(int(Cn[i]*perc),MS[j]) #assume a centrain percentage are must be in-major students, 
                                                    #controlled by perc
                                                    #but capped at the max possible students taking this major
                outMajorS=Cn[i]-inMajorS

                currentInMajorPopulation=range(cumMS[j],cumMS[j+1])

        currentInMajorS=np.random.choice(currentInMajorPopulation,inMajorS,replace=False)
        currentOutMajorPopulation=np.delete(range(N),currentInMajorS)
        #for the rest of the class, it can still be selected from in-major students.
        #thus, if we set perc=0, it is equivalent to no major restriction
        currentOutMajorS=np.random.choice(currentOutMajorPopulation,outMajorS[0],replace=False)

        currentS=np.append(currentInMajorS,currentOutMajorS)
        A2temp[currentS,i]=1

    A2=np.matmul(A2temp,np.transpose(A2temp))
    np.fill_diagonal(A2, 0)
    A2[A2>0]=1

    return A2,MS,CS


# In[ ]:


def plotUniversityNetwork(A2,M,MS):
    '''Helper Function
       plot the simulated university network, but exclude any isolated nodes from the graph'''
    N=np.shape(A2)[0]
    excludeNode2=sum(A2)==0
    excludeNodeIndex2=np.where(excludeNode2)[0]

    excludeMSmajor2=np.delete(MSmajor,excludeNodeIndex2)
    excludeMScounter=Counter(excludeMSmajor2)
    excludeMS=[0]
    for i in range(M):
        excludeMS=np.append(excludeMS,[excludeMScounter[i]],0)

    excludeMScumsum=np.cumsum(excludeMS)

    nodeLeft2=sum(excludeNode2==False)
    if nodeLeft2<N:
        A2exclude=np.reshape(A2,(N,N))
        A2exclude=np.delete(A2exclude,excludeNodeIndex2,0)
        A2exclude=np.delete(A2exclude,excludeNodeIndex2,1)

        G2v2=nx.Graph()
        G2v2.add_nodes_from([i for i in range(nodeLeft2)])

        G2v2edgeSame=np.empty((0,2))
        G2v2edgeDiff=np.empty((0,2))
        for i in range(nodeLeft2):
            major1=excludeMSmajor2[i]
            for j in range(i, nodeLeft2):
                major2=excludeMSmajor2[j]
                if A2exclude[i,j]==1:
                    G2v2.add_edge(i,j)
                    if major1==major2: #same major, color coded edge
                        G2v2edgeSame=np.append(G2v2edgeSame,np.array([[i,j]]),0)
                    else:
                        G2v2edgeDiff=np.append(G2v2edgeDiff,np.array([[i,j]]),0)

                else: continue

        pos2v2 = nx.spring_layout(G2v2)  # positions for all nodes
        colorVector=['b','r','m','k','g','brown','orange','cyan','burlywood','wheat','gold','goldenrod','darkgoldenrod','darkkhaki','khaki','forestgreen','limegreen','darkgreen','lightseagreen','teal','darkcyan','cadetblue','slategrey','cornflowerblue','royalblue','slateblue','blueviolet','indigo','violet','darkmagenta','mediumblue'] #create a long vector to accomodate more major selection
        #approx 30 colors, i.e. major numbers should not exclude the length of colorVector
        
        for j in range(M):
            nx.draw_networkx_nodes(G2v2, pos2v2, nodelist=range(excludeMScumsum[j],excludeMScumsum[j+1]), node_color=colorVector[j])

        nx.draw_networkx_edges(G2v2, pos2v2, edgelist=G2v2edgeSame.tolist(),width=1.0, alpha=0.5,edge_color='black')
        nx.draw_networkx_edges(G2v2, pos2v2, edgelist=G2v2edgeDiff.tolist(),width=1.0, alpha=0.5,edge_color='orange')

