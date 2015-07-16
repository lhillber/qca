#!/usr/bin/python3
import numpy as np
import networkx as nx

#Scripted by David Vargas
#---------------------------------------------
#Convention if directed networks are introduced,
#Aij=1 if there is an edge from j to i. This
#means one would sum over row i to get the total
#number of incoming edges to node i.
#Here I am summing over rows.
def strengths(mutualinformation):
    #Return the strength of all nodes in the network in order.
    return np.sum(mutualinformation,axis=1)
def density(matrix):
    #Calculates density, also termed connectance in some
    #literature. Defined on page 134 of Mark Newman's book
    #on networks.
    l=len(matrix)
    lsq=l*(l-1)
    return sum(sum(matrix))/lsq
def clustering(matrix):
    #Calculates the clustering coefficient
    #as it is define in equation 7.39 of
    #Mark Newman's book on networks. Page 199.
    l=len(matrix)
    matrixcube=np.linalg.matrix_power(matrix,3)
    matrixsq=np.linalg.matrix_power(matrix,2)
    #Zero out diagonal entries. So we do not count edges as
    #connected triples.
    for i in range(len(matrixsq)):
        matrixsq[i][i]=0
    denominator=sum(sum(matrixsq))
    numerator=np.trace(matrixcube)
    if denominator<1e-14:
        return 0.
    else:
        return numerator/denominator
def disparity(matrix):
    #Calculates the average disparity of a network under
    #the assumption that it is completely connected.
    #Disparity defined on page 199 of doi:10.1016/j.physrep.2005.10.009 
    #Equation 2.39
    l=len(matrix)
    numerator=sum(matrix**2)
    denominator=sum(matrix)**2
    logos=denominator>1E-14
    numerator=numerator[logos]/(l-1)
    denominator=denominator[logos]
    return sum(numerator/denominator)
def maxdisparity(matrix):
    l=len(matrix)
    numerator=sum(matrix**2)
    denominator=sum(matrix)**2
    logos=denominator>1E-14
    numerator=numerator[logos]
    denominator=denominator[logos]
    ys=sum(numerator/denominator)
    return np.max(ys)
def mindisparity(matrix):
    l=len(matrix)
    numerator=sum(matrix**2)
    denominator=sum(matrix)**2
    logos=denominator>1E-14
    numerator=numerator[logos]
    denominator=denominator[logos]
    ys=sum(numerator/denominator)
    return np.min(ys)

#NetworkX additions
#---------------------------------------------
def distance(mutualinformation):
    #Initialize array
    length=len(mutualinformation)
    thisdistance=np.zeros((length,length))
    #If an element has value less than (10^-14) in absolute value,
    #then treat it as of value (10^-16), set its distance
    #to 10^16. Otherwise set it mij^(-1).
    for i in range(length):
        for j in range(length):
            if np.abs(mutualinformation[i,j])<=1E-14:
                thisdistance[i,j]=np.power(10.,16)
            else:
                thisdistance[i,j]=np.power(mutualinformation[i,j],-1)
    return thisdistance
def geodesic(distance,i,j):
    #Initialize networkx graph object
    #NetworkX indexing starts at zero.
    #Ends at N-1 where N is the number of nodes.
    latticelength=len(distance)
    G=nx.Graph(distance)
    #Use networkx algorithm to compute the shortest path from
    #the first lattice site to the last lattice site.
    pathlength=nx.shortest_path_length(G,source=i-1,target=j-1,weight='weight')
    if pathlength>np.power(10.,15):
        return np.nan
    return pathlength
def harmoniclength(distance):
    #page 11, equation 2 The Structure and Function of Complex Networks
    #If the geodesic distance between two nodes is a number then 
    #append it to alist to include it in the sum.
    l=len(distance)
    factor=1./(0.5*l*(l-1))
    alist=[]
    for i in range(1,len(distance)+1):
        for j in range(i+1,len(distance)+1):
            geo=geodesic(distance,i,j)
            if not np.isnan(geo):
                alist.append(1./geo)
    if sum(alist)==0:
        return 0
    else:
        return factor*sum(alist)

def strengthdist(mutualinformation,bincount):
    #Compute the weighted analog of a degree distribution.
    strengths=nm.strengths(mutualinformation)
    maxinfo=np.max(strenghts)
    mininfo=np.min(strenghts)
    return(np.histogram(mininfo,maxinfo,bin=np.linspace(mininfo,maxinfo,bincount+1)))




