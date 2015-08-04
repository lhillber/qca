#!/usr/bin/python
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
    #as it is defined in equation 7.39 of
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
    #if there are no closed paths of length
    #three the clustering is automatically
    #set to zero.
    if numerator==0.:
        return 0.
    else:
        return numerator/denominator
def disparity(matrix):
    #Disparity defined on page 199 of doi:10.1016/j.physrep.2005.10.009 
    #Equation 2.39, Here I take the average of this quantity over the
    #entire network
    l=len(matrix)
    numerator=sum(matrix**2)/l
    denominator=sum(matrix)**2
    #Check for zero denominator.
    if sum(denominator)==0.:
        return np.nan
    else: 
        return sum(numerator/denominator)
def disparitylattice(matrix):
    #Local disparity across the entire lattice
    #Must return logos so the user can determine
    #the nodes with non-zero disparity.
    l=float(len(matrix))
    numerator=sum(matrix**2)
    denominator=sum(matrix)**2
    if sum(denominator)==0.:
        return np.nan
    else:
        disparity=numerator/denominator
        nodes=np.array(range(1,int(l+1)))
        nodedict={}
        for i in range(len(nodes)):
            nodedict[nodes[i]]=disparity[i]
        return nodedict

def strengthdist(mutualinformation,bincount):
    #Compute the weighted analog of a degree distribution.
    strens=strengths(mutualinformation)
    maxinfo=np.max(strens)
    mininfo=np.min(strens)
    return(np.histogram(mininfo,maxinfo,bin=np.linspace(mininfo,maxinfo,bincount+1)))

#NetworkX additions
#---------------------------------------------
def distance(mutualinformation):
    #Initialize array
    length=len(mutualinformation)
    thisdistance=np.zeros((length,length))
    #If an element has value less than (10^-14) in absolute value,
    #then treat it as of value (10^-16), set its distance
    #to 10^16. Otherwise set it mij^(-1).
    #The value (10^-14) must be adjusted in coordination with
    #the cutoff applied to the weighted network of interest.
    #The purpose of setting this value equal to 10^16 is
    #so that the geodesic path length algorithm will ignore
    #any edges less than the cutoff treating.
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
    #The pathlength is tested for an unreasonable value.
    #An unreasonable value for the path length is a function of the cutoff
    #applied to the weighted network under analysis and the value set in
    #the distance function thisdistance[i,j]=np.power(10.,16)
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
    denominator = factor*sum(alist)
    if denominator == 0.0:
        return 0.0
    else:
        return 1./denominator
def eigenvectorcentralitynx0(mutualinformation):
    #Uses the power method to find eigenvector of 
    #a weighted adjacency matrix.
    G=nx.Graph(mutualinformation)
    eigvcent=nx.eigenvector_centrality(G, weight='weight',max_iter=2000)
    return eigvcent
def eigenvectorcentralitynx(mutualinformation,startingvector):
    #Identical to eigenvectorcentralitynx0, but requires an additional argument startingvector.
    #starting vector provides an initial guess for the eigen vector centrality of all nodes.
    #startingvector must be a python dictionary. key = node, value = eigenvector centrality estimate.
    G=nx.Graph(mutualinformation)
    eigvcent=nx.eigenvector_centrality(G, weight='weight',max_iter=2000,nstart=startingvector)
    return eigvcent


