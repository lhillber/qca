#!/usr/bin/python3

import numpy as np
from math import log

def probabilities(data,samplespace):
    #Computes the probability of elem in sample space
    #By counting the occurrences of elem in data.
    n=float(len(data))
    probdict={}
    for elem in samplespace:
        count=0.
        for datum in data:
            if datum==elem:
                count+=1.
        probdict[elem]=count/n
    return probdict

def discretizetimesite(data,i,bins):
    return np.digitize(data[:,i],bins,right=False)-1

def discretizetimetwosite(data,i,j,binsi,binsj):
    N=len(data)
    sitei=discretizetimesite(data,i,binsi)
    sitej=discretizetimesite(data,j,binsj)
    tuples=[]
    for k in range(N):
        tuples.append(tuple([sitei[k],sitej[k]]))
    return tuples

def plogp(probability):
    #If probability==0 we adopt the standard convention
    #that plogp=0.  This is justified by the fact that
    #lim_{x->0} x*logx = 0.
    if probability<1e-14:
        return 0.0
    else:
        return probability*log(probability, 2)

def shannonentropy(probabilities,d=2):
    dictcheck=isinstance(probabilities,dict)
    q=0
    #If not a dictionary assume a list
    if not dictcheck:
    #Compute plogp on all probabilities in probabilities
        for p in probabilities:
            q-=plogp(p)
    else:
        probabilities=probabilities.values()
        for p in probabilities:
            q-=plogp(p)
    #Divide by np.log(d) to convert to base d logarithm.
    return q 
    #return q/np.jklog(d)

def mutualinformation(probdict1,probdict2,probdict12,d):
    dictcheck1=isinstance(probdict1,dict) 
    dictcheck2=isinstance(probdict2,dict)
    dictcheck3=isinstance(probdict12,dict)
    #Checks whether the inputs are probability distributions
    #or not.
    if dictcheck1 and dictcheck2 and dictcheck3:
        p1=probdict1.values()
        p2=probdict2.values()
        p12=probdict12.values()
    else:
        #If dictionary is not supplied a list of values is
        #assumed.
        p1=probdict1
        p2=probdict2
        p12=probdict12
    s1=shannonentropy(p1,d)
    s2=shannonentropy(p2,d)
    s12=shannonentropy(p12,d)
    mutualinformation = 0.5*(s1+s2-s12)
    return mutualinformation

def mutinfonetwork(data,samplespace=[0,1],bins=np.array([0.0,0.5,1.0]),d=2,shifts=0,means=False):
    """
    The floats in data are converted into elements of samplespace
    samplespace is a list of immutable data, for example integers.
    bins is a numpy array that sets the cutoffs for elements of
    samplespace in the conversion of data.  That is bins has the
    form np.array([0.0,0.5,1.0]).
    """
    samplespace2=[(item1,item2) for item1 in samplespace for item2 in samplespace]
    L=len(data[0])
    T=len(data)
    mutualinfo=np.zeros((L,L))
    if means==True:
        if shifts==0:
            left=bins[0]
            right=bins[2]
            bins=[]
            for i in range(L):
                bins.append(np.array([left,np.mean(data[:,i]),right]))
            for i in range(L):
                for j in range(i):
                    onesitediscrete=discretizetimesite(data,i,bins[i])
                    twositediscrete=discretizetimesite(data,j,bins[j])
                    onetwodiscrete= discretizetimetwosite(data,i,j,bins[i],bins[j])
                    prob1=probabilities(onesitediscrete,samplespace)
                    prob2=probabilities(twositediscrete,samplespace)
                    prob12=probabilities(onetwodiscrete,samplespace2)
                    mutualinfo[i][j]=mutualinformation(prob1,prob2,prob12,d)
        else:
            left=bins[0]
            right=bins[2]
            bins=[]
            for i in range(L):
                bins.append(np.array([left,np.mean(data[:,i]),right]))
            for i in range(L):
                for j in range(i):
                    onesitediscrete=discretizetimesite(data,i,bins[i])
                    twositediscrete=discretizetimesite(data,j,bins[j])
                    onetwodiscrete= discretizetimetwosite(data,i,j,bins[i],bins[j])
                    prob1=probabilities(onesitediscrete,samplespace)
                    prob2=probabilities(twositediscrete,samplespace)
                    prob12=probabilities(onetwodiscrete,samplespace2)
                    mutualinfo[i][j]=mutualinformation(prob1,prob2,prob12,d)
    else:
        if shifts==0:
            for i in range(L):
                for j in range(i):
                    onesitediscrete=discretizetimesite(data,i,bins)
                    twositediscrete=discretizetimesite(data,j,bins)
                    onetwodiscrete= discretizetimetwosite(data,i,j,bins,bins)
                    prob1=probabilities(onesitediscrete,samplespace)
                    prob2=probabilities(twositediscrete,samplespace)
                    prob12=probabilities(onetwodiscrete,samplespace2)
                    mutualinfo[i][j]=mutualinformation(prob1,prob2,prob12,d)
        else:
            #Will Define Shifting behavior if necessary.
            for i in range(L):
                for j in range(i):

                    onesitediscrete=discretizetimesite(data,i,bins)
                    twositediscrete=discretizetimesite(data,j,bins)
                    onetwodiscrete= discretizetimetwosite(data,i,j,bins)
                    prob1=probabilities(onesitediscrete,samplespace)
                    prob2=probabilities(twositediscrete,samplespace)
                    prob12=probabilities(onetwodiscrete,samplespace2)
                    mutualinfo[i][j]=mutualinformation(prob1,prob2,prob12,d)
    return mutualinfo+mutualinfo.transpose()

def spatialnetworksQ(corr,loc,d=2):
    L=len(loc[0])
    T=len(loc)
    mutualinfo=[]
    for t in range(T):
        mutualinfo.append(np.zeros((L,L)))
        for i in range(L):
            for j in range(i):
                pi1=loc[t][i]
                pi0=1.-pi1
                pj1=loc[t][j]
                pj0=1.-pj1
                pij11=corr[t][i][j]
                pij10=pi1-pij11
                pij01=pj1-pij11
                pij00=1-pi1-pj1+pij11
                probi=[pi1,pi0]
                probj=[pj1,pj0]
                probij=[pij00,pij01,pij10,pij11]
                mutualinfo[t][i][j]=mutualinformation(probi,probj,probij,d)
        mutualinfo[t]=mutualinfo[t]+mutualinfo[t].transpose()
    return mutualinfo

def spatialnetworksC(boards,w,d=2):
    if len(boards)!=len(w):
        print( "Number of Weights must be equal to number of boards!" )
    else:
        nb=len(boards)
        L=len(boards[0][0])
        T=len(boards[0])
        mutualinfo=[]
        pi={}
        pj={}
        pij={}
        for t in range(T):
            mutualinfo.append(np.zeros((L,L)))
            for i in range(L):
                for j in range(i):
                    pi[0]=0.0
                    pi[1]=0.0
                    pj[0]=0.0
                    pj[1]=0.0
                    pij[(0,0)]=0.0
                    pij[(0,1)]=0.0
                    pij[(1,0)]=0.0
                    pij[(1,1)]=0.0

                    for b in range(nb):
                        if boards[b,t,i]==0:
                            pi[0]+=w[b]
                            if boards[b,t,j]==0:
                                pj[0]+=w[b]
                                pij[(0,0)]+=w[b]
                            else:
                                pj[1]+=w[b]
                                pij[(0,1)]+=w[b]
                        else:
                            pi[1]+=w[b]
                            if boards[b,t,j]==0:
                                pj[0]+=w[b]
                                pij[(1,0)]+=w[b]
                            else:
                                pj[1]+=w[b]
                                pij[(1,1)]+=w[b]
                    prob1=pi.values()
                    prob2=pj.values()
                    prob12=pij.values()
                    mutualinfo[t][i][j]=mutualinformation(prob1,prob2,prob12,d)
            mutualinfo[t]=mutualinfo[t]+mutualinfo[t].transpose()
        return mutualinfo


                            
                            
            

def temporalshift(twocolumndata,shift=0):
    """
    Expects a numpy array.
    """
    twocolumndata[:,1]=np.roll(twocolumndata[:,1],shift)
    return twocolumndata



tests=False
if tests==True:
    print( 'Shannon Entropy Tests','\n','===================' )
    plist=[0.5,0.5]
    plist2=[1.0,0.0]
    plist3=[0.7,0.3]
    print( shannonentropy(plist,2) )
    print( shannonentropy(plist2,2) )
    print( shannonentropy(plist3,2) )
    print( '===================','\n' )

    print( 'Probability Tests','\n','===================' )
    testdata=[0,0,0,1,1]
    testdata2=[tuple([0,0]),tuple([0,1]),tuple([1,0])]
    samplespace2=[tuple([0,0]),tuple([0,1]),tuple([1,0]),tuple([1,1])]
    print( 'probabilities--',probabilities(np.array(testdata),np.array([0,1])) )
    print( 'probs2--',probabilities(testdata2,samplespace2) )
    print( '===================','\n' )

    print( 'Mutual Information Tests','\n','===================' )
    testdata=np.array([[0.8,0.8,0.8,0.2,0.2,0.6,0.8,0.8,0.8,0.2,0.2,0.6],[0.2,0.8,0.8,0.2,0.2,0.6,0.2,0.8,0.8,0.2,0.2,0.6],[0.8,0.8,0.8,0.2,0.2,0.6,0.8,0.8,0.8,0.2,0.2,0.6]]).transpose()
    testdata2=2*np.array([[0.5,0.5,0.5,0.25,0.25,0.125,0.5,0.5,0.5,0.25,0.25,0.125],[0.25,0.5,0.5,0.25,0.25,0.125,0.25,0.5,0.5,0.25,0.25,0.125],[0.5,0.5,0.5,0.25,0.25,0.125,0.5,0.5,0.5,0.25,0.25,0.125]]).transpose()
    testdata3=np.array([[1,0,0.25],[0,0.25,1],[0.25,0,1]]).transpose()
    onesitediscrete=discretizetimesite(testdata,0,np.array([0.0,0.5,1.0]))
    twositediscrete=discretizetimesite(testdata,1,np.array([0.0,0.5,1.0]))
    onetwodiscrete= discretizetimetwosite(testdata,0,1,np.array([0.0,0.5,1.0]),np.array([0.0,0.5,1.0]))
    prob1=probabilities(onesitediscrete,[0,1])
    prob2=probabilities(twositediscrete,[0,1])
    prob12=probabilities(onetwodiscrete,samplespace2)
    print(
            mutinfonetwork(testdata,samplespace=[0,1],bins=np.array([0.0,0.5,1.0]),d=2)
            )
    print(
            mutinfonetwork(testdata2,samplespace=[0,1],bins=np.array([0.0,0.5,1.0]),d=2)
            )
    print( "means=False",mutinfonetwork(testdata2,means=False) )
    print( "means=True",mutinfonetwork(testdata2,means=True) )
    print( '===================' )
    print( mutinfonetwork(testdata3,means=False) )
    print( '===================' )
    print( map(np.mean,testdata2.transpose()) )
    print( testdata2.transpose()[0] )
    print( testdata2.transpose()[1] )
    print( testdata2.transpose()[2] )
    print( '===================' )

    #int(np.all(1-np.bitwise_xor([0, 0], [0, 0])))
    abcd=np.array([[[0,0,1],[1,1,1]],[[1,0,0],[0,0,0]]])    
    spatialnetworksC(abcd,[0.5],2)
    print( spatialnetworksC(abcd,[0.5,0.5],2) )
    print( spatialnetworksC(abcd,[0.1,0.9],2) )
    correl=0.25*np.ones((4,6,6))
    local=0.5*np.ones((4,6))
    print( 'local',local )
    print( 'correl',correl )
    print( "aasdk" )
    print( spatialnetworksQ(correl,local,2) )

    print( 'abcd',abcd[:,1] )
    print( 'def',abcd[:,0,1] )
    print( 'ghi',abcd[:,0,2] )
    print( 'jkl',abcd[:,0,0] )
    print( 'ala',abcd[:,0,[0,1]] )
    print( 'ala2',map(tuple,abcd[:,0,[0,1]]) )
    print( 'ala3',map(lambda x: np.dot([1,1],x),abcd[:,0,[0,1]]) )
