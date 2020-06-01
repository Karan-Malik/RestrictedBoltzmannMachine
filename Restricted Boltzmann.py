# -*- coding: utf-8 -*-
"""
@author: Karan

"""

#Restricted Boltzmann Machine

import numpy as np
import pandas as pd
import torch

#importing dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

#Test and training set
training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set=np.array(training_set)

test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set=np.array(test_set)

#total number of users and movies
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

#to convert training and test set into the matrix form expected by the Boltzmann machine
def convert(data):
    new_data=[]
    for id_user in range(1,nb_users+1):
        id_movies=data[:,1][data[:,0]==id_user]
        id_ratings=data[:,2][data[:,0]==id_user]
        ratings=np.zeros(nb_movies)     
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

training_set=convert(training_set)
test_set=convert(test_set)

#converting to torch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

#considering movies rated below 3 as disliked (0) and 3 and above as liked(1)
#Movies which have not been rated have rating=-1
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1

test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1

#Contructing architecture of the RBM
class RBM():
    def __init__(self,nh,nv):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
    
    #sampling hidden nodes from visible nodes    
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    
    #sampling visible nodes from hidden nodes
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)

    #function for training the RBM
    def train(self,v0,vk,ph0,phk):
        self.W+=torch.mm(ph0,v0)-torch.mm(phk,vk)   
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk),0)
        
nv=len(training_set[0])
nh=100
batch_size=100

rbm=RBM(nh,nv)

#Training the RBM over 100 epochs
epochs=100
for epoch in range(1,epochs+1):
    train_loss=0
    s=0 #counter to normalize loss
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk=training_set[id_user:id_user+batch_size]
        v0=training_set[id_user:id_user+batch_size]
        ph0,_=rbm.sample_h(v0)
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss+=torch.mean(torch.abs(v0[v0>0]-vk[v0>0]))
        s+=1
    print('epoch: ',epoch,' loss: ',train_loss/s)
    
#Testing the RBM
test_loss=0
s=0
for id_user in range(nb_users):
    v=training_set[id_user:id_user+1]
    vt=test_set[id_user:id_user+1]
    if len(vt[vt>0])>0:
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        test_loss+=torch.mean(torch.abs(vt[vt>0]-v[vt>0]))
        s+=1
print('Test loss: ',str(test_loss/s))
    
