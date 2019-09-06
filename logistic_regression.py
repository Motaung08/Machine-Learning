# -*- coding: utf-8 -*-
"""
Created on Mon May 13 03:55:02 2019

@author: phantom
"""
'''Group members'''
'''Tshepo Nkambule 1611821
Tshepang Motaung 1431795
Tshifhiwa Mavhona 1613720
Dineo Ramakhothoane 1457154'''
import math
import random
import statistics
import numpy as np
from numpy import linalg as LA
import csv
import matplotlib.pyplot as plt
import pylab
import time
from mpl_toolkits.mplot3d import Axes3D

h=None

'''creates a logistic function'''

def plot(data,theta):
	y=[]
	xv=[]

	for X in data:
		x=X[0]
		y.append(h(x))
		xv.append(theta.dot(np.array(x)))

	plt.scatter(xv,y)
	plt.show()
	
def logistic_function(theta):
    return lambda x:1/(1+math.exp(-1*theta.dot(x)))
   
def mean_vector(data):
    vector=np.array((0,0,0))    
    for x in data:
        vector=vector+x[0]        
    return (1/len(data))*vector

'''determines best parameters for theta '''
def adjust_parameters(data,theta):
		temp=[theta[0],theta[1],theta[2]]		
		ht=logistic_function(theta)
		it=0;
		pl=[]
		xv=[]
		for k in range(0,len(data)):
				row=data[k]        
				X=row[0]
				y=row[1]
				theta[0]=theta[0]-0.01*(ht(X)-y)

				value=0
				for j in range(0,len(theta)):
							current_sum=it
							M_points=[data[a][0] for a in range(it)]
							M_label=[data[a][1] for a in range(it)]
							val=[((ht(M_points[a])-M_label[a])*M_points[a][j]+0.00*theta[j]) for a in range(0,it)]							
							value=sum(val)							
							theta[j]=theta[j]-0.01*value
							#print(sum(val))
				ht=logistic_function(theta)
  
				if(it>5000):
					break
				it+=1

			
		return theta
       
'''classify point using boundary if its probability is greater than 0.5 its human skin otherwise its not '''
def classiffy(x):
		nothumanskin=h(x)
		if(nothumanskin>0.5):
			return 1
		else:
			return 0

'''take all input vectors and try to give the a class and check if it matches given class'''
def perfomance(data):
		correct_label=0
		not_skin_error=0
		skin_error=0
		size=len(data)
		correct_human_label=0
		correct_not_human_label=0
		for i in range(len(data)):
				x=data[i][0]
				class_value=classiffy(x)
        
				if(class_value==data[i][1]):
					correct_label+=1
					if(class_value==0):
						correct_human_label+=1
					
					else:
						correct_not_human_label+=1
		
				else:
           # '''not skin labelled as skin'''
					if(class_value==0):
							skin_error+=1
            
           # '''skin labelled as not skin'''
					else:
							not_skin_error+=1

		print((correct_label/size)*100,"% of the  data was labelled correctly")
		print(100-(correct_label/size)*100,"% of the data was labelled incorrectly")
		print("Total data points",size)
		print("Total correct labels",correct_label)
		print(skin_error," points were mislabelled as human skin while they were actually not human skin")
		print("The number of correct human skin labels was",correct_human_label)
		print(not_skin_error," points were mislabelled as not human skin while they were human skin")
		print("The number of correct not human skin labels was",correct_not_human_label)
        
'''loading of data'''
def load_data(path):
    data=[]
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        for row in readCSV:
             bgr=row[:-1]
             '''the data is pre processed ie input variables are normalized by multiplying by (x-xmin)/(xmax-xmin) = 1/255'''
             for i in range(len(bgr)):
                 bgr[i]=float(bgr[i])/255
             bgr=np.array(bgr) 
             vector=[bgr,int(row[-1:][0])-1]
             data.append(vector)
    return data

def load_data_random(path):
    data=[]
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        for row in readCSV:
             bgr=row[:-1]
             '''the data is pre processed ie input variables are normalized by multiplying by (x-xmin)/(xmax-xmin) = 1/255'''
             for i in range(len(bgr)):
                 bgr[i]=float(bgr[i])
             bgr=np.array(bgr) 
             vector=[bgr,int(row[-1:][0])]
             data.append(vector)
    return data
    
random_training=load_data_random('random_training.csv')
validation_data=load_data('validation_data.csv')
training_data=load_data('training_data.csv')
testing_data=load_data('testing_data.csv')

'''initiaally guess theta as a mean vector'''
theta=np.array([0.4,0,0])

'''adjust theta accordingly'''
theta=adjust_parameters(random_training,theta)
'''define the logistic function with the new adjusted parameters'''
h=logistic_function(theta)

'''report the model perfomance on the testing data set'''

perfomance(testing_data)




