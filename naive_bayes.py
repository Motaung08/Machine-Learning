'''Group members'''
'''Tshepo Nkambule 1611821
Tshepang Motaung 1431795
Tshifhiwa Mavhona 1613720
Dineo Ramakhothoane 1457154'''

import math
import statistics
import numpy as np
import csv
import matplotlib.pyplot as plt
import pylab
import time
    
'''our first three normal distribution which are with respect to class 1'''
fB=None
fG=None
fR=None
    
'''second set of gausian pdf which are with respectto class 2'''
fBprime=None
fGprime=None
fRprime=None

''' class probabilities'''    
class1=None
class2=None
    

'''the normal distribution which will be used to give conditional class probabilities we return f and f prime that we
construct gausians with the 4 parameters contained in summary '''
def normal_distribution(summary):
	mean,stddev=summary['mean-1'],summary['stdev-1']
	f=lambda x:((1)*(1/math.sqrt(2*math.pi*stddev**2))*math.exp(-1*((x-mean)**2)/(2*stddev**2)))
	mean2,stddev2=summary['mean-2'],summary['stdev-2']
	g=lambda x:((1)*(1/math.sqrt(2*math.pi*stddev2**2))*math.exp(-1*((x-mean2)**2)/(2*stddev2**2)))
	return f,g

'''given an attribute find the column index in which it is located in array'''
def locate(attr):
	if(attr=='B'):
		return 0

	elif(attr=='G'):
		return 1

	elif(attr=='R'):
		return 2

'''obtains summary of an attribute  mean and standard deviation'''
def summary(attr,data):
	summary={}
	summary['attribute']=attr
	class1=[]
	class2=[]
	column=locate(attr)
	#we want to compute the mean of that attribute with respect to all classes since we have two its straight forwad
	for row in data:
				if(row[1]==1):
					class1.append(row[0][column])
				else:
					class2.append(row[0][column])
	summary['mean-1']=statistics.mean(class1)
	summary['stdev-1']=statistics.stdev(class1)
	summary['mean-2']=statistics.mean(class2)
	summary['stdev-2']=statistics.stdev(class2)

	return summary

'''give the occurence of class x so we take x to be the occurence of class 1 which is human skin''' 
def class_probability(data):
	counter=0
	for vector in data:
		if vector[1]==1:
			counter+=1
			
	return counter/len(data)

''' want to measure perfomance on training data so count the misclassifications we count values classified in the wrong
 category i.e how many non skin were classified as skin and how many skin was classified as non skin and then how many were classified correctly'''
'''returns class which input is said to belong'''
def classiffy(x):
    ''' compute the numerator for naive bayes'''
    num=(fB(x[0])*fG(x[1])*fR(x[2]))*class1
    '''compute denominator for naive bayes'''
    den=num+(fBprime(x[0])*fGprime(x[1])*fRprime(x[2]))*class2
    '''use log probabilities'''
    humanskin=math.log((num)/(den))
    not_humanskin=math.log(1-num/den)
    if humanskin>=not_humanskin:
        return 1
    else:
        return 2

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
					if(class_value==1):
						correct_human_label+=1
					else:
						correct_not_human_label+=1
				else:
           # '''not skin labelled as skin'''
					if(class_value==1):
							skin_error+=1  
           # '''skin labelled as not skin'''
					else:
							not_skin_error+=1
	  
		print("Total data points",size)
		print("Total correct labels",correct_label)
		print((correct_label/size)*100,"% of the  data was labelled correctly")
		print(100-(correct_label/size)*100,"% of the data was labelled incorrectly")
		print("The number of correct human skin labels was",correct_human_label)	
		print(skin_error," points were mislabelled as human skin while they were actually not human skin")
		print("The number of correct not human skin labels was",correct_not_human_label)        
		print(not_skin_error," points were mislabelled as not human skin while they were human skin")

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
             vector=[bgr,int(row[-1:][0])]
             data.append(vector)
             
    return data
             
''' load the data'''        
training_data=load_data('training_data.csv')         
#validation_data=load_data('validation_data.csv')
testing_data=load_data('testing_data.csv')

'''get the classs probabilitie just compute 1 and the other is one subtract computed probability'''
class1=class_probability(training_data)
class2=1-class1

'''for each attribute compute the stats summary which has mean and stdev with repect to a class'''
summaryB=summary('B',training_data)
summaryG=summary('G',training_data)
summaryR=summary('R',training_data)

'''intialize our 6 normal distributions with parameters from summaries f is pdf class 1 then fprim is wrs to class2'''
fB,fBprime=normal_distribution(summaryB)
fG,fGprime=normal_distribution(summaryG)
fR,fRprime=normal_distribution(summaryR)

#print("\nTraining perfomance")
#perfomance(training_data)

print("\nTesting perfomance")
perfomance(testing_data)


