import math

#gradient descent in one dimension

f=lambda x:x**3-4*x+4*math.exp(x)

def gradientDescent(f):
	x=[i for i in range(-100000,3,100)]
	
	minimum=-2

	xdev=lambda x:3*x**2-4*math.exp(x)

	for i in range(0,len(x)):
		temp=x[i]
		x[i]=x[i]-1*xdev(x[i])	
		print(x[i])
		if(abs(x[i]-temp)<0.4):
			break

	#print(x)

gradientDescent(f)
