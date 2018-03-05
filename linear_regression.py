from  __future__  import print_function

import numpy as np
import matplotlib.pyplot as plt

#Define the vector of input samples as x, with 20 values sampled from a uniform distribution between 0 and 1
x=np.random.uniform(0,1,20)

#Generate the target values t from x with small gaussian noise so the estimation won't be perfect
#Define a function  f that represents the ine that generates t without noise 
def f(x):
	return x*2

#create the targets t with some gaussian noise 
noise_variance=0.2
#Gaussian noise error for each sample in x 
noise=np.random.randn(x.shape[0])*noise_variance
t=f(x)+noise
#t=f(x)

#Plot the target t versuse the input x 
plt.plot(x,t,'o',label='t')
#plot the initial line 
plt.plot([0,1],[f(0),f(1)],'b-',label='f(x)')
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$y$',fontsize=15)
plt.ylim([0,2])
plt.title('input(x) versus targets(t)')
plt.grid()
plt.legend(loc=2)
plt.show()

# define the neural function y=x*w
def nn(x,w):
	return x*w

#define the cost function
def cost(y,t):
	return ( (t-y)**2 ).sum()

def gradient(w,x,t):
	return 2*x*(nn(x,w)-t)

def delta_w(w_k,x,t,learning_rate):
	return learning_rate*gradient(w_k,x,t).sum()

w=0.1
learning_rate=0.1
#Start performing the gradient descent updates, and print the weights and costs:
nb_of_iterations=4
w_cost=[(w,cost(nn(x,w),t))]
# list to store the weight, costs value
for i in range(nb_of_iterations):
	dw=delta_w(w,x,t,learning_rate)
	#get the delta w update
	w=w-dw 
	#Update the current weight parameter
	w_cost.append((w,cost(nn(x,w),t)))
	#Add weight, cost to list 

#Print the final w, and cost
for i in range(0,len(w_cost)):
	print('w({}):{:.4f}\t cost:{:.4f}'.format(i,w_cost[i][0],w_cost[i][1]))
#Plot the target t versus the input x
plt.plot(x,t,'o',label='t')
#Plot the initial line
plt.plot([0,1],[f(0),f(1)],'b-',label='f(x)')
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$y$',fontsize=15)
plt.ylim([0,2])
plt.title('inputs(x) versus (t)')
plt.grid()
plt.legend(loc=2)
plt.show()

w=0
nb_of_iterations=50
for i in range(nb_of_iterations):
	dw=delta_w(w,x,t,learning_rate)
	#get the delta w update
	w=w-dw
	#update the current weight parameter

#Plot the fitted line against the target line 
#Plot the target t versus input x
plt.plot(x,t,'o',label='t')
#Plot the initial line
plt.plot([01,1],[f(0),f(1)],'b-',label='f(x)')
#plot the fitted line 
plt.plot([0,1],[0*w,1*w],'r-',label='fitted line')
plt.xlabel('input x')
plt.ylabel('target x')
plt.ylim([0,2])
plt.title('input vs target')
plt.grid()
plt.legend(loc=2)
plt.show()