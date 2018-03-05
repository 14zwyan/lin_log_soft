import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter,ListedColormap
from matplotlib import cm 

#Define and generate the smaples 
nb_of_samples_per_class=20
#The mean of red class 
red_mean=[-1,0]
#The meand of blue class 
blue_mean=[1,0]
#standard deviation of both classes 
std_dev=1.2

#Generate samples from both classes 
x_red=np.random.randn(nb_of_samples_per_class,2)*std_dev+red_mean
x_blue=np.random.randn(nb_of_samples_per_class,2)*std_dev+blue_mean


#Merge samples in set of input variables x, and corresponding set of output variables t 
X=np.vstack((x_red,x_blue))
t=np.vstack(( np.zeros((nb_of_samples_per_class,1)), np.ones((nb_of_samples_per_class,1)) ))

#Plot both class on the x1,x2 plane 
plt.plot( x_red[:,0] ,x_red[:,1],'ro',label='class red')
plt.plot( x_blue[:,0],x_blue[:,1],'bo',label='class blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$',fontsize=15)
plt.ylabel('$x_2$',fontsize=15)
plt.axis([-4,4,-4,4])
plt.title('red vs blue classes in the input space')
plt.show()

#Defint the logistic function 
def logistic(z):
	return 1/(1+np.exp(-z))

#Defint the neural network function y=1/(1+numpy.exp(-x*w))
def nn(x,w):
	return logistic( np.dot(x,w.T) )

#Defidne the neural network prediction function that only returns 1 or 0 depending on the predicted class
def nn_predict(x,w):
	return np.around(nn(x,w))

#Define the cost function 
def cost(y,t):
	return np.sum( np.multiply(-t,np.log(y)) -np.multiply( (1-t),np.log(1-y) ))

#Plot the cost in function of the weights
#Defint a vector of weights for which we want to plot the cost 
#Compute the cost nb_of_ws times in each dimension 
nb_of_ws=100
# weight 1
ws1=np.linspace(-5,5,num=nb_of_ws)
ws2=np.linspace(-5,5,num=nb_of_ws)
ws_x,ws_y=np.meshgrid(ws1,ws2)
cost_ws=np.zeros((nb_of_ws,nb_of_ws))
#Fill the cost matrix for each combination of weights 
for i in range(nb_of_ws):
	 for j in range(nb_of_ws):
		cost_ws[i,j]=cost( nn(X,np.asmatrix([ws_x[i,j],ws_y[i,j]]) ), t)
#Plot the cost function surface
print 'ws1'
print ws1
print '---------'
print 'ws2'
print ws2
print '---------'
print 'ws_x'
print ws_x 
print '---------'
print 'ws_y'
print ws_y 
plt.contourf(ws_x,ws_y,cost_ws,20,cmap=cm.pink)
cbar=plt.colorbar()
cbar.ax.set_ylabel('$\\xi$',fontsize=15)
plt.xlabel('$w_1$',fontsize=15)
plt.ylabel('$w_2$',fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()

#define the gradient fucntion 
def gradient(w,x,t):
	return ( nn(x,w)-t ).T*x
	
#define the update function delta w which return the 
#delta w for each weight in a vector 
def delta_w(w_k,x,t,learning_rate):
	return learning_rate*gradient(w_k,x,t)

#Set the initial weight parameter 
w=np.asmatrix([-4,-2])
#Set the learning rate 
learning_rate=0.05

#Start the gradient descent updates and plot the iterations 
nb_of_iterations=10
#Number of gradient descent updates 
w_iter=[w]
#List to stor the weight values over iterations 
for i in range(nb_of_iterations):
	#Get the delta w update 
	dw=delta_w(w,X,t,learning_rate)
	#Update the weights 
	w=w-dw 
	#Store the weights for plotting 
	w_iter.append(w)

#Plot the first weight updates on the error surface 
#Plot the error surface 
plt.contourf(ws_x,ws_y,cost_ws,20,alpha=0.9,cmap=cm.pink)
cbar=plt.colorbar()
cbar.ax.set_ylabel('cost')

#Plot the updates 
for i in range(1,4):
	w1=w_iter[i-1]
	w2=w_iter[i]
	#Plot the weight-cost value and the line that represents the update 
	#Plot the weight cost value 
	plt.plot(w1[0,0],w1[0,1],'bo')
	plt.plot([w1[0,0],w2[0,0]],[w1[0,1],w2[0,1]],'b-')
	plt.text(w1[0,0]-0.2,w1[0,1]+0.4,'$w({})$'.format(i),color='b')
w1=w_iter[3]
#Plot the last weight 
plt.plot(w1[0,0],w1[0,1],'bo')
plt.title('Gradient descent updates on cost surfagce')
plt.grid()
plt.show()


# Plot the resulting decision boundary
# Generate a grid over the input space to plot the color of the
#  classification at that grid point
nb_of_xs = 200
xs1 = np.linspace(-4, 4, num=nb_of_xs)
xs2 = np.linspace(-4, 4, num=nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i,j] = nn_predict(np.asmatrix([xx[i,j], yy[i,j]]) , w)
# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])

# Plot the classification plane with decision boundary and input samples
plt.clf()
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='target red')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='target blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.title('red vs. blue classification boundary')
plt.show()