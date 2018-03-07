import numpy as np
import matplotlib.pyplot as plt 
from  matplotlib.colors import colorConverter,ListedColormap
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Define and generate the samples
#The number of sample in each class 
nb_of_samples_per_class=20 
#The mean of the blue class 
blue_mean=[0]
#The meand of the left red class 
red_left_mean=[-2]
#The mean of the right red class 
red_right_mean=[2]

#standard derivation of both classes
std_dev=0.5
#Generate sample from both classes 
x_blue=np.random.randn(nb_of_samples_per_class,1)*std_dev+blue_mean
x_red_left=np.random.randn(nb_of_samples_per_class/2,1)*std_dev+red_left_mean
x_red_right=np.random.randn(nb_of_samples_per_class/2,1)*std_dev+red_right_mean

#Merge samples in set of input variables x,and corresponding set of output variables t 
x=np.vstack(( x_blue,x_red_left,x_red_right ))
t=np.vstack( ( np.ones((nb_of_samples_per_class,1)), 
				np.zeros((nb_of_samples_per_class/2,1)),
				np.zeros((nb_of_samples_per_class/2,1))) )

#Plot samples from both classes as lines on a 1D space 
plt.figure(figsize=(8,0.5))
plt.xlim(-3,3)
plt.ylim(-1,1) 
#Plot samples 
plt.plot(x_blue,np.zeros_like(x_blue),'b|',ms=30)
plt.plot(x_red_left,np.zeros_like(x_red_left),'r|',ms=30)
plt.plot(x_red_right,np.zeros_like(x_red_right),'r|',ms=30)
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input samples from the blue and red class')
plt.xlabel('$x$',fontsize=15)
plt.show()

#Defint the rbf function
def rbf(z):
	return np.exp(-z**2)
	
#Plot the rbf function 
z=np.linspace(-6,6,100)
plt.plot(z,rbf(z),'b-')
plt.xlabel('$z$',fontsize=15)
plt.ylabel('$y$',fontsize=15)
plt.title('RBF function')
plt.grid()
plt.show()

#Function to compute hidden activation 
def hidden_activation(x,wh):
	return rbf( x*wh )

def logistic(z):
	return 1/(1+np.exp(-z))

#Define output layer feedward
def output_activation(h,wo):
	return logistic( h*wo -1 )

#Define the neural network function 
def nn(x,wh,wo):
	return output_activation( hidden_activation(x,wh),wo )

#Define the neural network prediction function that only returns 
#1 or 0 dependent on the predicted class 
def nn_predict(x,wh,wo):
	return np.around( nn(x,wh,wo) )

#Define the cost function 
def cost(y,t):
	return -np.sum( np.multiply(t,np.log(y)) + np.multiply( (1-t),np.log(1-y) ) )
	
#Defint a function to calculate the cost for a given set of parameters 
def cost_for_param(x,wh,wo,t):
	return cost( nn(x,wh,wo),t )

#Plot the cost in function of the weights 
#Define a vector of weights for which we want to plot the cost 
nb_of_ws=200
#hidden weights
wsh=np.linspace(-10,10,num=nb_of_ws)
#output weights 
wso=np.linspace(-10,10,num=nb_of_ws)
#generate grid 
ws_x,ws_y=np.meshgrid( wsh, wso)
cost_ws=np.zeros( (nb_of_ws,nb_of_ws) )
for i in range(nb_of_ws):
	for j in range(nb_of_ws):
		cost_ws[i,j]=cost( nn(x,ws_x[i,j],ws_y[i,j]),t )
		#print cost_ws[i,j]
#Plot the cost function surface 
fig=plt.figure()
ax=Axes3D(fig)
#plot the surface 
surf=ax.plot_surface(ws_x,ws_y,cost_ws,linewidth=0,cmap=cm.pink)
ax.view_init(elev=60,azim=-30)
cbar=fig.colorbar(surf)
ax.set_xlabel('$w_h$',fontsize=15)
ax.set_ylabel('$w_o$',fontsize=15)
ax.set_zlabel('$\\xi$',fontsize=15)
cbar.ax.set_ylabel('$\\xi$',fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()

def gradient_output(y,t):
	return y-t 

def gradient_weight_out(h,grad_output):
	return h*grad_output 

def gradient_hidden(wo,grad_output):
	return grad_output*wo
	
	
def gradient_weight_hidden(x,zh,h,grad_hidden):
	return grad_hidden*(-2)*zh*h*x

#define the update function to update the network parameters over i teration 
def backpro_update(x,t,wh,wo,learning_rate):
	#Compute the output of network 
	zh=x*wh
	h=rbf(zh)
	y=output_activation(h,wo)
	
	grad_output=gradient_output(y , t )
	d_wo=learning_rate*gradient_weight_out(h,grad_output)
	grad_hidden=gradient_hidden(wo,grad_output)
	d_wh=learning_rate*gradient_weight_hidden(x,zh,h,grad_hidden)
	return (wh-d_wh.sum(),wo-d_wo.sum())

#Run backprobagation 
#Set the initial weight parameter 
wh=2
wo=-5
#Set the leanring rate 
learning_rate=0.2

#Start the gradient descent updates and plot the iterations
nb_of_iterations=50
lr_update=learning_rate/nb_of_iterations
w_cost_iter=[ (wh, wo, cost_for_param(x,wh,wo,t))]
for i in range(nb_of_iterations):
	#Decrease the learning rate 
	learning_rate-= lr_update 
	#Update the weight via backpropagation
	wh,wo=backpro_update(x,t,wh,wo,learning_rate)
	w_cost_iter.append( (wh,wo, cost_for_param(x,wh,wo,t) ))

#Print the final cost 
print(' final cost is {:.2f} for weights wh {:2f} and wo:{:.2f}'.format(cost_for_param(x,wh,wo,t),wh,wo))

#Plot the weight updates on the error surface 
fig=plt.figure()
ax=Axes3D(fig)
surf=ax.plot_surface(ws_x,ws_y,cost_ws,linewidth=0,cmap=cm.pink)
ax.view_init(elev=60,azim=30)
cbar=fig.colorbar(surf)
cbar.ax.set_ylabel('$\\xi$',fontsize=15)

#Plot the updates 
for i in range( 1, len(w_cost_iter)):
	wh1,wo1,c1=w_cost_iter[i-1]
	wh2,wo2,c2=w_cost_iter[i]
	#Plot the weight-cost value and the line taht represents the update 
	ax.plot([wh1],[wo1],[c1],'w+')
	ax.plot([wh1,wh2],[wo1,wo2],[c1,c2],'w-')
#Plot the last weight 
wh1,wo1,c1=w_cost_iter[ len(w_cost_iter)-1]
ax.plot([wh1],[wo1],c1,'w+')
ax.set_xlabel('$w_h',fontsize=15)
ax.set_ylabel('$w_o$',fontsize=15)
ax.set_zlabel('$\\xi$',fontsize=15)
plt.title('Gradient descent update on cost surface')
plt.grid()
plt.show()