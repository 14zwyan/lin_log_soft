import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  colorConverter,ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Define the softmax function 
def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

#Plot the softmax output for 2 dimension for both classes
#Plot the output in function of the weights 
#Define a vector of weights for which we want to plot the output 
nb_of_zs=2000
zs=np.linspace(-10,10,num=nb_of_zs)
zs_1,zs_2=np.meshgrid( zs,zs )
#initialize the output 
y=np.zeros( (nb_of_zs,nb_of_zs,2) )
#Fill the output mastrix of each combination of input z's 
for i in range(nb_of_zs):
	for j in range( nb_of_zs ):
		y[i,j]=softmax( [ zs_1[i,j],zs_2[i,j] ] )
#Plot the cost function surfaces for both classes 
fig=plt.figure()
#Plot the cost function surface for t=1
ax=fig.gca( projection='3d' )
surf=ax.plot_surface(zs_1,zs_2,y[:,:,0],linewidth=0,cmap=cm.coolwarm)
ax.view_init(elev=30,azim=70)
cbar=fig.colorbar(surf)
ax.set_xlabel('$z_1$',fontsize=15)
ax.set_ylabel('$z_2$',fontsize=15)
ax.set_zlabel('$y_1$',fontsize=15)
ax.set_title('$P(t=1|\mathbf{z}$',fontsize=15)
plt.grid()
plt.show()