import numpy as nump
from matplotlib import pyplot as plt



random = nump.random(10)

import matplotlib.pyplot as plt
#1.Create two arrays X and Y as NumPy arrays.
x = nump.arange(11)
y = nump.arange(11)
xx, yy = nump.meshgrid(x, y, sparse=True)
#2. Define a variable Z = (X-5)^2 + (Y-5)^2
z = nump.power(xx-5, 2) + nump.power(yy-5,2)


plt.contourf(x,y,z, 20, alpha =0.5, cmap = plt.get_cmap('jet'))
#Add the title to this Figure "Firstname_Lastname_Quiz1Contour"
plt.title('Jonothan_Meyer_Quiz1Contour')
#Mark the X and Y axis and mark out the Z's minimum value on the figure.
plt.plot([5], [5], 'o', ms=12, markeredgewidth=3, color='orange')
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel(r'$X$',fontsize=16)
plt.ylabel(r'$Y$',fontsize=16)
plt.show()