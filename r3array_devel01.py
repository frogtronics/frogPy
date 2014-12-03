
# coding: utf-8

# In[21]:

from sympy import symbols, eye, sin, cos, Matrix, lambdify
from sympy import ones as ones_sym
from sympy import zeros as zeros_sym
from numpy import array, concatenate, arange, linspace, ones, vstack, reshape, identity
from scipy import sin as Sin  
from scipy import cos as Cos 

# References:
# 1.  General rotation matrix (of arbitrary axes and origin)
# "http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/""
# Aresult of rotating the point (x,y,z) about the line through (a,b,c) 
# with direction vector (u, v, w) where u^2 + v^2 + w^2 = 1 by angle theta


class r3array:
    def __init__(self, x_points, y_points, z_points):
        self.x_points = x_points
        self.y_points = y_points
        self.z_points = z_points
        self.npoints = len(x_points)
        self.MR = zeros_sym(4*self.npoints, 4*self.npoints)

    def MRcheck(self):
        return self.MR
    
    def rotmat(self, uvw_vector, abc_point):
        theta = symbols('theta')
        R = eye(4)
        u, v, w = uvw_vector[0], uvw_vector[1], uvw_vector[2]
        a, b, c = abc_point[0], abc_point[1], abc_point[2]
        R[0, 0] = u**2 + (v**2 + w**2)*cos(theta)
        R[1, 0] = u*v*(1 - cos(theta)) + w*sin(theta)
        R[2, 0] = u*w*(1 - cos(theta)) - v*sin(theta)
        R[3, 0] = 0
        R[0, 1] = u*v*(1 - cos(theta)) - w*sin(theta)
        R[1, 1] = v**2 + (u**2 + w**2)*cos(theta)
        R[2, 1] = v*w*(1 - cos(theta)) + u*sin(theta)
        R[3, 1] = 0
        R[0, 2] = u*w*(1 - cos(theta)) + v*sin(theta)
        R[1, 2] = v*w*(1 - cos(theta)) - u*sin(theta)
        R[2, 2] = w**2 + (u**2 + v**2)*cos(theta)
        R[3, 2] = 0
        R[0, 3] = (a*(v**2 + w**2) - u*(b*v + c*w))*(1 - cos(theta)) + (b*w - c*v)*sin(theta)
        R[1, 3] = (b*(u**2 + w**2) - v*(a*u + c*w))*(1 - cos(theta)) + (c*u - a*w)*sin(theta)
        R[2, 3] = (c*(u**2 + v**2) - w*(a*u + b*v))*(1 - cos(theta)) + (a*v - b*u)*sin(theta)
        R[3, 3] = 1
        # tile the rotation matrix daon the diagonal of a 4*n x 4*n identity matrix. n = n points
        starts = arange(0, (self.npoints - 1)*4 + 1, 4)
        ends = arange(4, self.npoints*4  + 1, 4)
        #MR = eye(4*npoints)
        for i in arange(self.npoints):
            self.MR[starts[i]: ends[i], starts[i]: ends[i]] = R
        return self.MR




# In[22]:

b = arange(6)
c = r3array(b, b, b)


# In[23]:

c.rotmat([1, 0, 0], [0, 0, 0])


# In[19]:




# In[70]:




# In[71]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



