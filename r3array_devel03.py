
# coding: utf-8

# In[2]:

from sympy import symbols, eye, sin, cos, Matrix, lambdify
from sympy import ones as ones_sym
from sympy import zeros as zeros_sym
from numpy import array, concatenate, arange, linspace, ones, vstack, reshape, identity, shape, asarray, ravel
from scipy import sin as Sin  
from scipy import cos as Cos
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# References:
# 1.  General rotation matrix (of arbitrary axes and origin)
# "http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/""
# Aresult of rotating the point (x,y,z) about the line through (a,b,c) 
# with direction vector (u, v, w) where u^2 + v^2 + w^2 = 1 by angle theta

class r3array:
    def __init__(self, x_points, y_points, z_points, first_point = 0, last_point = 0):
        if last_point == 0:
            self.last_point = len(x_points)
        else:
            self.last_point = last_point
        self.time_slice = slice(first_point, self.last_point, 1)
        self.x_points_all = x_points
        self.y_points_all = y_points
        self.z_points_all = z_points
        self.x_points = x_points[self.time_slice]
        self.y_points = y_points[self.time_slice]
        self.z_points = z_points[self.time_slice]
        theta = symbols('theta')
        self.npoints = len(self.x_points)
        self.MR = zeros_sym(4*self.npoints, 4*self.npoints)
        self.rv = zeros_sym(1, 4*self.npoints)
        a_1 = ones(shape = (self.npoints, ))
        self.xyz1_flat = ones_sym(1, self.npoints*4)
        self.thetasub = lambdify(theta, self.rv)
        for i in arange(self.npoints):
            self.xyz1_flat[i*4] = self.x_points[i]
            self.xyz1_flat[i*4 + 1] = self.y_points[i]
            self.xyz1_flat[i*4 + 2] = self.z_points[i]
            self.xyz1_flat[i*4 + 3] = a_1[i]
        
    def check(self):
        return self.rv
    
    def get_points(self, index = 0):
        ntot = self.npoints*4
        return self.xyz1_flat[index: (ntot -1): 4]
    
    def rotate(self, uvw_vector, abc_point, returnR_yes_no = 0):
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
        self.rv = self.MR.dot(self.xyz1_flat)
        #self.rv = asarray(self.rv_list)
        self.thetasub = lambdify(theta, self.rv) #overwrite
        if returnR_yes_no == 0:
            return
        else:
            return R
    
    def r_transform(self, theta):
        return asarray(self.thetasub(theta))
    
    def r_transform_partition(self, theta, index = 0):
        xyz1_r = asarray(self.thetasub(theta))
        ntot = self.npoints*4
        return xyz1_r[index: (ntot -1): 4]
    
    def r_transform_self(self, theta):
        #This function is useful if multiple transformations are needed
        #i.e. overwrite x, y, z points with first transformation then re-transform them
        #however many times
        xyz1_r = self.thetasub(theta)
        ntot = 4*self.npoints
        self.x_points = xyz1_r[0: (ntot -1): 4]
        self.y_points = xyz1_r[1: (ntot -1): 4]
        self.z_points = xyz1_r[2: (ntot -1): 4]
        for i in arange(self.npoints):
            self.xyz1_flat[i*4] = self.x_points[i]
            self.xyz1_flat[i*4 + 1] = self.y_points[i]
            self.xyz1_flat[i*4 + 2] = self.z_points[i]
    
    def plot3d(self, min_max_x = [-10, 10], min_max_y = [-10, 10], min_max_z = [-10, 10], xlabel = 'X', ylabel = 'Y', zlabel = 'Z', label_sizes = [12, 12, 12], **kwargs):
        #kwargs are passed to scatter() via Axes3D.scatter()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(xlabel, size= label_sizes[0])
        ax.set_ylabel(ylabel, size= label_sizes[1])
        ax.set_zlabel(zlabel, size= label_sizes[2])
        ax.set_xlim3d(min_max_x[0], min_max_x[1])
        ax.set_ylim3d(min_max_y[0], min_max_y[1])
        ax.set_zlim3d(min_max_z[0], min_max_z[1])
        sc = ax.scatter(self.x_points, self.y_points, self.z_points, **kwargs)
        plt.show()
    
    def iplot3d(self, func = 'r_update', slider_min_max = [0, 10], min_max_x = [-10, 10], min_max_y = [-10, 10], min_max_z = [-10, 10], xlabel = 'X', ylabel = 'Y', zlabel = 'Z', label_sizes = [12, 12, 12], **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(xlabel, size= label_sizes[0])
        ax.set_ylabel(ylabel, size= label_sizes[1])
        ax.set_zlabel(zlabel, size= label_sizes[2])
        ax.set_xlim3d(min_max_x[0], min_max_x[1])
        ax.set_ylim3d(min_max_y[0], min_max_y[1])
        ax.set_zlim3d(min_max_z[0], min_max_z[1])
        self.sc = ax.scatter(self.x_points, self.y_points, self.z_points, **kwargs)
        slider_ax = plt.axes([0.1, 0.1, 0.8, 0.02])
        slider = Slider(slider_ax, "Offsadsset", slider_min_max[0], slider_min_max[1], valinit=0, color='#AAAAAA')
        func_str = 'self.' + func
        slider.on_changed(eval(func_str))
        plt.show()

    def r_update(self, val):
        #offsets_updated = offsets + array([val, val, val])
        xyz1_r = self.r_transform(val)#apply_rot(rm, theta, val, 0)
        ntot = 4*self.npoints
        xs_ = xyz1_r[0: (ntot -1): 4]
        ys_ = xyz1_r[1: (ntot -1): 4]
        zs_ = xyz1_r[2: (ntot -1): 4]
        self.sc._offsets3d = (xs_, ys_ , zs_ )
        return self.sc
    
    def time_update(self, val):
        i = round(val)
        xs_ = self.x_points_all[i*self.npoints: i*self.npoints + self.npoints]
        ys_ = self.y_points_all[i*self.npoints: i*self.npoints + self.npoints]
        zs_ = self.z_points_all[i*self.npoints: i*self.npoints + self.npoints]
        self.sc._offsets3d = (xs_, ys_ , zs_ )
        return self.sc   
    
    def test(self):
        return self.x_points
    
def drawcoil(npts = 50, ncoils = 3): 
    # generate a coil in 3d points for testing and demonstration
    xs = linspace(-3.14*ncoils, 3.14*ncoils, npts)
    ys = Sin(xs)
    zs = Cos(xs)
    return array([xs, ys, zs])


# In[30]:

points = drawcoil()
c = r3array(points[0], points[1], points[2], 0, 5)
c.rotate([0, 0, 1], [0, 0, 0])
c.r_transform_self(2)
#c.test()
c.get_points(0)



# In[24]:

c.iplot3d(zlabel = 'dadfa', c= 'g', func = 'time_update')


# In[25]:

c.plot3d(c = 'r' , zlabel = 'BFDAF')


# In[3]:

from matplotlib import pyplot as plt
from numpy import genfromtxt, max, min, zeros
from scipy import argmax
path = '/Users/chrisrichards/Desktop/CODE/frogPy/lizard model/clemente_lizard_part1.txt'
dat = genfromtxt(path)


# In[48]:

point = 1 #which point to fix, ordered from proximal to distal
point_map = [3, 6, 7, 2, 4, 5, 1, 0, 8] # to re-order the points
fix_point_i = point_map[point]

#data for input must be arranged as follows:
#rows are time going to ntime rows
#columns are xyz values going to npts points
#(x1 y1 z1 x2 y2 z2 ... xnpts ynpts znpts) for first time point
#...
#...
#(x1 y1 z1 x2 y2 z2 ... xnpts ynpts znpts) for nth time point

npts = dat.shape[1] #number of xyz points*3 i.e. columns
ntime = dat.shape[0] #number of time points i.e. rows

#pull out x, y, and z data as separate arrays
#This is to prepare for input into matplotlib
#plotting functions that want xs, ys, zs as input

xs_raw = dat[:, 0 : npts: 3]
ys_raw = dat[:, 1 : npts: 3]
zs_raw = dat[:, 2 : npts: 3]
flatten_size = dat.shape[0]*npts/3

#### decide which point to fix in space
### so all other points move relative to it
### i.e. shift the origin of the reference frame
### to create local origin at this point


fix_point = array([xs_raw1[fix_point_i], ys_raw1[fix_point_i], zs_raw1[fix_point_i]])

#extract out the x, y and z data for the landmark point
#only ... this is needed to calculate offsets

fixed_point_xs = xs_raw[:, fix_point_i]
fixed_point_ys = ys_raw[:, fix_point_i]
fixed_point_zs = zs_raw[:, fix_point_i]

#organize the data as 2d arrays to make simplify code
# downstream

xs_2d = reshape(xs_raw, (ntime, npts/3))
ys_2d = reshape(ys_raw, (ntime, npts/3))
zs_2d = reshape(zs_raw, (ntime, npts/3))

#cacluate temporal offsets for x, y, z points
#i.e. xyz displacement of the fixed point's
#motion between time 0 and subsequent time points

x_offsets = fixed_point_xs - fixed_point_xs[0]
y_offsets = fixed_point_ys - fixed_point_ys[1]
z_offsets = fixed_point_zs - fixed_point_zs[2]

#create 2d arrays of x, y, z data with the above
#offsets subtracted.  

xs_fixed = zeros((ntime, npts/3))
ys_fixed = zeros((ntime, npts/3))
zs_fixed = zeros((ntime, npts/3))
for i in arange(len(x_offsets)):
    for j in arange(npts/3):
        xs_fixed[i, j] = xs_2d[i, j] - x_offsets[i]
        ys_fixed[i, j] = ys_2d[i, j] - y_offsets[i]
        zs_fixed[i, j] = zs_2d[i, j] - z_offsets[i]

#flatten the x, y and z data so it can be inserted
#into matplotlib functions
xs = ravel(xs_fixed)
ys = ravel(ys_fixed)
zs = ravel(zs_fixed)

#####################################################
#BRUTE FORCE FORCE RANGE INTO SMALLEST CUBE POSSIBLE#
#####################################################

maxX = max(xs)
maxY = max(ys)
maxZ = max(zs)
minX = min(xs)
minY = min(ys)
minZ = min(zs)
rangeX = maxX - minX
rangeY = maxY - minY
rangeZ = maxZ - minZ
max_range = max([rangeX, rangeY, rangeZ])
maxX_box = maxX + 0.5*(abs(max_range - rangeX))
maxY_box = maxY + 0.5*(abs(max_range - rangeY))
maxZ_box = maxZ + 0.5*(abs(max_range - rangeZ))
minX_box = minX - 0.5*(abs(max_range - rangeX))
minY_box = minY - 0.5*(abs(max_range - rangeY))
minZ_box = minZ - 0.5*(abs(max_range - rangeZ))
x_range = [minX_box, maxX_box]
y_range = [minY_box, maxY_box]
z_range = [minZ_box, maxZ_box]
####################################################

highlight_color = 'r'
highlight_point = fix_point_i
colormap = zeros((npts/3, ), dtype = '|S1')
for i in arange(npts/3):
    if i == highlight_point:
        colormap[i] = highlight_color
    else:
        colormap[i] = 'b'

liz = r3array(xs, ys, zs, 0, 9)
#liz.plot3d(min_max_x = [-1000, 1000], min_max_y = [-1000, 1000], min_max_z= [-1000, 1000])
liz.iplot3d(zlabel = 'dadfa', c = colormap, func = 'time_update', slider_min_max = [0, ntime -1], min_max_x = x_range, min_max_y = y_range, min_max_z= [-1000, 1000])


# In[6]:

flatten_size


# In[201]:

fixed_point_xs = xs_raw[:, 6]
fixed_point_ys = ys_raw[:, 6]
fixed_point_zs = zs_raw[:, 6]
x_offsets = fixed_point_xs - fixed_point_xs[0]
y_offsets = fixed_point_ys - fixed_point_ys[0]
z_offsets = fixed_point_zs - fixed_point_zs[0]
xs_fixed = zeros((len(xs_raw1)))
ys_fixed = zeros((len(xs_raw1)))
zs_fixed = zeros((len(xs_raw1)))
for i in arange(len(x_offsets)):
    for j in arange(npts/3):
        xs_fixed = xs_raw1 - x_offsets[i]
        ys_fixed = ys_raw1 - y_offsets[i]
        zs_fixed = zs_raw1 - z_offsets[i]


# In[17]:

xs_raw2 = reshape(xs_raw1, (162, npts/3))
dat.shape


#### 

# In[16]:

reshape(xs_raw, (162, npts/3)).shape


# In[20]:

highlight_color = 'r'
highlight_point = 3
colors = zeros((npts, ), dtype = '|S3')
for i in arange(npts):
    if i == highlight_point:
        colors[i] = highlight_color
    else:
        colors[i] = colors[i]


# In[59]:

for i in arange(npts/3):
    if i == highlight_point:
        print i
    else:
        print 'blah'


# In[52]:

a = [[-10, 4,  10], [-10, 4,10], [-10, 4, 10]]


# In[51]:

b = [[-3, 5, 3], [-4, 5, 4], [-5, 5, 5]]


# In[53]:

concatenate((a, b))


# In[47]:




# In[26]:

from sympy.physics.mechanics import ReferenceFrame, Vector
from sympy import symbols
N = ReferenceFrame('N')
N1 = ReferenceFrame('N1')
N2 = ReferenceFrame('N2')
N3 = ReferenceFrame('N3')
q1,q2,q3 = symbols('q1 q2 q3')
N1.orient(N,'Axis', [q1, N.x])
N2.orient(N1,'Axis', [q2, N1.x])
N3.orient(N2,'Axis', [q3, N2.z])
dot(N3.x, N2.x)


# In[37]:

from numpy import arange
from sympy import Matrix, eye
m = eye(3)
a1, a2, a3, b1, b2, b3 = symbols('a1, a2, a3, b1, b2, b3')
A = m.subs(1, a1)
B = m.subs(1, b1)
for i in arange(3):
    for j in arange(3):
         m[i, j] = (A[i, j]).dot(B[i, j])
    
   
    


# In[116]:

theta = symbols('theta')
offset = 1
Rz = liz.rotate([0, 0, 1], [0, 0, 0], 1)
Rz[0:4, 3] = [[offset],[offset],[offset],[1]]
Rz90 = Rz.subs(theta, 1.57)
point = [2, 0, 0, 1]
Rz90.dot(point)


# In[89]:

B.orient(N, 'Axis', [q1, q2, q3], '123')
B.dcm(N)


# In[74]:




# In[75]:

0


# In[103]:

sl = slice(0, 1, 1)


# In[76]:

0



# In[105]:

slice


# In[107]:

x = 0
if x == 0:
    blah = 2
else:
    blah =4
blah


# In[76]:




# In[76]:




# In[96]:

i = 1
l = 10
a = arange(60)
a[l*i:l*i + l]


# In[1]:

import sympy


# In[2]:

Matrix


# In[8]:

from sympy import Matrix, eye, symbols
from numpy import vstack


# In[7]:

m = eye(3)
m


# In[9]:

x, y, z = symbols('x, y, z')


# In[10]:

m.dot([x, y, z])


# In[14]:

(m.inv()).dot([0, 0, 0])


# In[52]:

a = r3array([1, 0, 0], [0, 1, 0], [0, 0, 1])
R = a.rotate([0, 1, 0], [0, .5, 0], 1)
m1 = a.r_transform_partition(.3, 0)
m2 = a.r_transform_partition(.3, 1)
m3 = a.r_transform_partition(.3, 2)
m = vstack((m1, m2, m3))
mmatrix = eye(3)
for i in arange(3):
    for j in arange(3):
        mmatrix[i, j] = m[i, j] 
(mmatrix.inv()).dot([0, 0, 0])


# In[48]:

mmatrix.inv()


# In[26]:




# In[26]:




# In[70]:

from sympy.physics.mechanics import ReferenceFrame, Vector
q0, q1, q2, q3, q4 = symbols('q0 q1 q2 q3 q4')
N = ReferenceFrame('N')


# In[75]:

N.dcm(N)


# In[26]:




# In[103]:

from sympy.physics.mechanics import ReferenceFrame, Vector
N = ReferenceFrame('N')
x, y , q1 = symbols('x, y, q1')
B = ReferenceFrame('B')
B.orient(N, 'Axis', [q1, N.x])
B.dcm(N)


# In[99]:




# In[44]:

sort


# In[47]:

from scipy import argmax
argmax([3, 6, 7676, 4])


# In[ ]:




# In[ ]:



