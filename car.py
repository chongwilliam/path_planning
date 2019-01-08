import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class car:
    def __init__(self):
        self.r = 0

    pass

class quintic_poly:
    def __init__(self, xs, vs, as, xe, ve, ae):
        self.xs = xs  # starting position
        self.vs = vs  # starting velocity
        self.as = as  # starting acceleration
        self.xe = xe  # ending position
        self.ve = ve  # ending velocity
        self.ae = ae  # ending acceleration

        self.x0 = xs
        self.x1 = vs
        self.x2 = as/2.0

    def calc_poly(self, tf):
        A = np.array([tf**3, tf**4, tf**5],
        [3*t**2, 4*t**3, 5*t**4]),
        [6*t, 12*t**2, 20*t**3])

        b = np.array([xe-self.x0-self.x1*t-self.x2*t**2],
        [ve-self.x1-2*self.x2*t],
        [ae-2*self.x2])

        x_coeff = np.linalg.solve(A, b)
        self.x3 = x_coeff[0]
        self.x4 = x_coeff[1]
        self.x5 = x_coeff[2]

    def first_der(self, t):
        return self.x1 + 2*self.x2*t + 3*self.x3*t**2 + 4*self.x4*t**3 + 5*self.x5*t**4

    def second_der(self, t):
        return 2*self.x2 + 6*self.x3*t + 12*self.x4*t**2 + 20*self.x5*t**3

    def third_der(self, t):
        return 6*self.x3 + 24*self.x4*t + 60*self.x5*t**2

def opt_path(fren_paths, T):
    pass

def gen_path(x, y, opts='cubic'):  # generate path with spline options
    if (len(x) != len(y)):
        print('Different number of x and y points')
    else:
        tck = interpolate.splrep(x, y)
        x_new = np.linspace(0, np.amax(x), 100)
        y_new = interpolate.splev(x_new, tck)
        return x_new, y_new

def obs_detect():
    pass



#main

# way points
wx = np.array([0.0, 10.0, 20.5, 35.0, 71.0])
wy = np.array([0.0, -6.0, 5.0, 6.5, 0.0])
# obstacle lists
ob = np.array([[20.0, 10.0],
               [30.0, 6.0],
               [30.0, 8.0],
               [35.0, 8.0],
               [50.0, 3.0]
               ])

wx_new, wy_new = gen_path(wx, wy)
plt.figure()
plt.plot(wx_new, wy_new)
plt.show()
