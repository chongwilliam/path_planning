import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# global parameters
kj = 1.0
kt = 0.1
kd = 0.1
ks_d = 0.1

sd = 1.0  # desired speed (m/s)

d_min = 0.0
d_max = 5.0  # 5 meters max road width
t_min = 4.0  # seconds
t_max = 5.0

d_samples = 50  # sampling within range
t_samples = 50

max_speed = sd*5
max_accel = 0.0
max_curvature = 1.0

class car:
    def __init__(self, x0, y0, theta_0):
        self.x = x0
        self.y = y0
        self.theta = theta_0

class quintic_poly:
    def __init__(self, xs, vs, acs, xe, ve, ae):
        self.xs = xs  # starting position
        self.vs = vs  # starting velocity
        self.acs = acs  # starting acceleration
        self.xe = xe  # ending position
        self.ve = ve  # ending velocity
        self.ae = ae  # ending acceleration

        self.x0 = xs
        self.x1 = vs
        self.x2 = acs/2.0

    def calc_poly(self, tf):
        A = np.array([tf**3, tf**4, tf**5],
        [3*t**2, 4*t**3, 5*t**4],
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

    def third_der(self, T, dt):
        t = np.linspace(0, T, T/dt)
        points = 6*self.x3 + 24*self.x4*t + 60*self.x5*t**2
        return t, points

    def calc_points(self, T, dt):
        t = np.linspace(0, T, T/dt)
        points = self.x0 + self.x1*t + self.x2*t**2 + self.x3*t**3 + self.x4*t**4 + self.x5*t**5
        return t, points

def calc_path(lat_path, long_path, T, dt, opts='high'):
    if (opts == 'high'):
        # lateral movement (high speed trajectory)
        t, x = lat_path.third_der(T, dt)
        lat_high_jerk = np.trapz(x, t)
        lat_high_cost = kj*lat_high_jerk + kt*T + kd*x[-1]**2
    elif (opts == 'low'):
        # lateral movement (low speed trajectory)
        s, x = lat_path.third_der(S, ds)
        lat_low_jerk = np.trapz(x, s)
        lat_low_cost = kj*lat_low_jerk + kt*S + kd*x[-1]**2

    # longitudinal movement (velocity keeping path follower)
    t, x = long_path.third_der(T, dt)
    long_path_jerk = np.trapz(x, t)
    long_path_cost = kj*long_path_jerk + kt*T + ks_d*(long_path.ve - sd)**2

def gen_path(x, y, opts='cubic'):  # generate path with spline options
    if (len(x) != len(y)):
        print('Different number of x and y points')
    else:
        tck = interpolate.splrep(x, y)
        x_new = np.linspace(0, np.amax(x), 100)
        y_new = interpolate.splev(x_new, tck)
        y_der = interpolate.splev(x_new, tck, 1)
        return x_new, y_new, np.arctan(y_der)

def opt_path(x_12, y_12,  ):
    lat_paths = []
    long_paths = []
    # lateral variation
    for d in np.linspace(d_min, d_max, d_samples):
        for t in np.linspace(t_min, t_max, t_samples):
            lat_paths.append(quintic_poly(xs, vs, acs, xe, 0.0, 0.0))




            

def cart2fren(wx, wy, heading):
    s = np.zeros_like(wx)
    d = np.zeros_like(wx)
    for i in range(len(wx)):
        s[i] = wx[i]*np.cos(heading) + wy[i]*np.sin(heading)
        d[i] = wx[i]*np.sin(heading) + wy[i]*np.cos(heading)
    return s, d

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

wx_new, wy_new, heading = gen_path(wx, wy)
plt.figure()
plt.plot(wx_new, wy_new)
for i in range(len(tan_vector)):
    plt.arrow(wx_new[i], wy_new[i], 5*np.cos(tan_vector[i]), 5*np.sin(tan_vector[i]))
plt.show()
