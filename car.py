import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# global parameters
kj = 1.0
kt = 0.1
kd = 0.1
ks_d = 0.1

sd = 1.0  # desired speed (m/s)

ds_min = -0.1  # min delta speed
ds_max = 0.1
d_min = -1.0  # assume center lane with 3 lanes
d_max = 1.0  # 5 meters max road width
t_min = 1  # seconds
t_max = 2

s_samples = 50  # samples within range
d_samples = 50
t_samples = 50

# max checks
max_speed = sd*5
max_accel = 0.0
max_curvature = 1.0

class quartic_poly:
    def __init__(self, xs, vs, acs, ve, ae, tf):  # start (pos, vel, acc), end (vel, acc)
        self.xs = xs
        self.vs = vs
        self.acs = acs
        self.ve = ve
        self.ae = ae

        self.x0 = xs
        self.x1 = vs
        self.x2 = acs/2.0

        self.cost = 0
        self.t = tf

        self.points = np.zeros(5)

        A = np.array([[3.0*tf**2, 4.0*tf**3],
        [6.0*tf, 12.0*tf**2]])

        b = np.array([ve - self.x1 - 2.0*self.x2*tf,
        ae - 2.0*self.x2])

        x_coeff = np.linalg.solve(A,b)
        self.x3 = x_coeff[0]
        self.x4 = x_coeff[1]

        self.t_pts, self.points = self.calc_all_points(tf)

    def calc_poly(self, tf):

        self.points = np.array([self.calc_point(0.0), self.calc_point(2.0*tf/5.0),
        self.calc_point(3.0*tf/5.0), self.calc_point(4.0*tf/5.0), self.calc_point(5.0*tf/5.0)])

    def calc_point(self, t):
        return self.x0 + self.x1*t + self.x2*t**2 + self.x3*t**3 + self.x4*t**4

    def calc_all_points(self, T, n_pts=50):  # overloaded for jerk calc
        t = np.linspace(0, T, n_pts)
        points = self.x0 + self.x1*t + self.x2*t**2 + self.x3*t**3 + self.x4*t**4
        return t, points

    def first_der(self, t):
        return self.x1 + 2.0*self.x2*t + 3.0*self.x3*t**2 + 4.0*self.x4*t**3

    def second_der(self, t):
        return 2.0*self.x2 + 6.0*self.x3*t + 12.0*self.x4*t**2

    def third_der(self, t):
        return 6.0*self.x3 + 24.0*self.x4*t

    def third_der_jerk(self, T, n_pts=50):  # overloaded for jerk calc
        t = np.linspace(0, T, n_pts)
        points = 6.0*self.x3 + 24.0*self.x4*t
        return t, points

class quintic_poly:
    def __init__(self, xs, vs, acs, xe, ve, ae, tf):  # start (pos, vel, acc), end (pos, vel, acc)
        self.xs = xs  # starting position
        self.vs = vs  # starting velocity
        self.acs = acs  # starting acceleration
        self.xe = xe  # ending position
        self.ve = ve  # ending velocity
        self.ae = ae  # ending acceleration

        self.x0 = xs
        self.x1 = vs
        self.x2 = acs/2.0

        self.cost = 0
        self.t = tf

        A = np.array([[tf**3, tf**4, tf**5],
        [3.0*tf**2, 4.0*tf**3, 5.0*tf**4],
        [6.0*tf, 12.0*tf**2, 20.0*tf**3]])

        b = np.array([xe-self.x0-self.x1*tf-self.x2*tf**2,
        ve-self.x1-2.0*self.x2*tf,
        ae-2.0*self.x2])

        x_coeff = np.linalg.solve(A, b)
        self.x3 = x_coeff[0]
        self.x4 = x_coeff[1]
        self.x5 = x_coeff[2]

        self.t_pts, self.points = self.calc_all_points(tf)

    def calc_poly(self, tf):
        self.points = np.array([self.calc_point(0.0), self.calc_point(2.0*tf/5.0),
        self.calc_point(3.0*tf/5.0), self.calc_point(4.0*tf/5.0), self.calc_point(tf)])

    def calc_point(self, t):
        return self.x0 + self.x1*t + self.x2*t**2 + self.x3*t**3 + self.x4*t**4 + self.x5*t**5

    def calc_all_points(self, T, n_pts=50):  # overloaded for jerk calc
        t = np.linspace(0, T, n_pts)
        points = self.x0 + self.x1*t + self.x2*t**2 + self.x3*t**3 + self.x4*t**4 + self.x5*t**5
        return t, points

    def first_der(self, t):
        return self.x1 + 2.0*self.x2*t + 3.0*self.x3*t**2 + 4.0*self.x4*t**3 + 5.0*self.x5*t**4

    def second_der(self, t):
        return 2.0*self.x2 + 6.0*self.x3*t + 12.0*self.x4*t**2 + 20.0*self.x5*t**3

    def third_der(self,t):
        return 6.0*self.x3 + 24.0*self.x4*t + 60.0*self.x5*t**2

    def third_der_jerk(self, T, n_pts=50):
        t = np.linspace(0, T, n_pts)
        points = 6.0*self.x3 + 24.0*self.x4*t + 60.0*self.x5*t**2
        return t, points

def calc_path(fren_path, T, opts):
    if (opts == 'lat_high'):
        # lateral movement (high speed trajectory)
        t, x = fren_path.third_der_jerk(T)
        lat_high_jerk = abs(np.trapz(x, t))
        lat_high_cost = kj*lat_high_jerk + kt*T + kd*fren_path.points[-1]**2
        return lat_high_cost
    elif (opts == 'lat_low'):
        # lateral movement (low speed trajectory)
        s, x = fren_path.third_der_jerk(S)
        lat_low_jerk = abs(np.trapz(x, s))
        lat_low_cost = kj*lat_low_jerk + kt*S + kd*fren_path.points[-1]**2
        return lat_low_cost
    elif (opts == 'long'):
        # longitudinal movement (velocity keeping path follower)
        t, x = fren_path.third_der_jerk(T)
        long_path_jerk = abs(np.trapz(x, t))
        long_path_cost = kj*long_path_jerk + kt*T + ks_d*(fren_path.ve - sd)**2
        return long_path_cost

def opt_path(s_start, d_start, s_end, d_end):  # s_start = [s0, v0, a0] s_end = [ve, ae] d_start = [d0, v0, a0] d_end = [de, ve, ae]
    lat_paths = []
    long_paths = []

    # lateral variation
    for d in np.linspace(d_min, d_max, d_samples):
        for t in np.linspace(t_min, t_max, t_samples):
            new_lat = quintic_poly(d_start[0], d_start[1], d_start[2], d_end[0], 0.0, 0.0, t)
            # new_lat.calc_poly(t)
            # new_lat.cost = calc_path(new_lat, t, t/t_samples, 'lat_high')
            new_lat.cost = calc_path(new_lat, t, 'lat_high')
            lat_paths.append(new_lat)

    # longitudinal variatoin
    for s in np.linspace(ds_min, ds_max, s_samples):
        for t in np.linspace(t_min, t_max, t_samples):
            # s_end[0] = s_start[0] + s_start[1]*t + 0.5*s_start[2]*t**2
            new_long = quartic_poly(s_start[0], s_start[1], s_start[2], s_end[0], 0.0, t)
            # new_long.calc_poly(t)
            # new_long.cost = calc_path(new_long, t, t/t_samples, 'long')
            new_long.cost = calc_path(new_long, t, 'long')
            long_paths.append(new_long)

    # compare cost
    lat_min = 1e5
    long_min = 1e5

    best_lat = lat_paths[0]
    best_long = long_paths[0]

    for i in range(len(lat_paths)):
        if (lat_paths[i].cost <= lat_min):
            lat_min = lat_paths[i].cost
            best_lat = lat_paths[i]

        if (long_paths[i].cost <= long_min):
            long_min = long_paths[i].cost
            best_long = long_paths[i]

    return best_lat, best_long

# map generation and safety checks
def gen_path(x, y, opts='cubic'):  # generate path with spline options
    if (len(x) != len(y)):
        print('Different number of x and y points')
    else:
        tck = interpolate.splrep(x, y)
        x_new = np.linspace(0, np.amax(x), 100)
        y_new = interpolate.splev(x_new, tck)
        y_der = interpolate.splev(x_new, tck, 1)
        arc_len = np.zeros_like(x_new)

        for i in range(len(x_new)-1):  # obtain arc length for each x,y coordinate
            arc_len[i+1] = arc_len[i] + np.sqrt((x_new[i+1]-x_new[i])**2 + (y_new[i+1]-y_new[i])**2)

        return x_new, y_new, np.arctan(y_der), arc_len

def fren2cart(s, d, x, y, heading, arc_len):  # s, d given map
    x_fren = np.zeros_like(s)
    y_fren = np.zeros_like(s)
    for i in range(np.size(s)):
        idx = np.abs(arc_len - s[i]).argmin()  # get closest x and y coordinates from given s arc length
        yaw = heading[idx]  # get heading angle to derive tangent and normal vectors
        x_fren[i] = x[idx] - d[i]*np.sin(yaw)
        y_fren[i] = y[idx] + d[i]*np.cos(yaw)
        # x[i] = s[i]*np.cos(heading[i]) - d[i]*np.sin(heading[i])
        # y[i] = s[i]*np.sin(heading[i]) + d[i]*np.cos(heading[i])
    return x_fren, y_fren

def obs_detect(x, y, obs):
    pass

def max_curve():
    pass

# MAIN
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

wx_new, wy_new, heading, s = gen_path(wx, wy)
# d = np.zeros_like(s)
# print(np.size(wx_new))
#
# plt.figure()
# plt.plot(wx_new, wy_new,'bo')
# for i in range(len(heading)):
#     plt.arrow(wx_new[i], wy_new[i], 5*np.cos(heading[i]), 5*np.sin(heading[i]))
# plt.show()

# plt.figure()  # plot arc length
# plt.plot(s)

# d_test = np.ones_like(s)*(-0.1)
# x_fren, y_fren = fren2cart(s, d_test, wx_new, wy_new, heading, s) # recover x, y points to verify function
# plt.figure()
# plt.plot(wx_new, wy_new, 'b')
# plt.plot(x_fren, y_fren, 'r')
# plt.show()

c_speed = 1.0  # current speed [m/s]
c_d = 1.0  # current lateral position [m]
c_d_d = 0.0  # current lateral speed [m/s]
c_d_dd = 0.0  # current lateral acceleration [m/s^2]
s0 = 0.0  # current course position

s_start = [s0, 0.0, 0.0]
d_start = [c_d, c_d_d, c_d_dd]
i = 0  # counter
t = 0  # loop timer
# s_end = [s[1], 0.0, 0.0]
# d_end = [0.0, 0.0, 0.0]

x_path = np.zeros(1)
y_path = np.zeros(1)

plt.figure()

while(1):
    s_end = [s[i+1], 0.0, 0.0]  # desired end conditions
    d_end = [0.0, 0.0, 0.0]

    print('start condition', s_start, d_start)
    print('end condition', s_end, d_end)
    # plt.pause(0.1)

    lat_path, long_path = opt_path(s_start, d_start, s_end, d_end)

    # re-assign start points
    s_start = [long_path.points[-1], long_path.first_der(long_path.t), long_path.second_der(long_path.t)]
    d_start = [lat_path.points[-1], lat_path.first_der(lat_path.t), lat_path.second_der(lat_path.t)]

    # update the graph with the (x,y) points from the lat/long trajectory (splined)
    x_fren, y_fren = fren2cart(long_path.points, lat_path.points, wx_new, wy_new, heading, s)
    x_path = np.append(x_path, x_fren)
    y_path = np.append(y_path, y_fren)

    i = i + 1  # increase loop counter
    t = t + long_path.t  # loop timer based on the time length of the path planned

    if (np.sqrt((wx_new[-1]-x_fren[-1])**2 + (wy_new[-1]-y_fren[-1])**2) <= 1e-2):
        print('Reached End Goal')
        break
    elif i >= 1e4:
        print('Time Ran Out')
        break

    # update graph
    # print(x_path)
    # print(y_path)
    plt.plot(x_path, y_path,'k')
    # plt.plot(wx_new, wy_new,'b')
    plt.grid(True)
    plt.pause(0.1)
    # plt.show()
