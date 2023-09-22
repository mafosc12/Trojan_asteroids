import math
import matplotlib.pyplot as plt
import numpy as np
import time
start_time = time.process_time()

# Get position of Sun
def sun_pos(Msun, Mjup, R):
    sun_x = -Mjup * R / (Msun + Mjup)
    return sun_x

# Get position of Jupiter
def jup_pos(Msun, Mjup, R):
    jup_x = Msun * R / (Msun + Mjup)
    return jup_x

# Get position of L4
def L4_pos(Msun, Mjup, R):
    L4_x = R / 2 * ( (Msun - Mjup) / (Msun + Mjup) )
    return L4_x

def get_ang_vel(Grav, Msun, Mjup, R):
    omega = math.sqrt(Grav * (Msun + Mjup) / (R ** 3))
    return omega

# transpose perturbations out of orbit/in orbit into x,y coordinates
def transpose(init_perp_disp, init_perp_vel, init_ang_disp, init_ang_vel, L4, L4_rad):
    L4_angle = math.atan2(L4[1], L4[0])
    new_angle = L4_angle + init_ang_disp # new angle after perturbations
    new_rad = L4_rad + init_perp_disp  # new radius after perturbations
# displacements
    new_x = new_rad * math.cos(new_angle) # new x coord after perturbations
    new_y = new_rad * math.sin(new_angle)  # new x coord after perturbations
    x_pert = new_x - L4[0]  # x disp in cartesian
    y_pert = new_y - L4[1]  # y disp in cartesian
#
    x_pert = round(x_pert, 5)  # can get error with big decimal
    y_pert = round(y_pert, 5)  # can get error with big decimal

# velocity space
# Perp velocity contribution
    vx_init = init_perp_vel * math.cos(new_angle)
    vy_init = init_perp_vel * math.sin(new_angle)
# ang vel contribution
    vx_angular = -init_ang_vel * new_rad * math.sin(new_angle) # minus as +ve is towards sun
    vy_angular = init_ang_vel * new_rad * math.cos(new_angle)
# get total
    vx_init = vx_init + vx_angular
    vy_init = vy_init + vy_angular
#
    vx_init = round(vx_init, 5)  # can get error with big decimal
    vy_init = round(vy_init, 5)  # can get error with big decimal
#
#
    return x_pert, y_pert, vx_init, vy_init, L4_angle

# 4th Order Runge Kutta
def runge4(t, PV, dt): # note here PV = pos & vel vector
    dPV1 = dPVdt(t, PV)*dt #does dPV/dt on the Vector PV and the old t
    dPV2 = dPVdt(t+0.5*dt, PV+0.5*dPV1)*dt #does dX/dt on the Vector PV+1/2dPV1 and the old t+1/2dt
    dPV3 = dPVdt(t+0.5*dt, PV+0.5*dPV2)*dt # etc..
    dPV4 = dPVdt(t+dt, PV+dPV3)*dt
    return PV+dPV1/6.0+dPV2/3.0+dPV3/3.0+dPV4/6.0 # this returns new updated vector PV


# gets derivative of the pos, vel vector
def dPVdt(t, PV):
    x = PV[0]
    vx = PV[1]
    y = PV[2]
    vy = PV[3]
    z = PV[4]
    vz = PV[5]
#
# GREENSPAN EQUS 2.8, 2.9, 2.10
#
    mjR = x + Mjup * R / (Msun + Mjup)
    msR = x - Msun * R / (Msun + Mjup)
    d1 = math.sqrt((mjR ** 2) + (y ** 2) + (z ** 2))
    d2 = math.sqrt((msR ** 2) + (y ** 2) + (z ** 2))

# Acceleration in x direction
    accx = -Grav * Msun * mjR / (d1 ** 3)   # gravity
    accx = accx - Grav * Mjup * msR / (d2 ** 3)  # gravity
    accx = accx + 2 * omega * vy   # Coriolis
    accx = accx + (omega ** 2) * x # Centripetal
#    print(accx)
# Acceleration in y direction
    accy = -Grav * Msun * y / (d1 ** 3)  # gravity
    accy = accy - Grav * Mjup * y / (d2 ** 3)  # gravity
    accy = accy - 2 * omega * vx   # Coriolis
    accy = accy + (omega ** 2) * y # Centripetal
#    print(accy)
# Acceleration in z direction
    accz = -Grav * Msun * z / (d1 ** 3)  # gravity
    accz = accz - Grav * Mjup * z / (d2 ** 3)  # gravity

    return np.array([vx, accx, vy, accy, vz, accz]) # returns vector of dPV/dt to runge4

def max_min_ang(Max_ang, Min_ang, PV, L4):     # Calculates the max and min wander angle of all timesteps
    x1 = L4[0]  # x pos L4
    y1 = L4[1]  # y pos L4
    x2 = PV[0]  # x pos of timestep location
    y2 = PV[2]  # y pos of timestep location

    dot = (x1 * x2) + (y1 * y2)  # scalar product
    det = (x1 * y2) - (x2 * y1)  # determinant
    ang = math.atan2(det, dot)  # get angle anticlockwise (if > pi will be negative clockwise)
    if ang < 0.0:  # if angle is < 0 make it positive, but only if its below the x-axis (L4 angle is pi/3)
        if ang < -math.pi / 3:
            ang = (2 * math.pi) - abs(ang) ## THIS IS WRONG AS PYTHON IS DIFFERENT
    if ang > Max_ang: # store the maximum ever angle
        Max_ang = ang
    if ang < Min_ang: # store the minimum ever angle
        Min_ang = ang

    return Max_ang, Min_ang

def wand_angle(Max_ang, Min_ang):    # Calculates maximum TOTAL wander angle & distance
    wander_angle = Max_ang - Min_ang
    wander_distance = wander_angle * L4_rad
    return wander_angle, wander_distance

def plots_xy(x_pos, y_pos):  # PLOTS IN X_Y PLANE
    plt.figure(figsize=(6, 6))
#    plt.axis([-0.5, 4.5, 1, 6])
    plt.axis([-6, 6, -6, 6]) # for sun and jupiter
#    plt.axis([2, 3, 4, 5]) # for L4 close = make sure same distance on both axes or girds not equal
    plt.plot(x_pos, y_pos, color='green', marker='o', linewidth=1, markersize=1) # L4
    plt.plot(SUN[1], SUN[2], color='yellow', marker='o', linewidth=0, markersize=40)
    plt.plot(JUP[1], JUP[2], color='blue', marker='o', linewidth=0, markersize=20)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    years = int(t_end)
    text1 = "{} rotations, {} steps/rotation, time steps {}, years {}".format(no_rots, no_ts_rot, nsteps, years)
    plt.text(-5.8, -2.0, text1, fontsize=9) # 0,3.2 is x and y location on axes
    text2 = "radial disp (AU)= {}, orbital disp (rad)= {}, z disp (AU)= {}".format(init_perp_disp,
                                                                                         init_ang_disp, z_pert)
    plt.text(-5.8, -2.4, text2, fontsize=9)  # 0,3.2 is x and y location on axes
    text3 = "radial vel (AU/y) = {}, orbital ang vel (rad/y) = {}, z vel (AU/y) = {}".format(init_perp_vel,
                                                                                                init_ang_vel, vz_init)
    plt.text(-5.8, -2.8, text3, fontsize=9)  # 0,3.2 is x and y location on axes
    text4 = "X-Y Plane: total wander angle (rad) {}".format("%.3f" % wander_angle)
    plt.text(-5.8, -3.2, text4, fontsize=9)  # 0,3.2 is x and y location on axes
    text5 = "X-Y Plane: total wander circumference (AU) {}".format("%.3f" % wander_distance)
    plt.text(-5.8, -3.6, text5, fontsize=9)  # 0,3.2 is x and y location on axes
    text6 = "wander_xy"
    plt.savefig(text6)
    plt.grid()
    plt.show()

def plots_xz(x_pos, z_pos): # PLOTS IN X_Z PLANE
    plt.figure(figsize=(6, 6))
#    plt.axis([2., 3, -.5, 0.5])
    plt.axis([-6, 6, -6, 6]) # for sun and jupiter
#    plt.axis([2, 3, 4, 5]) # for L4 close = make sure same distance on both axes or girds not equal
    plt.plot(x_pos, z_pos, color='green', marker='o', linewidth=1, markersize=1) # L4
    plt.plot(SUN[1], 0, color='yellow', marker='o', linewidth=0, markersize=40) # z = 0 sun
    plt.plot(JUP[1], 0, color='blue', marker='o', linewidth=0, markersize=20) # z = 0 jup
    plt.xlabel('x (AU)')
    plt.ylabel('z (AU)')
    years = int(t_end)
    text1 = "{} rotations, {} steps/rotation, time steps {}, years {}".format(no_rots, no_ts_rot, nsteps, years)
    plt.text(-5.8, -2.0, text1, fontsize=9) # 0,3.2 is x and y location on axes
    text2 = "radial disp (AU)= {}, orbital disp (rad)= {}, z disp (AU)= {}".format(init_perp_disp,
                                                                                         init_ang_disp, z_pert)
    plt.text(-5.8, -2.4, text2, fontsize=9)  # 0,3.2 is x and y location on axes
    text3 = "radial vel (AU/y) = {}, orbital ang vel (rad/y) = {}, z vel (AU/y) = {}".format(init_perp_vel,
                                                                                                init_ang_vel, vz_init)
    plt.text(-5.8, -2.8, text3, fontsize=9)  # 0,3.2 is x and y location on axes
    text4 = "X-Y Plane: total wander angle (rad) {}".format("%.3f" % wander_angle)
    plt.text(-5.8, -3.2, text4, fontsize=9)  # 0,3.2 is x and y location on axes
    text5 = "X-Y Plane: total wander circumference (AU) {}".format("%.3f" % wander_distance)
    plt.text(-5.8, -3.6, text5, fontsize=9)  # 0,3.2 is x and y location on axes
    text6 = "wander_xz"
    plt.savefig(text6)
    plt.grid()
    plt.show()

#
# constants (units are distance = AU, Mass = Sun, time = year)
#
Grav = 4.0 * (math.pi) ** 2
Msun = 1.0 # mass of Sun
Mjup = 0.001 # 0.001 # mass of Jupiter
R = 5.2 # Distance from Sun to Jupiter
#
SUN = np.array([Msun, 0., 0.]) # generates vector for SUN (mass, x, y)
sun_x = sun_pos(Msun, Mjup, R)
SUN[1] = sun_x
SUN[2] = 0
#print(SUN)
#
JUP = np.array([Mjup, 0., 0.]) # generates vector for JUPITER (mass, x, y)
jup_x = jup_pos(Msun, Mjup, R)
JUP[1] = jup_x
JUP[2] = 0
#print(JUP)
#
L4 = np.array([0., 0., 0.]) # generates vector for L4 (x, y, z)
L4_x = L4_pos(Msun, Mjup, R)
L4[0] = L4_x
L4[1] = math.sqrt(3) / 2 * R # y pos of L4
L4[2] = 0
L4_rad = math.sqrt(L4[0]**2 + L4[1]**2) # Radius of L4
#print(L4)
#
omega = get_ang_vel(Grav, Msun, Mjup, R) # ang_vel of system about centre of rotation rad/year
period = 2 * math.pi / omega
#print(period)

#
# initial L4 perturbations and velocities
#
# define initial conditions normal/perpendicular to orbit and in the orbit
# (cylindrical ref frame)
# note for perpendicular +ve is further from sun, -ve is nearer sun
# note for inside orbit +ve is towards sun (anticlockwise), -ve is towards Jup (clockwise)
#
init_perp_disp = 0.03 # initial perturbation out of orbit (in XY plane) IN AU
init_perp_vel = 0.0 # initial velocity out of orbit (in XY plane) IN AU/YEAR
# restricted to +/- pi/3 -->
init_ang_disp = 0.0 # initial perturbation angle along the orbit IN RAD, restricted to +/- pi/3
# restricted to +/- pi/3 -->
init_ang_vel = 0.0 # initial angular velocity along the orbit IN RAD/YEAR, , restricted to +/- pi/3
# out of XY plane:
z_pert = 0.0  # initial perturbation in z direction (out of plane)
vz_init = 0.0  # initial velocity in z direction (out of plane)
#
no_rots = 50 # number of rotations to run through
no_ts_rot = 150 # number of time steps per rotation
dt = period / no_ts_rot  # timestep in years
t_end = no_rots * period  # end time in years
nsteps = int(t_end / dt)
#
# now get the equivalent displacements/velocities in cartesian frame
#
x_pert, y_pert, vx_init, vy_init, L4_angle = transpose(init_perp_disp, init_perp_vel,
                                                       init_ang_disp, init_ang_vel, L4, L4_rad)
#
#
x_pos = [0.0]*nsteps # creates a list of the x positions (nstep element list) for plotting
y_pos = [0.0]*nsteps # creates a list of the y positions (nstep element list) for plotting
z_pos = [0.0]*nsteps # creates a list of the z positions (nstep element list) for plotting

# SOLVE EOM USING 4th ORDER RUNGE
# PV IS VECTOR OF POSITIONS AND VELOCITIES (x, vx, y, vy)
PV = np.array([x_pert+L4[0], vx_init, y_pert+L4[1], vy_init, z_pert+L4[2], vz_init]) # generates vector of x, vx, y, vy, z, vz (and puts in initial values)


##################### MAIN TIMESTEP LOOP ###############################################################
for i in range(nsteps):  # MAIN TIMESTEP LOOP
    x_pos[i] = PV[0]   #Makes ith element of list of x positions = first location in array PV = x position = for plots
    y_pos[i] = PV[2]   #Makes ith element of list of y positions = first location in array PV = y position = for plots
    z_pos[i] = PV[4]   #Makes ith element of list of z positions = first location in array PV = z position = for plots
    # update the vector X to the next time step
    PV = runge4(i*dt, PV, dt) # calls the runge kutta with old time, old PV and dt and returns new PV
    # Get max and min wander angle in x-y plane
    if i == 0:
        Max_ang = 0.0
        Min_ang = 0.0
    Max_ang, Min_ang = max_min_ang(Max_ang, Min_ang, PV, L4)
#####################################################################################

# Get wander angle from max and min x locations (IN X-Y PLANE)
wander_angle, wander_distance = wand_angle(Max_ang, Min_ang)
print('Total wander angle (rad):',"%.3f" % wander_angle)
print('Total wander distance circumference (AU):',"%.3f" % wander_distance)

# Do plots in X-Y PLANE
plots_xy(x_pos, y_pos)

# Do plots in X-Z PLANE
plots_xz(x_pos, z_pos)
#print(z_pos)

# Get CPU time
end_time = time.process_time()
cpu_time = end_time - start_time
print('CPU time:', cpu_time, 's')

print(x_pert)
print(y_pert)
print(vx_init)
print(vy_init)
print(L4)
print(L4_rad)
print(L4_angle)
