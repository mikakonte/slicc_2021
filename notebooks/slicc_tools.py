'''
slicc_tools.py
Last updated: 21 June 2021
'''
import numpy as np
from matplotlib import pyplot as plt


def runge_kutta(x_i, func, dt = 0.1):
    '''
    Fourth-order Runge-Kutta method that takes in the arguments:
    x_i = initial condition in vector form (numpy array)
    func = function governing the derivative of one of the variables
    dt = time step / step size, defaults to 0.1
    '''
    
    #Calculate the intermediary parameter values
    k_1 = func(x_i) * dt
    k_2 = func(x_i + 1/2 * k_1) * dt
    k_3 = func(x_i + 1/2 * k_2) * dt
    k_4 = func(x_i + k_3) * dt
    
    #Calculate the final condition of the system
    x_f = x_i + 1/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    
    #Return the final condition in vector form
    return x_f


def get_real_grid(x_min, x_max, num):
    '''
    Returns a square xy grid grid in a given interval with arguments:
    x_min, x_max = boundaries along one axis
    num = number of grid points along one axis, i.e. total number of points = num * num
    '''
    
    #Create arrays of xy coordinate values
    x = np.linspace(x_min, x_max, num)
    y = np.linspace(x_min, x_max, num)
    
    return [[a, b] for a in x for b in y]


def plot_direction_field(func, x_min, x_max, num, d1 = 12, d2 = 10):
    '''
    Plots the direction field of a given function with arguments:
    func = governing equation which should return a 2D vector with x_deriv, y_deriv
    x_min, x_max = the graphing boundaries along one axis
    num = number of grid points along one axis, i.e. total number of points = num * num
    d1, d2 = dimensions of the pyplot figure, defaults to 12x10
    '''
    
    #Format figure for plotting
    plt.figure(figsize=(d1, d2))

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.xticks(range(x_min, x_max+1), range(x_min, x_max+1))
    plt.yticks(range(x_min, x_max+1), range(x_min, x_max+1))

    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    
    #Create loop for plotting the direction field
    for x in get_real_grid(x_min, x_max, num):
        #Plot arrows with length representing the derivative at a given point
        plt.arrow(x[0], x[1], 0.02*func(x)[0], 0.02*func(x)[1], \
                  width = 0.001, head_width = 0.05, ec = 'k', fc = 'k', alpha = 1)
    
    return


def plot_trajectory(func, x_0, t, dt = 0.1, c = 'r'):
    '''
    Plots a single trajectory of a system from a given initial condition with arguments:
    func = governing equation which should return a 2D vector with x_deriv, y_deriv
    x_0 = initial condition in 2D vector form
    t = number of steps to graph
    dt = time step / step size, defaults to 0.1
    c = colour of the graph, defaults to red
    '''
    
    #Use plot instead of scatter for increased efficiency
    for step in range(0, t+1):
        plt.plot(x_0[0], x_0[1], 'o', ms = 3, color = c, alpha = 1)
    
        #Retrieve the next value of the system using fourth-order Runge-Kutta
        x_0 = runge_kutta(x_0, func, dt)
    
    return


def lorenz(v, r, sigma, b):
    '''
    The governing equations of the Lorenz system for further use
    '''
    x, y, z = v[0], v[1], v[2]
    
    x_deriv = sigma * (y - x)
    y_deriv = r*x - y - x*z
    z_deriv = x*y - b*z
    
    return np.array([x_deriv, y_deriv, z_deriv])


def lorenz_values(r, sigma, b, t, time_step = 0.01, transient = 0, v_0 = [0, 1, 0]):
    '''
    Returns lists of the (x, y, z) and t values of the Lorenz system with the given arguments:
    r = the Rayleigh number
    sigma = the Prandtl number
    b = the value of parameter b
    t = length of time over which the system is iterated
    time_step = length of a single time step used for iteration; defaults to 0.01
    transient = duration deleted from the beginning of the lists to remove a transient
    v_0 = initial condition in (x, y, z) format; defaults to (0, 1, 0)
    '''
    
    #Create lists of the (x, y, z) and time values
    dim_values = [[], [], []]
    t_values = []
    
    #Define the Lorenz equations with fixed parameters
    def lorenz_var(v):
        return lorenz(v, r = r, sigma = sigma, b = b)
    
    #Declare initial condition
    v = v_0
    
    #Iterate over time steps
    for step in np.arange(0, t, time_step):
        t_values.append(step)
        
        #Add new values of (x, y, z) to the list dim_values
        for m in range(3):
            dim_values[m].append(v[m])
     
        #Retrieve the next coordinates using fourth-order Runge-Kutta
        v = runge_kutta(v, lorenz_var, dt = time_step)
    
    return dim_values, t_values


def logistic(r, x_input = np.linspace(0, 1, 100)):
    '''
    Returns a list of the x_n values after one iteration of the logistic equation with the arguments:
    r = the chosen value of parameter r
    x_input = list of x values to use as initial conditions; defaults to 100 evenly spaced values in the interval [0, 1]
    '''
    
    #Create an empty list to hold x_n values
    x_n = []
    
    #Iterate over the given x_input list
    for x in x_input:
        x_n.append(r * x * (1 - x))
             
    return x_n
        

def logistic_map(S, N, r_lim = [1, 4], x_input = np.random.uniform(0, 1)):
    '''
    Returns two lists (r, x) of the logistic map with the arguments:
    S = the number of r values used for plotting
    N = the number of iterations per value of r
    x_input = the initial value of x; defaults to a random value between 0 and 1
    '''
    
    #Create list of r values in the r_lim range; defaults to [0, 4]
    r_values = np.linspace(r_lim[0], r_lim[1], S)
    
    #Declare empty list of x values
    x_values = []
    
    #Iterate system N times for each value of r
    for r in r_values:
        i = 0
        
        #Choose randomised x value as the initial condition
        x = x_input
        while i < N:
            x = r * x * (1 - x)
            i += 1
        x_values.append(x)
    
    return r_values, x_values