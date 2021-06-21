'''
slicc_tools.py
Last updated: 21 June 2021
'''
import numpy as np

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

def plot_direction_field(x_min, x_max, num, func):
    '''
    Plot the direction field of a given function with arguments:
    x_min, x_max = the graphing boundaries along one axis
    num = number of grid points along one axis, i.e. total number of points = num * num
    func = governing equation which should return a 2D vector with x_deriv, y_deriv
    '''
    
    #Format figure for plotting
    plt.figure(figsize=(12, 10))

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.xticks(range(-x_min, x_max+1), range(-x_min, x_max+1))
    plt.yticks(range(-x_min, x_max+1), range(-x_min, x_max+1))

    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    
    #Create loop for plotting the direction field
    for x in get_real_grid(x_min, x_max, num):
        #Plot arrows with length representing the derivative at a given point
        plt.arrow(x[0], x[1], 0.02*func(x)[0], 0.02*func(x)[1], \
                  width = 0.001, head_width = 0.05, ec = 'k', fc = 'k', alpha = 1)
    
    return

def plot_trajectory(x_0, t, dt = 0.1, c = 'r', func):
    '''
    Plot a single trajectory of a system from a given initial condition with arguments:
    x_0 = initial condition in 2D vector form
    t = number of steps to graph
    dt = time step / step size, defaults to 0.1
    c = colour of the graph, defaults to red
    func = governing equation which should return a 2D vector with x_deriv, y_deriv
    '''
    
    #Use plot instead of scatter for increased efficiency
    for step in range(0, t+1):
        plt.plot(x_0[0], x_0[1], 'o', ms = 3, color = c, alpha = 1)
    
        #Retrieve the next value of the system using fourth-order Runge-Kutta
        x_0 = runge_kutta(x_0, func, dt)
    
    return