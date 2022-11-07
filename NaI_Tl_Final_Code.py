#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.optimize import curve_fit
from numpy import inf as INF

# folder path
dir_path = r"C:\Users\Home\Desktop\Lab Data Master\SI Detector 0"

# to store file names
res = []

# construct path object
d = pathlib.Path(dir_path)

# iterate directory
for entry in d.iterdir():
    # check if it a file
    if entry.is_file():
        res.append(entry)
        
CHANNELS = np.array(list(range(1024)))
BACKGROUND = (np.genfromtxt(res[2],skip_header=12,skip_footer=15)/2289)

def Isotope_Choice(x):
    if x == 0:
        COUNTS = abs(((np.genfromtxt(res[x],skip_header=12,skip_footer=14))/120) - BACKGROUND)
        Roi_Start = 30
        Roi_End = 60
        Roi_Start_2 = 0
        Roi_End_2 = 0
        Top = 100
    if x == 4:
        COUNTS = abs(((np.genfromtxt(res[x],skip_header=12,skip_footer=14))/300) - BACKGROUND)
        Roi_Start = 370
        Roi_End = 500
        Roi_Start_2 = 0
        Roi_End_2 = 0
        Top = 8
    if x == 1:
        COUNTS = abs(((np.genfromtxt(res[x],skip_header=12,skip_footer=15))/647) - BACKGROUND)
        Roi_Start = 5
        Roi_End = 35
        Roi_Start_2 = 205
        Roi_End_2 = 270
        Top = 17.5
    if x == 3:
        COUNTS = abs(((np.genfromtxt(res[x],skip_header=12,skip_footer=14))/1804) - BACKGROUND)
        Roi_Start = 699
        Roi_End = 850
        Roi_Start_2 = 0
        Roi_End_2 = 0
        Top = 0.06
    return (COUNTS, Roi_Start, Roi_End, Roi_Start_2, Roi_End_2, Top)

#Am:res[0]
#Ba:res[1]
#Co:res[3]
#Cs:res[4]

COUNTS,Roi_Start, Roi_End, Roi_Start_2, Roi_End_2, Top = Isotope_Choice(4)

def plot_spectrum(ax, channels, counts, 
                  xlabel='Channels', ylabel='Counts/Second', 
                  **kwargs):
    """Helper function to plot spectra."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    return ax.scatter(channels, counts, **kwargs)


TWO_PI = np.pi * 2.

def gaussian(x, mu, sig, a):
    return a * np.exp(-0.5 * (x-mu)**2 / sig**2) / np.sqrt(TWO_PI * sig**2)

def double_gaussian(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gaussian(x, mu1, sigma1, A1) + gaussian(x, mu2, sigma2, A2)

def in_interval(x, xmin=-INF, xmax=INF):
    """Boolean mask with value True for x in [xmin, xmax) ."""
    _x = np.asarray(x)
    return np.logical_and(xmin <= _x, _x < xmax) 

def filter_in_interval(x, y, xmin, xmax):
    """Selects only elements of x and y where xmin <= x < xmax."""
    # TODO: check x.shape == y.shape?
    _mask = in_interval(x, xmin, xmax)
    return [np.asarray(x)[_mask] for x in (x, y)]

def colourmask(x, xmin=-INF, xmax=INF, cin='red', cout='gray'):
    """Colour cin if within region of interest, cout otherwise."""
    # compute mask as integers 0 or 1
    _mask = np.array(in_interval(x, xmin, xmax), dtype=int)

    # convert to colours
    colourmap = np.array([cout, cin])
    return colourmap[_mask]

def simple_model_fit(model, channels, counts, roi, **kwargs):
    """Least squares estimate of model parameters."""
    # select relevant channels & counts
    _channels, _counts = filter_in_interval(channels, counts, *roi)
    
    # fit the model to the data
    popt, pcov = curve_fit(model, _channels, _counts, **kwargs)
    return popt, pcov

def format_result(params, popt, pcov):
    """Display parameter best estimates and uncertainties."""
    # extract the uncertainties from the covariance matrix
    perr = np.sqrt(np.diag(pcov))
    
    # format parameters with best estimates and uncertainties
    # TODO: should probably round these to a sensible precision! 
    _lines = (f"{p} = {o} ± {e}" for p, o, e in zip(params, popt, perr))
    return "\n".join(_lines)

def plot_model(ax, model, xrange, ps, npoints=1001, **kwargs):
    """Plots a 1d model on an Axes smoothly over xrange."""
    _channels = np.linspace(*xrange, npoints)
    _counts   = model(_channels, *ps)
    
    return ax.plot(_channels, _counts, **kwargs)

def first_moment(x, y):
    return np.sum(x * y) / np.sum(y)
    
def second_moment(x, y):
    x0 = first_moment(x, y)
    return np.sum((x-x0)**2 * y) / np.sum(y)

def gaussian_initial_estimates1(channels, counts):
    """Estimates of the three parameters of the gaussian distribution."""
    mu0 = first_moment(channels, counts)
    sig0 = np.sqrt(second_moment(channels, counts))
    a0 = np.sum(counts)
    
    return (mu0, sig0, a0)

def gaussian_initial_estimates2(channels, counts):
    """Estimates of the three parameters of the gaussian distribution."""
    mu0 = first_moment(channels[0], counts[-1])
    mu1 = first_moment(channels[-1], counts[-1])
    sig0 = np.sqrt(second_moment(channels, counts))
    sig1 = np.sqrt(second_moment(channels, counts))
    a0 = np.sum(counts)
    a1 = np.sum(counts)
    
    return (mu0, sig0, a0, mu1, sig1, a1)

def plot_single(x, y, z):

    GAUSSIAN_PARAMS = ('mu', 'sig', 'a')
    ROI = (x, y)

    # make initial estimates
    _channels, _counts = filter_in_interval(CHANNELS, COUNTS, *ROI)
    _p0 = gaussian_initial_estimates1(_channels, _counts)

    # show the initial guesses
    print("> the initial estimates:")
    print("\n".join(f"{p} = {o}" for p, o in zip(GAUSSIAN_PARAMS, _p0)))

    # do the fit
    popt, pcov = simple_model_fit(gaussian, CHANNELS, COUNTS, ROI, p0=_p0)

    # display result
    print("> the final fitted estimates:")
    print(format_result(GAUSSIAN_PARAMS, popt, pcov))

    fig, ax = plt.subplots(1)
    colours = colourmask(CHANNELS, xmin=x, xmax=y)

    # plot the data, showing the ROI 
    plot_spectrum(ax, CHANNELS, COUNTS, c=colours, marker='+')

    # plot the model with its parameters
    plot_model(ax, gaussian, (x, y), popt, c='k')
    # puts a limit on the range of x (Chanels)
    plt.xlim(x-30,y+30)
    plt.ylim(top=z)
    print("Peak counts:",gaussian(popt[0], popt[0], popt[1], popt[2]))
    
def plot_two(x1, y1, x2, y2, z):

    GAUSSIAN_PARAMS = ('mu', 'sig', 'a')
    ROI_1 = (x1, y1)
    ROI_2 = (x2, y2)

    # make initial estimates
    _channels, _counts = filter_in_interval(CHANNELS, COUNTS, *ROI_1)
    _p0 = gaussian_initial_estimates1(_channels, _counts)

    # show the initial guesses
    print("> the initial estimates:")
    print("\n".join(f"{p} = {o}" for p, o in zip(GAUSSIAN_PARAMS, _p0)))

    # do the fit
    popt, pcov = simple_model_fit(gaussian, CHANNELS, COUNTS, ROI_1, p0=_p0)

    # display result
    print("> the final fitted estimates:")
    print(format_result(GAUSSIAN_PARAMS, popt, pcov))

    fig, ax = plt.subplots(1)
    colours = colourmask(CHANNELS, xmin=x1, xmax=y2)

    # plot the data, showing the ROI 
    plot_spectrum(ax, CHANNELS, COUNTS, c=colours, marker='+')

    # plot the model with its parameters
    plot_model(ax, gaussian, (x1, y1), popt, c='k')
    # puts a limit on the range of x (Chanels)
    plt.xlim(x1-30,y2+30)
    plt.ylim(top=z)
    print("Peak counts:",gaussian(popt[0], popt[0], popt[1], popt[2]))
    
    _channels, _counts = filter_in_interval(CHANNELS, COUNTS, *ROI_2)
    _p0 = gaussian_initial_estimates1(_channels, _counts)

    # show the initial guesses
    print("> the initial estimates:")
    print("\n".join(f"{p} = {o}" for p, o in zip(GAUSSIAN_PARAMS, _p0)))

    # do the fit
    popt, pcov = simple_model_fit(gaussian, CHANNELS, COUNTS, ROI_2, p0=_p0)

    # display result
    print("\n> the final fitted estimates:")
    print(format_result(GAUSSIAN_PARAMS, popt, pcov))
    
    plot_model(ax, gaussian, (x2, y2), popt, c='k')
    print("Peak counts:",gaussian(popt[0], popt[0], popt[1], popt[2]))
    
def plot_double(x, y, z):

    GAUSSIAN_PARAMS = ('mu', 'sig', 'a','mu2', 'sig2', 'a2')
    ROI = (x, y)

    # make initial estimates
    _channels, _counts = filter_in_interval(CHANNELS, COUNTS, *ROI)
    _p0 = gaussian_initial_estimates2(_channels, _counts)

    # show the initial guesses
    print("> the initial estimates:")
    print("\n".join(f"{p} = {o}" for p, o in zip(GAUSSIAN_PARAMS, _p0)))

    # do the fit
    popt, pcov = simple_model_fit(double_gaussian, CHANNELS, COUNTS, ROI, p0=_p0)

    # display result
    print("> the final fitted estimates:")
    print(format_result(GAUSSIAN_PARAMS, popt, pcov))

    fig, ax = plt.subplots(1)
    colours = colourmask(CHANNELS, xmin=x, xmax=y)

    # plot the data, showing the ROI 
    plot_spectrum(ax, CHANNELS, COUNTS, c=colours, marker='+')

    # plot the model with its parameters
    plot_model(ax, double_gaussian, (x-30,y+30), popt, c='k')
    # puts a limit on the range of x (Chanels)
    plt.xlim(x-50,y+50)
    plt.ylim(top=z)
    print("Peak counts 1:",gaussian(popt[0], popt[0], popt[1], popt[2]))
    print("Peak counts 2:",gaussian(popt[3], popt[3], popt[4], popt[5]))

#Used to view the full spectrum and find ROI
fig, ax = plt.subplots(1)
plot_spectrum(ax, CHANNELS, COUNTS, marker='+')
#plt.xlim(0,400)
#plt.ylim(top=0.1)
plt.savefig('NaI_Cs_Full_Spe')    

#choose what type of gaussian
if Top == 0.06:
    plot_double(Roi_Start, Roi_End, Top)
    
if Top == 17.5:
    plot_two(Roi_Start, Roi_End, Roi_Start_2, Roi_End_2, Top)
    
if Top % 2 == 0:
    plot_single(Roi_Start, Roi_End, Top)   
plt.savefig('NaI_Cs_Peak')     

#x = values of E, y = channel number
x = [59.5409,661.657,1173.228,1332.492,30.85,356.02]
y = [43.6654,428.01362,739.7084,835.56301,22.99253,233.83294]

# define the true objective function
def line(x, a, b):
	return a * x + b

fit, _ = curve_fit(line, x, y)
# summarize the parameter values
a, b = fit
print('y = %.5f * x + %.5f' % (a, b))
# plot input vs output
fig, ax = plt.subplots(1)
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x), max(x), 1)
# calculate the output for the range
y_line = line(x_line, a, b)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '-', color='red')
ax.set_xlabel('Energy keV')
ax.set_ylabel('Channel')
plt.title('NaI Absolute Efficiency vs Energy')
plt.savefig('NaI_Calibration_Curve')    
plt.show() 

sigma_values = [3.3013,13.54451,20.7093,15.9591,2.136740,11.206383627]

FWHM = [item * 2.355 for item in sigma_values]

Resolution = [(i / j) for i, j in zip(FWHM,x)]

R_Squared = [i*i for i in (Resolution)]


def resolution_eqn(E, a, b, c):
    R_2 = a*(E**-2) + b*(E**-1) + c
    return (np.sqrt(R_2))

fit, _ = curve_fit(resolution_eqn, x, Resolution)
a,b,c = fit
print('R2 = %.5f * E-2 + %.5f * E-1 + %.5f' % (a, b, c))
fig, ax = plt.subplots(1)
plt.scatter(x, Resolution)
#plt.xscale("log")
#plt.yscale("log")
x_line = np.arange(min(x), max(x), 1)
# calculate the output for the range
y_line = resolution_eqn(x_line, a, b, c)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '-', color='red')
ax.set_xlabel('Energy keV')
ax.set_ylabel('Resolution')
plt.title('NaI Resolution')
plt.savefig('NaI_Resolution')  
plt.show()
  
#half life for am,cs,co and ba in seconds
lam_ary = []
half_life = [13631588000, 949038600 ,166259956, 332336980]

for x in range(4):
    lam = 0.693/half_life[x]
    lam_ary.append(lam)

#time elapsed
t_NaI = [1377488609, 1377832959, 1377228322, 1377231157]  #am ba- cs co

#activity of lab 2 sources
A_am2 = 37000*11.92*np.exp(-lam_ary[0]*t_NaI[0])*120
A_cs2 = 37000*12.41*np.exp(-lam_ary[1]*t_NaI[2])*300
A_co2 = 37000*11.32*np.exp(-lam_ary[2]*t_NaI[3])*1804
A_ba2 = 37000*10.82*np.exp(-lam_ary[3]*t_NaI[1])*647

#values for area under curve of each peak
all_a = [582.3593376, 87.61288885, 35.61158216, 0.826123515, 0.55937093, 153.2732646]
all_Ac = [A_am2, A_ba2, A_ba2, A_co2, A_co2, A_cs2]
FEPE = []
for x in range(6):
    result = all_a[x]/all_Ac[x]
    FEPE.append(result)

x = [59.5409, 30.8535602, 356.02, 1173.228, 1332.492, 661.657]

def efficency_eqn(E, a, b, c):
    Eff = a + b*np.log(E) + c*(np.log(E))**2
    return (Eff)
fit, _ = curve_fit(efficency_eqn, x, FEPE)
a,b,c = fit
print('Ea = %.5f + %.5f * log(E) + %.5f * (log(E))^2' % (a, b, c))
fig, ax = plt.subplots(1)
plt.scatter(x, FEPE)
#plt.xscale("log")
#plt.yscale("log")
x_line = np.arange(min(x), max(x), 1)
# calculate the output for the range
y_line = efficency_eqn(x_line, a, b, c)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '-', color='red')
ax.set_xlabel('Energy keV')
ax.set_ylabel('Absolute Efficiency')
plt.xscale("log")
plt.yscale("log")
plt.title('NaI Absolute Efficiency vs Energy')
plt.savefig('NaI_Absolute Efficiency') 
plt.show()

radius = 0.0254
length = 0.0508
distance = 0.23

area_of_detector = ((np.pi*radius**2)*abs(np.cos(0)) + 2*radius*length*abs(np.sin(0)))
geo_factor = area_of_detector/(4*np.pi*distance**2) 

intrinsic_Eff = FEPE/geo_factor

fit, _ = curve_fit(efficency_eqn, x, intrinsic_Eff)
a,b,c = fit
print('Ei = %.5f + %.5f * log(E) + %.5f * (log(E))^2' % (a, b, c))
fig, ax = plt.subplots(1)
plt.scatter(x, intrinsic_Eff) 
#plt.xscale("log")
#plt.yscale("log")
x_line = np.arange(min(x), max(x), 1)
# calculate the output for the range
y_line = efficency_eqn(x_line, a, b, c)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '-', color='red')
ax.set_xlabel('Energy keV')
ax.set_ylabel('Intrinsic Efficiency')
plt.xscale("log")
plt.yscale("log")
plt.title('NaI Intrinsic Efficiency vs Energy')
plt.savefig('NaI_Intrinsic Efficiency')
plt.show()


# In[ ]:




