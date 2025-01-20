from scipy.integrate import quad
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib as mb
import corner as corner
import csv
import time

class Directed2DVectorised:
    def __init__(self,sourceLocation,receiverLocations,surfaceFunction,frequency,a,sourceAngle = -np.pi/4,method = 'trapz',userMinMax = None,userSamples = 9000,absolute = True):

        '''
        Init function for the Kirchhoff approximation using a radiation from a baffled piston source.
        Inputs:

        sourceLocation: List of x and y component of the source location
        receiverLocations: List of lists of x and y components of receiver locations
        surfaceFunction: Either a function that only takes x as an argument, or an array of values that relate to surface elevation.
        frequency: Source frequency
        a: Piston aperture
        sourceAngle: Angle of the source
        method: Integration methods, can be trapz, simps or cumtrapz. simp is usually recommended
        userMinMax: [a,b] start and end point of the surface
        userSamples: integer showing density of samples
        absolute: Bool, absolute value
        '''

        self.sourceLocation = np.array(sourceLocation)
        self.receiverLocationsX = np.array(receiverLocations)[:,0]
        self.receiverLocationsY = np.array(receiverLocations)[:,1]
        self.k = (2*np.pi*frequency)/343
        self.a = a
        self.sourceAngle = sourceAngle
        self.surfaceFunction = surfaceFunction
        self.method = method
        self.absolute = absolute
        if userMinMax == None:
            self.min = self.receiverLocationsX.min()
            self.max = self.receiverLocationsX.max()
        else:
            self.min = userMinMax[0]
            self.max = userMinMax[1]
        self.number = self.receiverLocationsX.shape[0]
        self.samples = userSamples
        self.receiverLocationsX = self.receiverLocationsX.reshape(-1,1) + np.zeros((self.number,self.samples))
        self.receiverLocationsY = self.receiverLocationsY.reshape(-1,1) + np.zeros((self.number,self.samples))
        self.sourceLocationX = self.sourceLocation[0] + np.zeros((self.number,self.samples))
        self.sourceLocationY = self.sourceLocation[1] + np.zeros((self.number,self.samples))
        #BUG[IGGY] - FIX this, something is breaking and I'm not really sure why
        if type(self.surfaceFunction) == np.ndarray:
            self.x = np.linspace(self.min,self.max,self.samples).reshape(1,-1) + np.zeros((self.number,self.samples))
            self.surfaceVals = self.surfaceFunction

            np.gradient(self.surfaceVals, self.x[0][1] - self.x[0][0], edge_order = 2, axis = None)
            self.derivativeVals = np.gradient(self.surfaceVals, self.x[0][1] - self.x[0][0], edge_order = 2, axis = None)
            self.doubleDerivativeVals = np.gradient(self.derivativeVals, self.x[0][1] - self.x[0][0], edge_order = 2, axis = None)

            self.surfaceVals = self.surfaceVals.reshape(1,-1) + np.zeros((self.number,self.samples))
            self.derivativeVals = self.derivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))
            self.doubleDerivativeVals = self.doubleDerivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))


        else:

            self.x = np.linspace(self.min,self.max,self.samples).reshape(1,-1) + np.zeros((self.number,self.samples)) #see if this can be changed
            self.surfaceVals = surfaceFunction(self.x)
            self.derivativeVals = np.gradient(self.surfaceVals[0], self.x[0][1] - self.x[0][0],edge_order=2, axis=None)
            self.doubleDerivativeVals = np.gradient(self.derivativeVals, self.x[0][1] - self.x[0][0],edge_order=2, axis=None)

            self.derivativeVals = self.derivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))
            self.doubleDerivativeVals = self.doubleDerivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))

        if isinstance(self.surfaceVals,float):
            self.surfaceVals = self.surfaceVals + np.zeros((self.number,self.samples))
            self.derivativeVals = self.derivativeVals + np.zeros((self.number,self.samples))


    def surfaceChecker(self, relaxed = True, hyper_accurate = False):

        '''
        Check if the surface satisfies the Kirchhoff criteria.
        Inputs:
        relaxed - Increases from source angle to 1, although it seems it's just 1
        hyper_accuare - More accurate calculation for highly oscillatory functions, when there is no array defining surface
        elevation.
        '''

        if not hyper_accurate:
            #self.doubleDerivativeVals = sp.misc.derivative(self.surfaceFunction,self.x,n=2)
            numerator = 1 + (self.derivativeVals)**2
            denominator = self.doubleDerivativeVals

        else:
            #For highly oscillatory functions
            x = np.linspace(self.min, self.max, 10*self.samples)
            fun_val = self.surfaceFunction(x)
            derivativeVals = np.gradient(fun_val, x[1] - x[0],edge_order=2, axis=None)
            doublederivativeVals = np.gradient(derivativeVals, x[1] - x[0],edge_order=2, axis=None)
            numerator = 1 + derivativeVals**2
            denominator = doublederivativeVals

        self.curvature = (numerator**1.5)/np.abs((denominator))
        self.condition = 1/((self.k*self.curvature)**0.333333333)

        if self.condition.max() > 1:
            print("Condition failed")
            print(self.condition.max())

            self.checker = False
        else:
            self.checker = True

        return self.checker



    def __CalculateAngle(self):
        ac = []
        ab = []
        for i in range(len(self.sourceLocationX[0])):
            ac.append([1,0])
            tempFrac = 1/np.sqrt(((self.x[0][i]-self.sourceLocationX[0][i])**2+
                           (self.surfaceVals[0][i]-self.sourceLocationY[0][i])**2))
            ab.append([tempFrac*(self.x[0][i]-self.sourceLocationX[0][i]),
                       tempFrac*(self.surfaceVals[0][i] -
                                 self.sourceLocationY[0][i])])
        s = np.array(ac)
        r = np.array(ab)
        SdotR = np.einsum('ij,ij->i',s,r)
        modSmodR = np.linalg.norm(r,axis=1)*np.linalg.norm(s,axis=1)
        temp = np.abs(np.arccos(SdotR/modSmodR) - (-self.sourceAngle + np.pi/2))
        output = mb.repmat(temp,self.number,1)
        return output



    def __Integrand(self):

        r =  - self.x + self.sourceLocationX
        r2 =  - self.x + self.receiverLocationsX
        l = self.sourceLocationY - self.surfaceVals
        l2 = self.receiverLocationsY - self.surfaceVals
        R1 = np.sqrt( r*r + l*l )
        R2 = np.sqrt( r2*r2 + l2*l2 )
        qz = -self.k * ( l2*1/R2 + l*1/R1 )
        qx = -self.k *( (-self.derivativeVals*(-self.surfaceVals + self.sourceLocationY)+self.x-self.sourceLocationX)*1/R1
                     + (-self.derivativeVals*(-self.surfaceVals + self.receiverLocationsY)+self.x-self.receiverLocationsX)*1/R2)

        theta = np.arccos(l*1/R1) - (- self.sourceAngle + np.pi/2)
        #theta2 = self.__CalculateAngle()
        #Seems like theta2 is incorrect. This means I need to go back
        #and change the others in the non-vectorised version
        Directivity = (sp.special.jn(1,self.k*self.a*np.sin(theta)))/(self.k*self.a*np.sin(theta))
        Directivity[np.isnan(Directivity)] = 0.5
        #Directivity2 = (sp.special.jn(1,self.k*self.a*np.sin(theta2)))/(self.k*self.a*np.sin(theta2))
        #Directivity2[np.isnan(Directivity2)] = 0.5

        #G = (2*(qz)*Directivity*np.exp(1j*(R1+R2)*self.k)*1)/np.sqrt(R1*R2)
        que = qz - self.derivativeVals*qx
        F = 2*que*Directivity*np.exp(0+1j*(R1+R2)*self.k)*1/np.sqrt(R1*R2)

        return F

    def Scatter(self,absolute=False,norm=True,direct_field=False):
        F  = self.__Integrand()
        p = np.zeros(F.shape[0],dtype=np.complex128)
        if direct_field:
            r =  - self.receiverLocationsX.T[0] + self.sourceLocationX.T[0]
            l = - self.receiverLocationsY.T[0] + self.sourceLocationY.T[0]
            R1 = np.sqrt( r*r + l*l )
            theta = np.arccos(l*1/R1) - (- self.sourceAngle + np.pi/2)
            Directivity = (sp.special.jn(1,self.k*self.a*np.sin(theta)))/(self.k*self.a*np.sin(theta))
            Directivity[np.isnan(Directivity)] = 0.5
            #This may be needed for boundary conditions
            #Directivity[ 3/2*np.pi > theta > np.pi/2] = 0
            print(theta)
            field = 2*Directivity*np.exp(0+1j*(R1)*self.k)*1/np.sqrt(R1)
            p += field
        if self.method == 'trapz':
            p += -1j/(2*np.pi*self.k)*np.trapz(F,self.x,axis=1)

            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p
        elif self.method == 'simp':
            p += -1j/(2*np.pi*self.k)*sp.integrate.simpson(F,x=self.x,axis=1)

            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p


            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p


        elif self.method == 'cumtrapz':
            p += -1j/(2*np.pi*self.k)*sp.integrate.cumtrapz(F,self.x,axis=1)
            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p


def fourierRes(signal, timeIncrement,axis = 0):
    '''Just give us the fourier transform of a signal'''
    import scipy as sp
    sampleFrequency = 1/timeIncrement
    n = signal.shape[0]
    yf = np.fft.fft(signal,axis=axis)
    #xf = sp.fft.fftshift(sp.fft.fftfreq(n,timeIncrement))
    xf = np.fft.fftfreq(n,timeIncrement)
    return np.fft.fftshift(xf),np.fft.fftshift(yf)


def spatialRes(signal):
    '''Merely a FFT.'''
    y = sp.fft.fft(signal,axis=0)
    return y


def Doppler(signals,dt,axis = 0,real=True):
    '''Takes signals of shape (number of signals, number of samples, number of receivers ).
    Returns the Doppler spectrum of all the receivers.'''
    n = np.array(signals[0]).size
    fSig = []

    for s in signals:
        x,Fsignals = fourierRes(np.array(s),dt,axis=axis)
        fSig.append(Fsignals)
    freq = x
    if real:
        return freq,1/n*np.mean(np.abs(np.array(fSig))**2,axis=0)
    else:
        return freq,1/n*np.mean(np.real(np.array(fSig)),axis=0),1/n*np.mean(np.imag(np.array(fSig)),axis=0)

import numpy.matlib
def fourier_coefs(signal, x):
    '''Return the Fourier transform from a signal.'''
    dx = x[1] - x[0] #Dx
    N = len(x)
    freqs = np.fft.fftfreq(N, dx)[:N//2]
    fft = (np.fft.fft(signal)/N)[:N//2]
    return freqs, fft

def decompose(freqs, fft, ranges, x):
    '''Decompose a Fourier transform and the frequencies into coefficients. Returns
       the full decomposed surface, the coefficients, and the frequency.'''
    c = fft[0]
    a_s = -2*np.real(fft[1:ranges])
    b_s = -2*np.imag(fft[1:ranges])
    f_comp = 2*np.pi*freqs[1:ranges]
    temp_a = np.reshape(a_s,(-1,1)) + np.zeros((ranges-1,len(x)))
    temp_b = np.reshape(b_s,(-1,1)) + np.zeros((ranges-1,len(x)))
    temp_f = np.reshape(f_comp,(-1,1)) + np.zeros((ranges-1,len(x)))
    summies = (temp_a * np.cos((temp_f * x)) + temp_b * np.sin((temp_f * x)))
    summation = np.sum(summies, axis = 0)
    summation = summation + c
    return np.real(summation), a_s, b_s,f_comp

def extract_components(a_coef, b_coef, freqs):
    '''Extract the fouerier components that aren't extremely small. '''
    surface_parameters = []
    for i in range(len(a_coef)):
        if a_coef[i] > 0.000001 or b_coef[i] > 0.000001:
            surface_parameters.append([a_coef[i], b_coef[i], freqs[i]])
    return surface_parameters