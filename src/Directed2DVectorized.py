from scipy.integrate import quad
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d.axes3d import get_test_data
from numpy import matlib as mb
import corner as corner
import pytensor.tensor as pt
import pymc as pm

from src.SymbolicMath import SymGradient, SymIntegral
from src.SignalAnalysis import *

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

class Directed2DVectorisedSymbolic:
    def __init__(self,sourceLocation,receiverLocations,surfaceFunction,frequency,a,sourceAngle=-np.pi/4,userMinMax=None,userSamples=9000,absolute=True):

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

        self.x = np.linspace(self.min,self.max,self.samples).reshape(1,-1) + np.zeros((self.number,self.samples)) #see if this can be changed
        self.surfaceVals = surfaceFunction(self.x)
        self.derivativeVals = SymGradient(self.surfaceVals[0], self.x[0])
        self.doubleDerivativeVals = SymGradient(self.derivativeVals, self.x[0])

        self.derivativeVals = self.derivativeVals.reshape((1, -1)) + pt.zeros((self.number, self.samples))
        self.doubleDerivativeVals = self.doubleDerivativeVals.reshape((1, -1)) + pt.zeros((self.number, self.samples))

        if isinstance(self.surfaceVals,float):
            self.surfaceVals = self.surfaceVals + pt.zeros((self.number,self.samples))
            self.derivativeVals = self.derivativeVals + pt.zeros((self.number,self.samples))


    def surfaceChecker(self, relaxed=True, hyper_accurate=False):

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
            #TODO: When needed convert to symbolic
            #x = np.linspace(self.min, self.max, 10*self.samples)
            #fun_val = self.surfaceFunction(x)
            #derivativeVals = np.gradient(fun_val, x[1] - x[0],edge_order=2, axis=None)
            #doublederivativeVals = np.gradient(derivativeVals, x[1] - x[0],edge_order=2, axis=None)
            #numerator = 1 + derivativeVals**2
            #denominator = doublederivativeVals
            pass

        self.curvature = (numerator**1.5)/np.abs((denominator))
        self.condition = 1/((self.k*self.curvature)**0.333333333)

        surf_constraint = self.condition.max() > 1
        potential = pm.Potential("surface_c", pm.math.log(pm.math.switch(surf_constraint, 1, 1e-10)))

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

        Directivity = (pt.math.j1(self.k*self.a*np.sin(theta)))/(self.k*self.a*np.sin(theta))
        Directivity_with_nan_handling = pt.switch(pt.isnan(Directivity), 0.5, Directivity)

        que = qz - self.derivativeVals*qx
        cos_term = pt.cos((R1 + R2) * self.k)  # Real part of the exponential
        sin_term = pt.sin((R1 + R2) * self.k)  # Imaginary part of the exponential

        F_Real = 2 * que * Directivity_with_nan_handling * cos_term / pt.sqrt(R1 * R2)
        F_Imag = 2 * que * Directivity_with_nan_handling * sin_term / pt.sqrt(R1 * R2)
        return (F_Real, F_Imag)

    def Scatter(self,absolute=False,norm=True,direct_field=False):
        F  = self.__Integrand()
        p = pt.zeros(F[0].shape[0])

        #Just use simpson integral for now
        integral_real = SymIntegral(F[0], x=self.x, axis=1)
        integral_imag = SymIntegral(F[1], x=self.x, axis=1)
        p_real = p + -1 / (2 * np.pi * self.k) * integral_real
        p_imag = p + -1 / (2 * np.pi * self.k) * integral_imag
        if norm == True:
            p = pt.sqrt(p_real**2 + p_imag**2)/np.max(pt.sqrt(p_real**2 + p_imag**2))
            return p

        elif absolute == True:
            p = pt.sqrt(p_real**2 + p_imag**2)
            return p
        else:
            return p