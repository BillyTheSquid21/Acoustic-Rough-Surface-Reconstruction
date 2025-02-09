import pymc as pm
import numpy as np
import arviz as az

import pytensor
import pytensor.tensor as pt

from src.Directed2DVectorized import Directed2DVectorised
from src.Directed2DVectorized import Directed2DVectorisedSymbolic

class AcousticParameterMCMC:
    '''
    A class that runs an MCMC sampling model to recover the parameters of a cosine series acoustic surface
    '''

    def GenerateFactor(sourceLocation, receiverLocations, sourceFrequency=14_000, userSamples=700):
        '''
        Generates a factor to scale response by flat plate response

        Inputs:
        sourceLocation: The location in world space of the acoustic source.
        recieverLocations: The locations in world space of the recievers. Must be an array of (x,y) coordinates.
        sourceFrequency: The frequency of the source acoustic wave
        userSampleDensity: The density of samples in the simulated scatter. Relates to resolution rather than sampling rate. 
        '''

        def zero(x):
            return 0*x

        flat = Directed2DVectorised(sourceLocation,receiverLocations,zero,sourceFrequency,0.02,np.pi/3,'simp',userMinMax=[-5,5],userSamples=userSamples,absolute=False)
        flatscatter = flat.Scatter(absolute=True,norm=False)
        return flatscatter.max().copy()

    def __init__(self, cosineCount, sourceLocation, receiverLocations, truescatter, userSampleDensity=700, sourceFrequency=14_000):
        '''
        Init function for the AcousticParameterMCMC class

        Inputs:
        cosineCount: Number of surface parameters to recover (Amplitude, Wavelength, Phase tuple). Must be a multiple of 3.
        sourceLocation: The location in world space of the acoustic source.
        recieverLocations: The locations in world space of the recievers. Must be an array of (x,y) coordinates.
        truescatter: The true scatter data that the chain compares response to. Must be amplitude only data for each reciever location.
        userSampleDensity: The density of samples in the simulated scatter. Relates to resolution rather than sampling rate.
        sourceFrequency: The frequency of the source acoustic wave
        '''
        
        assert cosineCount % 3 == 0

        self.cosineCount = cosineCount
        self.sourceLocation = sourceLocation
        self.receiverLocations = receiverLocations
        self.sourceFrequency = sourceFrequency
        self.userSampleDensity = userSampleDensity
        self.trueScatter = truescatter

    def run(self, surfaceFunction=None, kernel="NUTS", burnInCount=2000, sampleCount=5000, chainCount=4, scaleTrueScatter=True):
        '''
        Runs the pymc model with the selected kernel. Stores the parameters and trace afterwards.
        Also outputs the parameters as a .csv with the file name "kernel + .csv"

        Inputs:
        surfaceFunction: A function that takes x and 3 parameters. TODO: extend to 6+
        kernel: The type of kernel to use. Currently supports "metropolis-hastings" and "NUTS"
        burnInCount: The number of burn in iterations. Will be discarded.
        sampleCount: The number of samples to run the chain for.
        chainCount: The number of parallel chains to run. Useful for getting diagnostics.
        scaleTrueScatter: Sets whether to scale the true scatter by the flat plate response.
        '''

        # Raises memory usage but should be more efficient
        pytensor.config.allow_gc=False

        factor = 1.0
        if scaleTrueScatter:
            factor = AcousticParameterMCMC.GenerateFactor(self.sourceLocation, self.receiverLocations, self.sourceFrequency, self.userSampleDensity)

        factorizedScatter = self.trueScatter/factor
        factorizedScatter = factorizedScatter

        if not self.cosineCount % 3 == 0:
            raise Exception("Surface parameter count must be multiples of 3!")
        
        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        
        N = int(self.cosineCount / 3)

        with pm.Model() as model:
            # Model works by:
            # 1. Sample each parameter from their respective prior
            # 2. Apply a penalty if out of bounds
            # 3. Simulates the scattering of the acoustic waves against a surface from the parameters
            # 4. Scales the scattered signal by the factor from the flat plate response
            # 5. Apply a penalty if fails the surface check
            # 6. Calculates the likelihood with the multivariate normal betweem the simulated response and the observed response

            # Priors for each set of 3 params
            param1 = pm.TruncatedNormal('amp', mu=0.001, sigma=0.0015, lower=1e-6, upper=0.015, shape=(N,)) #Amplitude sampling                                                 
            param2 = pm.TruncatedNormal('wl', sigma=0.08, lower=1e-6, upper=0.4, shape=(N,))                #Wavelength sampling
            param3 = pm.TruncatedNormal('phase', mu=0.0, sigma=0.01, lower=-1.0, upper=1.0, shape=(N,))     #Phase sampling

            epsilon= 0.05  #Scales the error covariance matrix for the error between receivers

            # Surface function with evaluated parameters (if you need it before sampling)
            def newFunction(x):
                return surfaceFunction(x, param1, param2, param3)

            # Scatter operation which maintains symbolic links
            # Gives the same results as Directed2DVectorized class
            KA_Object = Directed2DVectorisedSymbolic(
                self.sourceLocation,
                self.receiverLocations,
                newFunction,
                self.sourceFrequency,
                0.02,
                np.pi / 3,
                userMinMax=[-1,1],
                userSamples=self.userSampleDensity,
                absolute=False
            )
            scatter = KA_Object.Scatter(absolute=True, norm=False)
            scatter = scatter / factor
            KA_Object.surfaceChecker(True) #Adds pm.Potential penalty if kirchoff criteria not met

            # Likelihood: Compare predicted response to observed data
            likelihood = pm.MvNormal('obs', mu=scatter, cov=np.eye(len(factorizedScatter))*epsilon, observed=factorizedScatter)

        trace = []
        posterior_samples = []
        if kernel == "metropolis-hastings":
            with model:
                # Define the Metropolis sampler
                step = pm.Metropolis(tune=True)

            with model:
                # Sample from the posterior
                trace = pm.sample(tune=burnInCount, draws=sampleCount, step=step, chains=chainCount, return_inferencedata=True)
        elif kernel == "NUTS":
            with model:
                # Define the NUTS sampler
                step = pm.NUTS(target_accept=0.238)

            with model:
                # Sample from the posterior
                trace = pm.sample(tune=burnInCount, draws=sampleCount, step=step, chains=chainCount, return_inferencedata=True, nuts_sampler="numpyro")
        else:
            raise Exception("Unrecognised kernel name!")
        
        # pymc statistics
        #model.profile(model.logp()).summary()
        #model.profile(pm.gradient(model.logp(), vars=None)).summary()

        # Flatten arrays with some numpy shaping trickery
        # Gives us columns so that each set of 3 params are next to each other
        total_samples = chainCount*sampleCount

        posterior = trace.posterior
        amps = posterior['amp'].values.transpose(2, 0, 1).reshape(N, total_samples)
        wls = posterior['wl'].values.transpose(2, 0, 1).reshape(N, total_samples)
        phases = posterior['phase'].values.transpose(2, 0, 1).reshape(N, total_samples)

        posterior_samples = np.row_stack((amps,wls,phases)).T
        posterior_samples = posterior_samples.reshape(total_samples, 3, N).transpose(0, 2, 1)
        posterior_samples = posterior_samples.reshape(total_samples, N*3)

        # Save to csv with header
        np.savetxt("results/" + kernel + ".csv", posterior_samples, delimiter=",", header=self._generateHeader(), fmt="%s")

        print("MCMC Results saved!")

        # Save parameter store
        self.posteriorSamples = posterior_samples
        self.trace = trace

    def _generateHeader(self):
        header_string = ""
        for i in range(0, int(self.cosineCount / 3)):
            itxt = str(i)
            header_string += "amp" + itxt + "," + "wl" + itxt + "," + "phase" + itxt + ","
        return header_string

    def plotTrace(self):
        '''
        Plots the trace of the model with arviz
        '''

        print(az.summary(self.trace))
        az.plot_trace(self.trace)