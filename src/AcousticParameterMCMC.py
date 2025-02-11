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

    def __init__(self, cosineCount, sourceLocation, receiverLocations, truescatter, userSampleDensity=700, sourceFrequency=14_000, beta=3, wl_scale=[0.1, 1.0]):
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

        self.cosineCount = cosineCount
        self.sourceLocation = sourceLocation
        self.receiverLocations = receiverLocations
        self.sourceFrequency = sourceFrequency
        self.userSampleDensity = userSampleDensity
        self.trueScatter = truescatter
        self.beta = beta
        self.wl_scale = wl_scale

    def run(self, surfaceFunction=None, kernel="NUTS", burnInCount=2000, sampleCount=5000, chainCount=4, targetAccRate=0.238, scaleTrueScatter=True):
        '''
        Runs the pymc model with the selected kernel. Stores the parameters and trace afterwards.
        Also outputs the parameters as a .csv with the file name "kernel + .csv"

        Inputs:
        surfaceFunction: A function that takes x and 3 parameters.
        kernel: The type of kernel to use. Currently supports "metropolis-hastings" and "NUTS"
        burnInCount: The number of burn in iterations. Will be discarded.
        sampleCount: The number of samples to run the chain for.
        chainCount: The number of parallel chains to run. Useful for getting diagnostics.
        tragetAccRate: The target acceptance rate.
        scaleTrueScatter: Sets whether to scale the true scatter by the flat plate response.
        '''

        # Raises memory usage but should be more efficient
        pytensor.config.allow_gc=False

        factor = 1.0
        if scaleTrueScatter:
            factor = AcousticParameterMCMC.GenerateFactor(self.sourceLocation, self.receiverLocations, self.sourceFrequency, self.userSampleDensity)

        factorizedScatter = self.trueScatter/factor
        factorizedScatter = factorizedScatter

        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        
        N = self.cosineCount

        # Generate the wavelength distributions
        # Special case for N = 1 is just the first element
        wl_means = pt.as_tensor_variable([self.wl_scale[0]])
        wl_sds = wl_means**1.1

        if N != 1:
            wl_means = pt.as_tensor_variable(np.logspace(np.log(self.wl_scale[0]), np.log(self.wl_scale[1]), N))
            wl_sds = wl_means**1.1

        with pm.Model() as model:

            # Priors for each set of 3 params       
            # Amplitudes depend on wavelengths                        
            wavelengths = pm.TruncatedNormal('wl', mu=wl_means, sigma=wl_sds, lower=1e-8, upper=0.4, shape=(N,))   
            phases = pm.TruncatedNormal('phase', mu=0.0, sigma=1.0, lower=-1.0, upper=1.0, shape=(N,))

            amp_means, amp_sds = self._amplitudePowerLaw(wavelengths)
            amplitudes = pm.TruncatedNormal('amp', mu=amp_means, sigma=amp_sds, lower=1e-8, upper=0.025, shape=(N,))

            epsilon= 0.05  #Scales the error covariance matrix for the error between receivers

            # Surface function with evaluated parameters (if you need it before sampling)rt
            def newFunction(x):
                return surfaceFunction(pt.as_tensor_variable(x), amplitudes, wavelengths, phases)

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
            KA_Object.surfaceChecker() #Adds pm.Potential penalty if kirchoff criteria not met

            # Likelihood: Compare predicted response to observed data
            likelihood = pm.MvNormal('obs', mu=scatter, cov=np.eye(len(factorizedScatter))*epsilon, observed=factorizedScatter)

        # View model graph
        #graph = pm.model_to_graphviz(model)
        #graph.render("model_graph", format="png", view=True)  # Saves and opens the file

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
                step = pm.NUTS(target_accept=targetAccRate)

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

    def _amplitudePowerLaw(self, wavelengths):
        '''
        Calculates the amplitude distributions assuming follows a wavelength power law
        '''
        wavenumbers = pt.abs((2.0*pt.pi)/wavelengths)
        amps_means = wavenumbers**(-self.beta/2.0)
        amps_sds = amps_means**1.1
        return amps_means, amps_sds

    def _generateHeader(self):
        header_string = ""
        for i in range(0, self.cosineCount):
            itxt = str(i)
            header_string += "amp" + itxt + "," + "wl" + itxt + "," + "phase" + itxt + ","
        return header_string

    def plotTrace(self):
        '''
        Plots the trace of the model with arviz
        '''

        print(az.summary(self.trace))
        az.plot_trace(self.trace)