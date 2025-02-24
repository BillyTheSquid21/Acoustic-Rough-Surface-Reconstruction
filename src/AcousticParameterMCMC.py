import pymc as pm
import numpy as np
import arviz as az
import shutil

import pytensor
import pytensor.tensor as pt

from src.Directed2DVectorized import Directed2DVectorised
from src.Directed2DVectorized import Directed2DVectorisedSymbolic

class AcousticParameterMCMC:
    '''
    A class that runs an MCMC sampling model to recover the parameters of a cosine series acoustic surface
    '''

    _amplitudeProposal = None
    _phaseProposal = None
    _wavelengthProposal = None
    _wavelengths = None

    def GenerateFactor(sourceLocation, sourceAngle, receiverLocations, pistonAperture, sourceFrequency=14_000, userSamples=700):
        '''
        Generates a factor to scale response by flat plate response

        Inputs:
        sourceLocation: The location in world space of the acoustic source.
        sourceAngle: Angle the acoustic source is at
        recieverLocations: The locations in world space of the recievers. Must be an array of (x,y) coordinates.
        pistonAperture: The aperture size of the baffled piston approximation
        sourceFrequency: The frequency of the source acoustic wave
        userSampleDensity: The density of samples in the simulated scatter. Relates to resolution rather than sampling rate. 

        Returns:
        float: The factor for normalization
        '''

        def zero(x):
            return 0*x

        flat = Directed2DVectorised(sourceLocation,receiverLocations,zero,sourceFrequency,pistonAperture,sourceAngle,'simp',userMinMax=[-5,5],userSamples=userSamples,absolute=False)
        flatscatter = flat.Scatter(absolute=True,norm=False)
        return flatscatter.max().copy()
    
    def LoadCSVData(path):
        '''
        Loads data from CSV and formats it to work with the symbolic math cosine sum functions.
        If data is compressed in tar.gz format will silently unzip

        Inputs:
        path: The path to the data

        Returns:
        numpy.array: A formatted array containing each set of parameters in (amp,wl,phase) tuples
        '''
        if path.endswith("tar.gz"):
            new_path = path.replace("tar.gz", "csv")
            shutil.unpack_archive(path, new_path)
            path = new_path

        posterior_samples = np.loadtxt(path, delimiter=",")
    
        # Convert each row into 3-element tuples for better multi
        return [list(zip(*row.reshape(-1, 3).T)) for row in posterior_samples]
    
    def generateFactor(self):
        '''
        Generates a factor to scale response by flat plate response

        Returns:
        float: The factor for normalization
        '''
        return AcousticParameterMCMC.GenerateFactor(self.sourceLocation, self.sourceAngle, self.receiverLocations, self.pistonAperture, self.sourceFrequency, self.userSampleDensity)

    def __init__(self, cosineCount, sourceLocation, sourceAngle, receiverLocations, truescatter, userSampleDensity=700, sourceFrequency=14_000, pistonAperture=0.02):
        '''
        Init function for the AcousticParameterMCMC class

        Inputs:
        cosineCount: Number of surface parameters to recover (Amplitude, Wavelength, Phase tuple). Must be a multiple of 3.
        sourceLocation: The location in world space of the acoustic source.
        sourceAngle: Angle the acoustic source is at
        recieverLocations: The locations in world space of the recievers. Must be an array of (x,y) coordinates.
        truescatter: The true scatter data that the chain compares response to. Must be amplitude only data for each reciever location.
        userSampleDensity: The density of samples in the simulated scatter. Relates to resolution rather than sampling rate.
        sourceFrequency: The frequency of the source acoustic wave
        pistonAperture: The aperture size of the baffled piston approximation
        '''

        self.cosineCount = cosineCount
        self.sourceLocation = sourceLocation
        self.receiverLocations = receiverLocations
        self.sourceFrequency = sourceFrequency
        self.userSampleDensity = userSampleDensity
        self.trueScatter = truescatter
        self.pistonAperture = pistonAperture
        self.sourceAngle = sourceAngle

    def run(self, surfaceFunction=None, kernel="NUTS", burnInCount=2000, sampleCount=5000, chainCount=4, targetAccRate=0.238, scaleTrueScatter=True, showGraph=False):
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
        showGraph: Shows and saves an image of the model graph
        '''

        # Raises memory usage but should be more efficient
        pytensor.config.allow_gc=False

        # Generate the factor to scale by the flat plate response
        factor = 1.0
        if scaleTrueScatter:
            factor = self.generateFactor()

        if self._wavelengths is not None:
            self.runWithWavelengths(surfaceFunction, kernel, burnInCount, sampleCount, chainCount, targetAccRate, factor, showGraph)
        else:
            self.runWithFull(surfaceFunction, kernel, burnInCount, sampleCount, chainCount, targetAccRate, factor, showGraph)  

    def runWithFull(self, surfaceFunction=None, kernel="NUTS", burnInCount=2000, sampleCount=5000, chainCount=4, targetAccRate=0.238, factor=1.0, showGraph=False):
        # Normalize the scatter to the flat plate response
        factorizedScatter = self.trueScatter/factor
        factorizedScatter = factorizedScatter

        # Set the number of waves to reconstruct
        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        N = self.cosineCount

        # Set the prior distributions
        ampSigmas = self.getAmplitudeProposal()
        phaseSigmas = self.getPhaseProposal()
        wlSigmas = self.getWavelengthProposal()
        with pm.Model() as model:

            # Proposal for each set of 3 params                  
            wavelengths = pm.TruncatedNormal('wl', sigma=wlSigmas, lower=0.0, upper=1.0, shape=(N,))
            amplitudes = pm.TruncatedNormal('amp', sigma=ampSigmas, lower=0.0, upper=0.025, shape=(N,), initval=1e-6 + pt.zeros(shape=(N,)))
            phases = pm.TruncatedNormal('phase', mu=0.0, sigma=0.25, lower=-0.5, upper=0.5, shape=(N,), initval=pt.zeros(shape=(N,)))

            error = 0.0075  #Scales the error covariance matrix for the error and noise between receivers

            # Surface function with evaluated parameters
            def newFunction(x):
                return surfaceFunction(pt.as_tensor_variable(x), amplitudes, wavelengths, phases)

            # Scatter operation which maintains symbolic links
            # Gives the same results as Directed2DVectorized class
            KA_Object = Directed2DVectorisedSymbolic(
                self.sourceLocation,
                self.receiverLocations,
                newFunction,
                self.sourceFrequency,
                self.pistonAperture,
                self.sourceAngle,
                userMinMax=[-1,1],
                userSamples=self.userSampleDensity,
                absolute=False
            )
            scatter = KA_Object.Scatter(absolute=True, norm=False)
            scatter = scatter / factor
            KA_Object.surfaceChecker() #Adds pm.Potential penalty if kirchoff criteria not met

            # Likelihood: Compare predicted response to observed data
            likelihood = pm.MvNormal('obs', mu=scatter, cov=np.eye(len(factorizedScatter))*error, observed=factorizedScatter)

        # View model graph
        if showGraph:
            graph = pm.model_to_graphviz(model)
            graph.render("model_graph", format="png", view=True)

        trace = []
        if kernel == "NUTS":
            with model:
                step = pm.NUTS(target_accept=targetAccRate)
                trace = pm.sample(tune=burnInCount, draws=sampleCount, step=step, chains=chainCount, return_inferencedata=True, nuts_sampler="numpyro")
        else:
            raise Exception("Unrecognised kernel name!")
        
        total_samples = chainCount*sampleCount
        self._writeData(kernel, trace, total_samples, N)

    def runWithWavelengths(self, surfaceFunction=None, kernel="NUTS", burnInCount=2000, sampleCount=5000, chainCount=4, targetAccRate=0.238, factor=1.0, showGraph=False):
        # Normalize the scatter by the flat plate response
        factorizedScatter = self.trueScatter/factor
        factorizedScatter = factorizedScatter

        # Set the number of waves to reconstruct
        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        N = self.cosineCount

        # Set the prior distributions
        ampSigmas = self.getAmplitudeProposal()
        phaseSigmas = self.getPhaseProposal()
        with pm.Model() as model:

            # Proposal for each set of 2 params                  
            amplitudes = pm.TruncatedNormal('amp', sigma=ampSigmas, lower=0.0, upper=0.025, shape=(N,), initval=1e-10 + pt.zeros(N))
            phases = pm.TruncatedNormal('phase', mu=0.5, sigma=phaseSigmas, lower=0.0, upper=0.999, shape=(N,), initval=1e-10 + pt.zeros(N))

            # Use known range of wavelengths
            wavelengths = pm.Data('wl', self._wavelengths)
            error = 0.0075  #Scales the error covariance matrix for the error and noise between receivers

            # Surface function with evaluated parameters
            def newFunction(x):
                return surfaceFunction(pt.as_tensor_variable(x), amplitudes, wavelengths, phases)

            # Scatter operation which maintains symbolic links
            # Gives the same results as Directed2DVectorized class
            KA_Object = Directed2DVectorisedSymbolic(
                self.sourceLocation,
                self.receiverLocations,
                newFunction,
                self.sourceFrequency,
                self.pistonAperture,
                self.sourceAngle,
                userMinMax=[-1,1],
                userSamples=self.userSampleDensity,
                absolute=False
            )
            scatter = KA_Object.Scatter(absolute=True, norm=False)
            scatter = scatter / factor
            KA_Object.surfaceChecker() #Adds pm.Potential penalty if kirchoff criteria not met

            # Likelihood: Compare predicted response to observed data
            likelihood = pm.MvNormal('obs', mu=scatter, cov=np.eye(len(factorizedScatter))*error, observed=factorizedScatter)

        # View model graph
        if showGraph:
            graph = pm.model_to_graphviz(model)
            graph.render("model_graph", format="png", view=True)

        trace = []
        if kernel == "NUTS":
            with model:
                step = pm.NUTS(target_accept=targetAccRate)
                trace = pm.sample(tune=burnInCount, draws=sampleCount, step=step, chains=chainCount, return_inferencedata=True, nuts_sampler="numpyro")
        else:
            raise Exception("Unrecognised kernel name!")

        total_samples = chainCount*sampleCount
        self._writeData(kernel, trace, total_samples, N)      

    def setAmplitudeProposal(self, stds):
        '''
        Set the standard deviation of the amplitude proposal distributions

        Inputs:
        stds: An array of standard deviations. Must have cosineCount members
        '''
        assert self.cosineCount == len(stds)
        self._amplitudeProposal = stds

    def getAmplitudeProposal(self):
        ampSigmas = []
        if self._amplitudeProposal is not None:
            ampSigmas = self._amplitudeProposal
        else:
            ampSigmas = 0.003 + np.zeros(self.cosineCount)
        return ampSigmas

    def setPhaseProposal(self, stds):
        '''
        Set the standard deviation of the phase proposal distributions

        Inputs:
        stds: An array of standard deviations. Must have cosineCount members
        '''
        # Make sure phase is already 2*pi scaled
        assert self.cosineCount == len(stds)
        self._phaseProposal = stds

    def getPhaseProposal(self):
        phaseSigmas = []
        if self._phaseProposal is not None:
            phaseSigmas = self._phaseProposal
        else:
            phaseSigmas = 0.1 + np.zeros(self.cosineCount)
        return phaseSigmas
    
    def setWavelengthProposal(self, stds):
        '''
        Set the standard deviation of the wavelength proposal distributions. Will clear wavelength data if set.

        Inputs:
        stds: An array of standard deviations. Must have cosineCount members
        '''
        # Make sure phase is already 2*pi scaled
        assert self.cosineCount == len(stds)
        self._wavelengthProposal = stds
        self._wavelengths = None

    def getWavelengthProposal(self):
        assert self._wavelengths == None

        wlSigmas = []
        if self._wavelengthProposal is not None:
            wlSigmas = self._wavelengthProposal
        else:
            wlSigmas = 0.08 + np.zeros(self.cosineCount)
        return wlSigmas

    def setWavelengths(self, wavelengths):
        '''
        Set the known wavelengths to reduce complexity. Will clear wavelength proposal data if set

        Inputs:
        wavelengths: An array of known wavelengths. Must have cosineCount members
        '''
        assert self.cosineCount == len(wavelengths)
        self._wavelengths = wavelengths
        self._wavelengthProposal = None
 
    def _writeData(self, kernel, trace, totalSamples, N):
        # Flatten arrays with some numpy shaping trickery
        # Gives us columns so that each set of 3 params are next to each other
        posterior_samples = []
        posterior = trace.posterior
        amps = posterior['amp'].values.transpose(2, 0, 1).reshape(N, totalSamples)

        wls = []
        if self._wavelengths is not None:
            wls = np.tile(np.array(self._wavelengths), (totalSamples, 1)).T
        else:
            wls = posterior['wl'].values.transpose(2, 0, 1).reshape(N, totalSamples)
        phases = posterior['phase'].values.transpose(2, 0, 1).reshape(N, totalSamples)

        posterior_samples = np.row_stack((amps,wls,phases)).T
        posterior_samples = posterior_samples.reshape(totalSamples, 3, N).transpose(0, 2, 1)
        posterior_samples = posterior_samples.reshape(totalSamples, N*3)

        # Save to csv with header
        np.savetxt("results/" + kernel + ".csv", posterior_samples, delimiter=",", header=self._generateHeader(), fmt="%s")
        print("MCMC Results saved!")

        # Save parameter store
        self.posteriorSamples = posterior_samples
        self.trace = trace

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
        az.plot_trace(self.trace, combined=False)