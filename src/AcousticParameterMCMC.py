import pymc as pm
import numpy as np
import arviz as az
import shutil

import pytensor
import pytensor.tensor as pt

from src.Directed2DVectorized import Directed2DVectorised
from src.Directed2DVectorized import Directed2DVectorisedSymbolic
from src.SymbolicMath import SymAngularMean

class AcousticParameterMCMC:
    '''
    A class that runs an MCMC sampling model to recover the parameters of a cosine series acoustic surface
    '''

    _amplitudeProposal = None
    _wavelengthProposal = None
    _wavelengths = None
    _phaseScale = 1.0 / (2.0 * np.pi)
    _error = 0.001

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
    
    def AngularMeanData(posterior_samples, cosine_count):
        '''
        For a loaded csv data, will make the phase the angular mean for all samples

        Inputs:
        posterior_samples: The samples to get the angular mean for
        cosine_count: The number of waves to calculate for

        Returns:
        The samples with the angular mean
        '''
        for j in range(cosine_count):
            phases = [t[j][2] for t in posterior_samples]
            mean_angle = SymAngularMean(np.array(phases))
            for i in range(len(posterior_samples)):
                t = posterior_samples[i]
                posterior_samples[i][j] = (t[j][0], t[j][1], mean_angle)
        return posterior_samples
    
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
        self._kernel = kernel

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

        # Set the number of waves to reconstruct
        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        N = self.cosineCount

        # Set the prior distributions
        ampSigmas = self.getAmplitudeProposal()
        wlSigmas = self.getWavelengthProposal()

        phase_init_vals = np.linspace(1e-6, 2.0*np.pi, N+1)[:N]
        cov_matrix = np.eye(len(factorizedScatter)) * self._error
        chol = np.linalg.cholesky(cov_matrix)
        with pm.Model() as model:

            # Proposal for each set of 3 params      
            # Von Mises for now as naturally works for angles but can be slower for more dimensions
            # I'll deal with that when this bit of code goes there         
            wavelengths = pm.TruncatedNormal('wl', sigma=wlSigmas, lower=0.0, shape=(N,))
            amplitudes = pm.TruncatedNormal('amp', sigma=ampSigmas, lower=0.0, shape=(N,), initval=1e-6 + pt.zeros(shape=(N,)))
            phases = pm.VonMises('phase', mu=0.0, kappa=0.0, initval=phase_init_vals, shape=(N,))

            # Surface function with evaluated parameters
            def newFunction(x):
                return surfaceFunction(pt.as_tensor_variable(x), amplitudes, wavelengths, phases*self._phaseScale)

            # Prevent label switching when wavelengths are not provided
            wavelength_penalty = pt.sum(pt.switch(pt.ge(wavelengths[:-1], wavelengths[1:]), 0, 1e6))
            pm.Potential("order-penalty", wavelength_penalty)

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
            likelihood = pm.MvNormal('obs', mu=scatter, chol=chol, observed=factorizedScatter)

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

        # Set the number of waves to reconstruct
        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        N = self.cosineCount

        # Set the prior distributions
        ampSigmas = self.getAmplitudeProposal()

        # Phases start spread out to avoid amplitudes bunching at 0
        phase_init_vals = np.linspace(1e-10, 2.0*np.pi, N+1)[:N]

        # Create the covariance matrix
        cov_matrix = np.eye(len(factorizedScatter)) * self._error
        chol = np.linalg.cholesky(cov_matrix)
        with pm.Model() as model:

            # Proposal for each set of 2 params     
            # For phase initialize to even spacing for each wave             
            amplitudes = pm.TruncatedNormal('amp', sigma=ampSigmas, lower=0.0, shape=(N,), initval=1e-10 + pt.zeros(N))
            phases = pm.Uniform('phase', lower=0.0, upper=2.0*np.pi, initval=phase_init_vals, shape=(N,))

            # Use known range of wavelengths
            wavelengths = pm.Data('wl', self._wavelengths)

            # Surface function with evaluated parameters
            def newFunction(x):
                return surfaceFunction(pt.as_tensor_variable(x), amplitudes, wavelengths, phases*self._phaseScale)

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
            likelihood = pm.MvNormal('obs', mu=scatter, chol=chol, observed=factorizedScatter)

        # View model graph
        if showGraph:
            graph = pm.model_to_graphviz(model)
            graph.render("model_graph", format="png", view=True)

        trace = []
        if kernel == "NUTS":
            with model:
                step = pm.NUTS(target_accept=targetAccRate)
                trace = pm.sample(tune=burnInCount, draws=sampleCount, step=step, chains=chainCount, return_inferencedata=True, nuts_sampler="numpyro")
        elif kernel == "ADVI":
            with model:
                advi_fit = pm.fit(n=burnInCount, method="advi")
                trace = advi_fit.sample(sampleCount)
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

    def setError(self, error):
        '''
        Set the error value between receiver values in the covariance matrix

        Inputs:
        error: The error value to use. Must be greater than zero.
        '''
        assert error > 0.0
        self._error = error
 
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
        phases = posterior['phase'].values.transpose(2, 0, 1).reshape(N, totalSamples) * np.array(self._phaseScale).reshape(-1, 1)

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
        import matplotlib.pyplot as plt

        az.rcParams['plot.max_subplots'] = 40
        summary = az.summary(self.trace, circ_var_names=["phase"])
        print(summary)

        with open("results/output.txt", "w") as txt:
            txt.write(summary.to_string())

        az.plot_energy(self.trace)

        # Rescale the trace values and deviations
        scaled_trace = self.trace.copy()

        # Phase
        scaled_trace.posterior["phase"] = self.trace.posterior["phase"]
        az.plot_trace(scaled_trace, var_names=["phase"], divergences=False, compact=False, combined=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.savefig("results/" + self._kernel.lower() + "-phase-pymc-trace.png")

        # Amp
        scaled_trace.posterior["amp"] = self.trace.posterior["amp"]
        az.plot_trace(scaled_trace, var_names=["amp"], divergences=False, compact=False, combined=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.savefig("results/" + self._kernel.lower() + "-amp-pymc-trace.png")

        # Set include vars to ignore internal mu and sigma for phase - these values aren't of interest

        # WL
        if self._wavelengths is None:
            scaled_trace.posterior["wl"] = self.trace.posterior["wl"]
            az.plot_trace(scaled_trace, var_names=["wl"], divergences=False, compact=False, combined=False)
            fig = plt.gcf()
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            plt.savefig("results/" + self._kernel.lower() + "-wl-pymc-trace.png")