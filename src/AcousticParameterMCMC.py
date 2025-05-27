import pymc as pm
import numpy as np
import arviz as az
import shutil
import timeit

import pytensor
import pytensor.tensor as pt

from src.Directed2DVectorized import Directed2DVectorised
from src.Directed2DVectorized import Directed2DVectorisedSymbolic
from src.SymbolicMath import AngularMean
from src.SymbolicMath import GetSpecularIndices

class AcousticParameterMCMC:
    '''
    A class that runs an MCMC sampling model to recover the parameters of a cosine series acoustic surface.

    Works on sum of cosine waves. Units are metres for amplitude and wavelength, and radians / 2*pi for phase.
    '''

    _amplitudeProposal = None
    _wavelengthProposal = None
    _wavelengths = None
    _phaseScale = 1.0 / (2.0 * np.pi)
    _error = 0.001
    _filename = ""

    def GenerateFactor(sourceLocation, sourceAngle, receiverLocations, pistonAperture, sourceFrequency=14_000, userSamples=700):
        '''
        Generates a factor to scale response by flat plate response

        Parameters:
            sourceLocation (tuple): The location in world space of the acoustic source.
            sourceAngle (float): Angle the acoustic source is at
            recieverLocations (Array or List): The locations in world space of the recievers. Must be an array of (x,y) coordinates.
            pistonAperture (float): The aperture size of the baffled piston approximation
            sourceFrequency (float): The frequency of the source acoustic wave
            userSampleDensity (int): The density of samples in the simulated scatter. Relates to resolution rather than sampling rate. 

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

        Parameters:
            path (str): The path to the data

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
    
    def AngularMeanData(posteriorSamples, cosineCount):
        '''
        For a loaded csv data, will make the phase the angular mean for all samples

        Parameters:
            posteriorSamples (np.array): The samples to get the angular mean for. Must match the cosine count.
            cosineCount (int): The number of waves to calculate for

        Returns:
            np.array: The samples with the angular mean
        '''
        for j in range(cosineCount):
            phases = [t[j][2] for t in posteriorSamples]
            mean_angle = AngularMean(np.array(phases))
            for i in range(len(posteriorSamples)):
                t = posteriorSamples[i]
                posteriorSamples[i][j] = (t[j][0], t[j][1], mean_angle)
        return posteriorSamples
    
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

        Parameters:
            cosineCount (int): Number of surface parameters to recover (Amplitude, Wavelength, Phase tuple). Must be a multiple of 3.
            sourceLocation (tuple): The location in world space of the acoustic source.
            sourceAngle (float): Angle the acoustic source is at
            recieverLocations (Array or List): The locations in world space of the recievers. Must be an array of (x,y) coordinates.
            truescatter (np.array): The true scatter data that the chain compares response to. Must be amplitude only data for each reciever location.
            userSampleDensity (int): The density of samples in the simulated scatter. Relates to resolution rather than sampling rate.
            sourceFrequency (float): The frequency of the source acoustic wave
            pistonAperture (float): The aperture size of the baffled piston approximation
        '''

        self.cosineCount = cosineCount
        self.sourceLocation = sourceLocation
        self.receiverLocations = receiverLocations
        self.sourceFrequency = sourceFrequency
        self.userSampleDensity = userSampleDensity
        self.trueScatter = truescatter
        self.pistonAperture = pistonAperture
        self.sourceAngle = sourceAngle

    def run(self, surfaceFunction=None, kernel="NUTS", burnInCount=2000, sampleCount=5000, chainCount=4, targetAccRate=0.238, scaleTrueScatter=True, truncateInds=False, showGraph=False):
        '''
        Runs the pymc model with the selected kernel. Stores the parameters and trace afterwards.
        Also outputs the parameters as a .csv with the file name "kernel + .csv"

        Parameters:
            surfaceFunction (func): A function that takes x and 3 parameters.
            kernel (str): The type of kernel to use. Currently supports "NUTS"
            burnInCount (int): The number of burn in iterations. Will be discarded.
            sampleCount (int): The number of samples to run the chain for.
            chainCount (int): The number of parallel chains to run. Useful for getting diagnostics. For GPU need multiple GPUs to run.
            tragetAccRate (float): The target acceptance rate. Between 0 and 1. Default is 0.8.
            scaleTrueScatter (bool): Sets whether to scale the true scatter by the flat plate response.
            truncateInds (bool): Sets whether to truncate indices to specular region
            showGraph (bool): Shows and saves an image of the model graph
        '''

        # Raises memory usage but should be more efficient
        pytensor.config.allow_gc=False
        self._kernel = kernel

        # Generate the factor to scale by the flat plate response
        factor = 1.0
        if scaleTrueScatter:
            factor = self.generateFactor()

        # Specular Restriction
        indices = [0,len(self.trueScatter)]
        if truncateInds:
            indices = GetSpecularIndices(self.sourceLocation, self.sourceAngle, self.sourceFrequency, self.receiverLocations)
            indices[0]  = int(indices[0]  * 0.8)
            indices[-1] = int(indices[-1] * 1.2)
            print("Restricted MCMC to indices: ", indices[0], " to ", indices[-1])

        # Normalize the scatter to the flat plate response
        factorizedScatter = self.trueScatter[indices[0]:indices[-1]]/factor

        # Set the number of waves to reconstruct
        if self.cosineCount == 0:
            raise Exception("Cannot have a surface with no parameters!")
        N = self.cosineCount

        # Set covariance matrix
        cov_matrix = np.eye(len(factorizedScatter)) * self._error
        chol = np.linalg.cholesky(cov_matrix)

        # Start a timer
        self._time = timeit.default_timer()

        # Set the prior distributions
        ampSigmas = self.getAmplitudeProposal()

        if self._wavelengths is not None:
            self._model = self._generate2ParamModel(ampSigmas, indices, chol, factorizedScatter, surfaceFunction, N, factor)
        else:
            # Set wavelengths here as otherwise aren't needed
            wlSigmas = self.getWavelengthProposal()
            self._model = self._generate3ParamModel(ampSigmas, wlSigmas, indices, chol, factorizedScatter, surfaceFunction, N, factor) 

        # View model graph
        if showGraph:
            graph = pm.model_to_graphviz(self._model)
            graph.render("results/model_graph", format="png", view=True) 

        trace = []
        if kernel == "NUTS":
            with self._model:
                step = pm.NUTS(target_accept=targetAccRate)
                trace = pm.sample(tune=burnInCount, draws=sampleCount, step=step, chains=chainCount, return_inferencedata=True, nuts_sampler="numpyro")
        else:
            raise Exception("Unrecognised kernel name!")
        
        total_samples = chainCount*sampleCount
        self._writeData(kernel, trace, total_samples, N)   

    def setAmplitudeProposal(self, stds):
        '''
        Set the standard deviation of the amplitude proposal distributions

        Parameters:
            stds (np.array): An array of standard deviations. Must have cosineCount members
        '''
        assert self.cosineCount == len(stds)
        self._amplitudeProposal = stds

    def getAmplitudeProposal(self):
        '''
        Get the standard deviations of the amplitude proposal distributions.

        Returns:
            np.array: The standard deviations of the amplitude proposal distributions
        '''
        ampSigmas = []
        if self._amplitudeProposal is not None:
            ampSigmas = self._amplitudeProposal
        else:
            ampSigmas = 0.003 + np.zeros(self.cosineCount)
        return ampSigmas

    def setPhaseProposal(self, stds):
        '''
        Set the standard deviation of the phase proposal distributions. In units of radians / 2*pi

        Parameters:
            stds (np.array): An array of standard deviations. Must have cosineCount members
        '''
        # Make sure phase is already 2*pi scaled
        assert self.cosineCount == len(stds)
        self._phaseProposal = stds

    def getPhaseProposal(self):
        '''
        Get the standard deviations of the phase proposal distributions.

        Returns:
            np.array: The standard deviations of the phase proposal distributions
        '''
        phaseSigmas = []
        if self._phaseProposal is not None:
            phaseSigmas = self._phaseProposal
        else:
            phaseSigmas = 0.1 + np.zeros(self.cosineCount)
        return phaseSigmas
    
    def setWavelengthProposal(self, stds):
        '''
        Set the standard deviation of the wavelength proposal distributions. Will clear wavelength data if set.

        Parameters:
            stds (np.array): An array of standard deviations. Must have cosineCount members
        '''
        # Make sure phase is already 2*pi scaled
        assert self.cosineCount == len(stds)
        self._wavelengthProposal = stds
        self._wavelengths = None

    def getWavelengthProposal(self):
        '''
        Get the standard deviations of the wavelength proposal distributions. Fails if wavelengths are directly set.

        Returns:
            np.array: The standard deviations of the wavelength proposal distributions
        '''
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

        Parameters:
            wavelengths (np.array): An array of known wavelengths. Must have cosineCount members
        '''
        assert self.cosineCount == len(wavelengths)
        self._wavelengths = wavelengths
        self._wavelengthProposal = None

    def setError(self, error):
        '''
        Set the variance value between receiver values in the covariance matrix.
        Should be noted that variance is the square of standard deviation.

        Parameters:
            error (float): The error value to use. Must be greater than zero.
        '''
        assert error > 0.0
        self._error = error

    def plotTrace(self):
        '''
        Plots the trace of the model with arviz
        '''
        import matplotlib.pyplot as plt

        az.rcParams['plot.max_subplots'] = 40
        summary = az.summary(self.trace, circ_var_names=["phase"], round_to=5)
        print(summary)

        with open("results/output.txt", "w") as txt:
            txt.write(summary.to_string())

        az.plot_trace(self.trace, var_names=["phase"], divergences=False, compact=False, combined=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        for ax in fig.axes:
            for line in ax.get_lines():
                line.set_color("#2c7bb6")
        plt.savefig("results/" + self._kernel.lower() + "-phase-pymc-trace.png")

        # Amp
        az.plot_trace(self.trace, var_names=["amp"], divergences=False, compact=False, combined=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        for ax in fig.axes:
            for line in ax.get_lines():
                line.set_color("#2c7bb6")
        plt.savefig("results/" + self._kernel.lower() + "-amp-pymc-trace.png")

        # Set include vars to ignore internal mu and sigma for phase - these values aren't of interest

        # WL
        if self._wavelengths is None:
            #scaled_trace.posterior["wl"] = self.trace.posterior["wl"]
            az.plot_trace(self.trace, var_names=["wl"], divergences=False, compact=False, combined=False)
            fig = plt.gcf()
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            for ax in fig.axes:
                for line in ax.get_lines():
                    line.set_color("#2c7bb6")
            plt.savefig("results/" + self._kernel.lower() + "-wl-pymc-trace.png")

    def setFileName(self, name):
        '''
        Sets the output file name.

        Parameters:
            name (str): The name of the output csv file
        '''
        self._filename = name
 
    def _writeData(self, kernel, trace, totalSamples, N):

        # Filename
        filename = self._filename
        if not filename:
            filename = "results/" + kernel

        # Get processing time
        dt = timeit.default_timer() - self._time
        with open(filename + "-time.txt", "w") as txt:
            print("Duration (s): " + str(dt))
            txt.write("Duration (s): " + str(dt))

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
        filename += ".csv"
        np.savetxt(filename, posterior_samples, delimiter=",", header=self._generateHeader(), fmt="%s")
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
    
    def _generate2ParamModel(self, ampSigmas, indices, chol, trueScatter, surfaceFunction=None, N=1, factor=1.0):
        '''
        Generates the 2 parameter model for up to N cosine waves.

        Parameters:
            ampSigmas (np.array): The standard deviations of the amplitude proposals.
            indices (List): The indices to restrict the reconstruction to.
            chol (np.array): The cholesky decomposition of the covariance matrix.
            trueScatter (np.array): The observed data to compare the model output to.
            surfaceFunction (func): The surface function to use in the scatter model.
            N (int): The number of cosine waves.
            factor (float): The normalizing factor to scale results by.

        Returns:
            pymc.Model: The model object
        '''

        with pm.Model() as model:

            # Proposal for each set of 2 params     
            # For phase initialize to even spacing for each wave             
            amplitudes = pm.TruncatedNormal('amp', sigma=ampSigmas, lower=0.0, shape=(N,), initval=1e-10 + pt.zeros(N))
            phases = pm.Uniform('phase', lower=0.0, upper=2.0*np.pi, initval=1e-6 + pt.zeros(shape=(N,)), shape=(N,))

            # Von mises is very slow at higher params so disable for now (using this for 40 param recovery)
            #phases = pm.VonMises('phase', mu=0.0, kappa=0.0, shape=(N,), initval=1e-6 + pt.zeros(shape=(N,)))

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
            scatter = scatter[indices[0]:indices[-1]] / factor
            KA_Object.surfaceChecker() #Adds pm.Potential penalty if kirchoff criteria not met

            # Likelihood: Compare predicted response to observed data
            likelihood = pm.MvNormal('obs', mu=scatter, chol=chol, observed=trueScatter)
        return model

    def _generate3ParamModel(self, ampSigmas, wlSigmas, indices, chol, trueScatter, surfaceFunction=None, N=1, factor=1.0):
        '''
        Generates the 3 parameter model for up to N cosine waves.

        Parameters:
            ampSigmas (np.array): The standard deviations of the amplitude proposals.
            wlSigma (np.array): The standard deviations of the wavelength proposals.
            indices (List): The indices to restrict the reconstruction to.
            chol (np.array): The cholesky decomposition of the covariance matrix.
            trueScatter (np.array): The observed data to compare the model output to.
            surfaceFunction (func): The surface function to use in the scatter model.
            N (int): The number of cosine waves.
            factor (float): The normalizing factor to scale results by.

        Returns:
            pymc.Model: The model object
        '''

        with pm.Model() as model:

            # Proposal for each set of 3 params      
            # Von Mises for now as naturally works for angles but can be slower for more dimensions
            # I'll deal with that when this bit of code goes there   
            # 
            # Truncate to 2.0 sigma which is equivalent to 95% of the distribution - if true values are beyond that the proposal is probaby bad   
            amplitudes = pm.TruncatedNormal('amp', lower=0.0, upper=2.0*ampSigmas, sigma=ampSigmas, shape=(N,), initval=1e-6 + pt.zeros(shape=(N,)))  
            wavelengths = pm.TruncatedNormal('wl', lower=0.0, upper=2.0*wlSigmas, sigma=wlSigmas, shape=(N,), initval=wlSigmas)
            phases = pm.VonMises('phase', mu=0.0, kappa=0.0, shape=(N,), initval=1e-6 + pt.zeros(shape=(N,)))

            # Surface function with evaluated parameters
            def newFunction(x):
                return surfaceFunction(pt.as_tensor_variable(x), amplitudes, wavelengths, phases*self._phaseScale)

            # Prevent label switching when wavelengths are not provided
            wavelength_penalty = pt.sum(pt.switch(pt.ge(wavelengths[:-1], wavelengths[1:]), 0, -1e6))
            pm.Potential("order-penalty", wavelength_penalty)

            # Scatter operation which maintains symbolic links
            # Gives the same results as original Directed2DVectorized class
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
            scatter = scatter[indices[0]:indices[-1]] / factor
            KA_Object.surfaceChecker() #Adds pm.Potential penalty if kirchoff criteria not met
            
            # Likelihood: Compare predicted response to observed data
            likelihood = pm.MvNormal('obs', mu=scatter, chol=chol, observed=trueScatter)

        return model