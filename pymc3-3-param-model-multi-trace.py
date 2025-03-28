import numpy as np
import matplotlib.pyplot as plt
import jax
import arviz as az
import corner
import scienceplots
import os
from tqdm import tqdm

from src.Directed2DVectorized import Directed2DVectorised
from src.AcousticParameterMCMC import AcousticParameterMCMC

from src.SymbolicMath import SymCosineSumSurfaceVectorized
from src.SymbolicMath import SymCosineSurface

def modelRun():
    plt.style.use('science')
    plt.rcParams["font.family"] = "Bitstream Charter"

    # Check jax backend
    print("jax device: ", jax.default_backend(), " ", jax.device_count())

    cosine_count = 1

    output_folder = "results/forward-model-amp-only"

    # Microphone array
    SourceLocation = [-0.20049999999999987,0.21884]
    ReceiverLocationsX =[-0.131500000000000,-0.112000000000000,-0.0924999999999999,
    -0.0719999999999999,-0.0514999999999999,-0.0309999999999999,-0.0104999999999999,0.00750000000000015,
    0.0290000000000001,0.0480000000000001,0.0685000000000001,0.0900000000000001,0.109500000000000,
    0.129000000000000,0.151000000000000,0.166500000000000,0.187500000000000,
    0.208000000000000,0.229500000000000,0.249500000000000,0.268500000000000,
    0.289000000000000,0.310000000000000,0.327500000000000,0.347500000000000,
    0.368500000000000,0.389000000000000,0.409500000000000,0.426500000000000,
    0.448500000000000,0.467000000000000,0.488500000000000,0.508500000000000,
    0.528000000000000]
    ReceiverLocationsY = [0.283024319442090,
                          0.283006369700232,
                          0.283208419958374,
                          0.283436216383600,
                          0.283364012808826,
                          0.283731809234052,
                          0.283969605659278,
                          0.283593036666793,
                          0.283656579259103,
                          0.284565756433703,
                          0.283773552858929,
                          0.283947095451239,
                          0.283819145709381,
                          0.284651195967522,
                          0.284087611643374,
                          0.284616677233179,
                          0.284777346741947,
                          0.285195143167173,
                          0.284838685759483,
                          0.285313609101167,
                          0.285382786275767,
                          0.285680582700993,
                          0.284461252209761,
                          0.284291810133734,
                          0.284866733475418,
                          0.285097402984186,
                          0.286005199409412,
                          0.285352995834638,
                          0.285200680675069,
                          0.286317096350921,
                          0.286443400441979,
                          0.286756943034289,
                          0.287121866375973,
                          0.286673916634115]
    RecLoc = []
    for i in range(len(ReceiverLocationsX)):
        RecLoc.append([ReceiverLocationsX[i],ReceiverLocationsY[i]])

    SourceAngle = np.pi / 3
    sourceFreq = 14_000
    sample_count = 100_000
    burn_in_count = 10_000
    userSamples = 700

    # Scatter data requirements
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)

    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SymCosineSurface(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,sourceFreq,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter

    # Generate scatter for varying amps and wavelengths
    # Only vary two for now
    p_count = 25

    generate_data = True
    if generate_data:
        params = []
        scatters = []
        ampspace = np.random.uniform(0.0, 0.01, p_count)
        wlspace = np.random.uniform(0.05, 0.05, p_count)
        for i in range(p_count):
            params.append((ampspace[i], wlspace[i], 0.0))
            
        np.savetxt("results/amp_wl_phase_params_14KHz.csv", params, delimiter=",")
            
        for p in tqdm (range(len(params)), desc="Generating param responses with noise"):
            s = generate_microphone_pressure(params[p])
            s += np.random.normal(loc=0.0, scale=0.05*factor, size=(34,))
            s = np.abs(s)
            scatters.append(s)

        params = np.array(params)
        scatters = np.array(scatters)

        plt.hist(params[:, 0], bins=50, alpha=0.5, label="amp")
        plt.legend()
        plt.show()

        plt.scatter(params[:, 0], params[:, 0])
        plt.show()

        plt.hist(params[:, 1], bins=50, alpha=0.5, label="wl")
        plt.legend()
        plt.show()

        plt.scatter(params[:, 1], params[:, 1])
        plt.show()
        np.savetxt("results/amp_wl_phase_scatter_14KHz.csv", scatters, delimiter=",")

    params = np.loadtxt("results/amp_wl_phase_params_14KHz.csv", delimiter=",")
    scatters = np.loadtxt("results/amp_wl_phase_scatter_14KHz.csv", delimiter=",")

    # Update flat response factor here to get in model scale
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)

    # Run the samplers and generate traces for each one
    run_samplers = False
    kernel = "NUTS"
    userSamples = 700
    current_p = ""

    if run_samplers:
        predicted_amps = []
        predicted_wls = []

        pbar = tqdm(range(len(params)))
        for i in pbar:
            p = params[i]
            s = scatters[i]
            current_p = "AMP-" + str(p[0]) + "-WL-" + str(p[1])
            pbar.set_postfix({'Current Params': current_p})
            filename = output_folder + "/" + current_p

            # Create the model MCMC sampler
            mcmc = AcousticParameterMCMC(cosineCount=cosine_count, 
                                            sourceLocation=SourceLocation, 
                                            sourceAngle=SourceAngle,
                                            receiverLocations=RecLoc, 
                                            truescatter=s, 
                                            userSampleDensity=userSamples, 
                                            sourceFrequency=sourceFreq)
            
            mcmc.setFileName(filename)
            mcmc.setAmplitudeProposal(np.array([0.01]))
            mcmc.setWavelengthProposal(np.array([0.2]))
            mcmc.setError(0.02)
            
            # Run the model MCMC sampler
            mcmc.run(kernel=kernel, 
                        chainCount=1, 
                        surfaceFunction=SymCosineSumSurfaceVectorized, 
                        burnInCount=burn_in_count, 
                        sampleCount=sample_count, 
                        scaleTrueScatter=True,
                        targetAccRate=0.9)
            
    # Read in all CSV's from the large set group
    def list_csv_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    
    def tokenize_filename(filename):
        return filename.rsplit('/', 1)[-1].split('-')

    csv_files = list_csv_files(output_folder)
    predicted_amps = []
    predicted_wls = []
    true_amps = []
    true_wls = []
    for file in csv_files:
        posterior_samples_grouped = np.array(AcousticParameterMCMC.LoadCSVData(file))
        posterior_samples_grouped = AcousticParameterMCMC.AngularMeanData(posterior_samples_grouped, cosine_count)
        posterior_samples_grouped = np.mean(posterior_samples_grouped,axis=0).squeeze()

        predicted_amps.append(posterior_samples_grouped[0])
        predicted_wls.append(posterior_samples_grouped[1])
        tokens = tokenize_filename(file)
        true_amps.append(float(tokens[1]))
        true_wls.append(float("0." + tokens[3].split('.')[1]))

    predicted_amps = np.array(predicted_amps)
    predicted_wls = np.array(predicted_wls)
    true_amps = np.array(true_amps)
    true_wls = np.array(true_wls)

    plt.scatter(true_amps, predicted_amps)
    plt.scatter(true_amps, true_amps)
    plt.show()
    plt.scatter(true_wls, predicted_wls)
    plt.scatter(true_wls, true_wls)
    plt.show()


if __name__ == "__main__":
    modelRun()