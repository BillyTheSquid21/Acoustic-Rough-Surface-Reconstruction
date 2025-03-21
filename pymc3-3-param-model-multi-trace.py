import numpy as np
import matplotlib.pyplot as plt
import jax
import arviz as az
import corner
import scienceplots
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
    sample_count = 6_000
    burn_in_count = 3_000
    userSamples = 700

    # Scatter data requirements
    factor = 1.0

    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SymCosineSurface(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,sourceFreq,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor  

    # Generate scatter for varying amps and wavelengths
    # Only vary two for now
    param_spacing = 10
    ampspace = np.linspace(0.001, 0.01, param_spacing)
    wlspace  = np.linspace(0.05, 0.1, param_spacing)

    gen_params_set = True
    if gen_params_set:
        params = []
        for w in range(param_spacing):
            for a in range(param_spacing):
                params.append((ampspace[a], wlspace[w], 0.0))

        params = np.array(params)
        np.savetxt("results/params.csv", params, delimiter=',')

        scatters = []
        for p in tqdm (params, desc="Generating param responses"):
            s = generate_microphone_pressure(p)
            # Add 2.5% noise
            maxval = np.max(s)
            s += np.random.normal(scale=0.025*maxval, size=(34,))
            s = np.abs(s)
            scatters.append(s)

        scatters = np.array(scatters)
        np.savetxt("results/scatters.csv", scatters, delimiter=',')

    params = np.loadtxt("results/params.csv", delimiter=',')
    scatters = np.loadtxt("results/scatters.csv", delimiter=',')

    # Update flat response factor here to get in model scale
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)

    # Run the samplers and generate traces for each one
    run_samplers = True
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
            filename = "results/large-set/" + current_p

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
            mcmc.setWavelengthProposal(np.array([0.1]))
            mcmc.setError(0.025 * np.max(s/factor))
            
            # Run the model MCMC sampler
            mcmc.run(kernel=kernel, 
                        chainCount=1, 
                        surfaceFunction=SymCosineSumSurfaceVectorized, 
                        burnInCount=burn_in_count, 
                        sampleCount=sample_count, 
                        scaleTrueScatter=True,
                        targetAccRate=0.9)
            
            posterior_samples_grouped = np.array(AcousticParameterMCMC.LoadCSVData(filename + ".csv"))
            posterior_samples_grouped = AcousticParameterMCMC.AngularMeanData(posterior_samples_grouped, cosine_count)
            posterior_samples_grouped = np.mean(posterior_samples_grouped,axis=0)

            predicted_amps.append((p[0], posterior_samples_grouped[0]))
            predicted_wls.append((p[1], posterior_samples_grouped[1]))

            import corner
            corner.corner(np.array(posterior_samples_grouped),bins=200,
                    quantiles=[0.16, 0.5, 0.84],labels=[r"$\zeta_1$", r"$\zeta_2$", r"$\zeta_3$"],
                    show_titles=True, title_fmt = ".4f")
            plt.savefig("results/" + kernel + " corner.png")
            plt.show()

        predicted_amps = np.array(predicted_amps)
        predicted_wls = np.array(predicted_wls)

        plt.scatter(predicted_amps)
        plt.show()
        plt.scatter(predicted_wls)
        plt.show()


if __name__ == "__main__":
    modelRun()