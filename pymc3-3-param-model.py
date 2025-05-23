import numpy as np
import matplotlib.pyplot as plt
import jax
import arviz as az
import corner
import scienceplots

from src.Directed2DVectorized import Directed2DVectorised
from src.AcousticParameterMCMC import AcousticParameterMCMC


from src.SymbolicMath import SymCosineSumSurfaceVectorized
from src.SymbolicMath import SymCosineSumSurface as SurfaceFunctionMulti
from src.SymbolicMath import SymCosineSumSurface
from src.SymbolicMath import SymCosineSurface
from src.SymbolicMath import SymRMS

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

    #LVV 2019 data at 14kHz
    comp = [0.01260586, 0.03210286, 0.06278351, 0.05601213, 0.03996005, 0.02276676,
        0.03520109, 0.05303403, 0.07947386, 0.06281864, 0.04293401, 0.0486769,
        0.04348036, 0.05091185, 0.05267429, 0.05519047, 0.0744659,  0.06930464,
        0.03456355, 0.02224533, 0.03089714, 0.02583821, 0.03186504, 0.02806474,
        0.05334908, 0.03035023, 0.0342647,  0.0377431,  0.03844866, 0.03600139,
        0.02590109, 0.01360588, 0.00829887, 0.00865361]
    
    sourceFreq = 14_000

    # Real data
    truescatter = comp

    p = (0.0015, 0.05, 0.0)

    userSamples = 700
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)
    pc_noise = 0.0
    def noise_sigma(s, pc):
        '''
        Get scale of noise from rms average of the signal
        '''
        return (SymRMS(np.array(s)))*pc

    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SymCosineSurface(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,sourceFreq,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        scatter += np.random.normal(loc=0.0, scale=noise_sigma(scatter, pc_noise), size=(34,))
        scatter = np.abs(scatter)
        return scatter
    
    plt.figure(figsize=(16,9))
    plt.grid()
    plt.plot(truescatter, color='black', label="Experimental Scattering Data")
    kascatter = generate_microphone_pressure(p)
    plt.plot(kascatter, color='blue', label="Kirchhoff Approximation Pressure Field of True Surface (No additive noise)")
    plt.xlabel("Microphone Index")
    plt.ylabel("Response")
    plt.legend()
    plt.savefig("results/response KA-Experimental.png")
    plt.show()

    # True params
    #p = [0.0015, 0.05, 0.0]

    sample_count = 25_000
    burn_in_count = 25_000
    run_model = True
    kernel = "NUTS"
    userSamples = 700
    if run_model:
        # Create the model MCMC sampler
        mcmc = AcousticParameterMCMC(cosineCount=cosine_count, 
                                     sourceLocation=SourceLocation, 
                                     sourceAngle=SourceAngle,
                                     receiverLocations=RecLoc, 
                                     truescatter=truescatter, 
                                     userSampleDensity=userSamples, 
                                     sourceFrequency=sourceFreq)
        
        mcmc.setAmplitudeProposal(np.array([0.01]))
        mcmc.setWavelengthProposal(np.array([0.1]))
        mcmc.setError(0.15**2)
        
        # Run the model MCMC sampler
        mcmc.run(kernel=kernel, 
                 chainCount=1, 
                 surfaceFunction=SymCosineSumSurfaceVectorized, 
                 burnInCount=burn_in_count, 
                 sampleCount=sample_count, 
                 scaleTrueScatter=True,
                 targetAccRate=0.9)
        
        mcmc.plotTrace()
    
    plt.rcParams.update({'font.size': 18})

    # Convert each row into 3-element tuples for better multi
    posterior_samples_grouped = np.array(AcousticParameterMCMC.LoadCSVData("results/" + kernel + ".csv"))

    # Scale true scatter here to compare to parameters
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)
    truescatter = truescatter / factor

    # Creating mean surfaces
    print("Creating mean parameters surface")
    x = np.linspace(0,0.6,500)

    hmm2 = posterior_samples_grouped.copy()
    hmm2 = AcousticParameterMCMC.AngularMeanData(hmm2, cosine_count)
    hmm2 = np.mean(hmm2,axis=0)
    mean = SymCosineSumSurface(x,hmm2)
    true = SymCosineSurface(x,p).copy()

    # Creating mean of all surfaces
    print("Creating individual surfaces")
    surfs = []
    x = np.linspace(0,0.6,500)
    for _ in posterior_samples_grouped:
        surfs.append(SymCosineSumSurface(x,_))

    print("Taking mean of all combined surfaces")
    mean_surf = np.mean(surfs,axis=0)

    print("Creating HDI interval")
    mins = []
    maxx = []
    for _ in range(500):
        vals = az.hdi(np.array(surfs).T[_],hdi_prob=0.68)
        mins.append(vals[0])
        maxx.append(vals[1])

    plt.figure(figsize = (16,6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)
    plt.fill_between(x,mins,maxx,color='#ffffbf',label='Credible interval (68\%)')
    plt.plot(x, mins, color='black', linewidth=0.75)  # Lower boundary
    plt.plot(x, maxx, color='black', linewidth=0.75)  # Upper boundary
    plt.plot(x,mean_surf, color="#2c7bb6", linestyle="dashed", label='Surface formed from the mean of the model surfaces')
    plt.plot(x,mean, color="#2c7bb6",label='Surface formed from the mean of the model parameters')
    plt.plot(x,true, color="#d7191c",label='True surface')

    choice_count = 300
    b = np.random.choice(range(sample_count),choice_count)

    #plt.xlabel("x (m)")
    #plt.ylabel("Surface elevation (m)")
    #plt.legend()
    plt.savefig("results/" + kernel.lower() + " reconstruction.png")

    print("Creating response plot")
    plt.figure(figsize=(16,6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)
    plt.plot(truescatter, color="#d7191c")

    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SurfaceFunctionMulti(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,sourceFreq,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor   
    
    for i in range(len(b)):
        plt.plot(generate_microphone_pressure(posterior_samples_grouped[b[i]]), 'k', color="black", alpha=0.04)
    
    plt.xlabel("Microphone index")
    plt.ylabel("Normalized Response")
    plt.savefig("results/" + kernel + " traces.png")

    import corner
    fig = corner.corner(np.array(posterior_samples_grouped)*np.array([100.0,100.0,2.0*np.pi]),bins=200,
              quantiles=[0.16, 0.5, 0.84],labels=[r"A (cm)", r"$\lambda$ (cm)", r"$\phi$ (rad)"],
              show_titles=True, title_fmt = ".4f", figsize=(16, 8))
    
    for ax in fig.axes:
        ax.grid(which='both')  # Enable both major and minor grid lines
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=0.75)

    plt.savefig("results/" + kernel + " corner.png")
    plt.show()

    plt.figure(figsize=(16,6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)

    x = np.linspace(0,0.6,500)

    plt.grid()
    plt.plot(x,np.sqrt((mean-true)**2)/np.std(true),color='#2c7bb6',linestyle='dashed',label="Relative error of the mean of the surfaces")
    plt.plot(x,np.sqrt((mean_surf-true)**2)/np.std(true),color='#2c7bb6',label="Relative error of the mean of the parameters")
    plt.xlabel("x (m)")
    plt.ylabel("Factored RSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    modelRun()