import numpy as np
import matplotlib.pyplot as plt
import jax
import arviz as az
import pytensor
import pytensor.tensor as pt

from src.Directed2DVectorized import Directed2DVectorised
from src.AcousticParameterMCMC import AcousticParameterMCMC

from src.SymbolicMath import SymCosineSumSurfaceVectorized
from src.SymbolicMath import SymCosineSumSurface as SurfaceFunctionMulti

def modelRun():

    # Check jax backend
    print("jax device: ", jax.default_backend(), " ", jax.device_count())

    # True params (synthetic!)
    ptrue = [(0.0045,0.075,0.00), (0.0015,0.1,0.01)]

    # True surface
    def trueF(x):
        return SurfaceFunctionMulti(x, ptrue)

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

    #LVV 2019 data at 14kHz
    comp = [0.01260586, 0.03210286, 0.06278351, 0.05601213, 0.03996005, 0.02276676,
        0.03520109, 0.05303403, 0.07947386, 0.06281864, 0.04293401, 0.0486769,
        0.04348036, 0.05091185, 0.05267429, 0.05519047, 0.0744659,  0.06930464,
        0.03456355, 0.02224533, 0.03089714, 0.02583821, 0.03186504, 0.02806474,
        0.05334908, 0.03035023, 0.0342647,  0.0377431,  0.03844866, 0.03600139,
        0.02590109, 0.01360588, 0.00829887, 0.00865361]

    # Real data
    #truescatter = comp

    # Test synthetic data
    KA_Object = Directed2DVectorised(SourceLocation,RecLoc,trueF,14_000,0.02,np.pi/3,'simp',userMinMax=[-1,1],userSamples=700,absolute=False)
    truescatter = KA_Object.Scatter(absolute=True,norm=False)

    # Synthetic data, add some random gaussian noise to the samples to make non deterministic
    # Noise is sampled around 0, with SD
    noise_sd = 0.0075
    for i in range(0, len(truescatter)):
        truescatter[i] += np.random.normal(0, noise_sd)

    # True function over linspace
    xsp = np.linspace(ReceiverLocationsX[0],ReceiverLocationsX[-1], 500)
    true = trueF(xsp)

    sample_count = 5_000
    burn_in_count = 5_000
    run_model = True
    #kernel = "metropolis-hastings"
    kernel = "NUTS"
    userSamples = 700
    posterior_samples = []
    if run_model:
        mcmc = AcousticParameterMCMC(cosineCount=3*len(ptrue), sourceLocation=SourceLocation, receiverLocations=RecLoc, truescatter=truescatter, userSampleDensity=userSamples, sourceFrequency=14_000)
        mcmc.run(kernel=kernel, chainCount=1, surfaceFunction=SymCosineSumSurfaceVectorized, burnInCount=burn_in_count, sampleCount=sample_count, scaleTrueScatter=True)
        mcmc.plotTrace()
        plt.savefig("results/" + kernel.lower() + " pymc trace.png")

    posterior_samples = np.loadtxt("results/" + kernel + ".csv", delimiter=",")
    
    # Convert each row into 3-element tuples for better multi
    posterior_samples_grouped = [list(zip(*row.reshape(-1, 3).T)) for row in posterior_samples]

    #Scale true scatter here to compare to parameters
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, RecLoc, 14_000, 700)
    truescatter = truescatter / factor

    # Creates a surface from each parameter set
    print("Creating posterior sample surfaces")
    surfs = []
    for _ in posterior_samples_grouped:
        hmm2 =  _.copy()
        surfs.append(SurfaceFunctionMulti(xsp,hmm2))

    print("Creating mean of parameters surface")
    mean = SurfaceFunctionMulti(xsp,np.mean(posterior_samples_grouped,axis=0))

    print("Creating mean of all surfaces")
    mean_surf = np.mean(surfs, axis=0)

    print("Plotting confidence interval")
    mins = []
    maxx = []
    for _ in range(500):
        vals = az.hdi(np.array(surfs).T[_],hdi_prob=0.68)
        mins.append(vals[0])
        maxx.append(vals[1])

    # Plot reconstruction
    print("Plotting reconstruction")
    b = np.random.choice(range(0,sample_count),400)
    plt.figure(figsize = (16,9))
    plt.grid()
    plt.fill_between(xsp,mins,maxx,color='grey',alpha=0.5,label='68% Credible interval')
    plt.plot(xsp,mean,label='Surface formed from the mean of the ' + kernel + ' model parameter')
    plt.plot(xsp,true,label='True surface')
    plt.plot(xsp,mean_surf, label='Surface formed from the mean of the generated ' + kernel + ' model functions')
    plt.legend(loc='upper right')

    plt.xlabel("x [m]")
    plt.ylabel("Surface elevation")
    plt.savefig("results/" + kernel + ".png")
    
    b = np.random.choice(range(0,sample_count),1000)

    plt.figure(figsize=(16,9))
    plt.grid()
    plt.plot(truescatter)

    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SurfaceFunctionMulti(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,14_000,0.02,np.pi/3,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor   

    for i in range(1000):
        lol = posterior_samples_grouped[b[i]].copy()
        plt.plot(generate_microphone_pressure(lol),'k',alpha=0.01)
    plt.xlabel("Microphone index")
    plt.ylabel("Response")
    plt.savefig("results/" + kernel + " traces.png")
    plt.show()

    import corner
    corner.corner(np.array(posterior_samples),bins=200,
              quantiles=[0.16, 0.5, 0.84],labels=[r"$\zeta_1$", r"$\zeta_2$", r"$\zeta_3$"],
              show_titles=True, title_fmt = ".4f")
    plt.savefig("results/" + kernel + " corner.png")
    plt.show()

    # Create the response array
    print("Generating responses")
    trace_count = 50
    trace_index = len(posterior_samples) - trace_count
    posterior_responses = []
    for i in range(trace_index, len(posterior_samples_grouped)):

        param = posterior_samples_grouped[i]
        posterior_responses.append(generate_microphone_pressure(param))

        if i % 10 == 0:
            index = i - trace_index
            print(round(100.0 * (float(index) / trace_count),2), "%")

    posterior_responses = np.array(posterior_responses)

    from sklearn.linear_model import LinearRegression

    print("Starting linear regression")
    lr = LinearRegression()
    lr.fit(posterior_samples[trace_index:], posterior_responses)
    print(lr.score(posterior_samples[trace_index:], posterior_responses))


if __name__ == "__main__":
    modelRun()