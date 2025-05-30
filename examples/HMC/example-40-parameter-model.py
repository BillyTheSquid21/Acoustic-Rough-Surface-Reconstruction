import numpy as np
import matplotlib.pyplot as plt
import jax
import arviz as az
import corner
import scienceplots

# Ensure additional folders are seen
import sys
import os
sys.path.insert(1, os.getcwd())
from src.SymbolicMath import CosineSumSurface
from src.SymbolicMath import CosineSumSurfaceM

from src.Directed2DVectorized import Directed2DVectorised
from src.AcousticParameterMCMC import AcousticParameterMCMC

from src.SymbolicMath import SymCosineSumSurface
from src.SymbolicMath import CosineSumSurface as SurfaceFunctionMulti
from src.SymbolicMath import SymRandomSurface

def modelRun():
    plt.style.use('science')
    plt.rcParams["font.family"] = "Bitstream Charter"

    # Check jax backend
    print("jax device: ", jax.default_backend(), " ", jax.device_count())

    cosine_count = 20

    # Microphone array
    SourceLocation = np.array([0.0765, 0.20498])
    SourceAngle = 1.11701
    RecLoc = np.array([[0.14550000000000002, 0.2690491103511812],
    [0.16500000000000004, 0.26899860151841387],
    [0.18450000000000003, 0.2691680926856465],
    [0.20500000000000002, 0.2693616603229937],
    [0.22550000000000003, 0.26925522796034074],
    [0.24600000000000002, 0.26958879559768795],
    [0.2665, 0.2697923632350351],
    [0.28450000000000003, 0.26938573969709595],
    [0.30600000000000005, 0.26941338380455754],
    [0.325, 0.270290836736733],
    [0.34550000000000003, 0.2694644043740801],
    [0.367, 0.26960204848154173],
    [0.3865, 0.26944153964877443],
    [0.406, 0.27024103081600703],
    [0.42800000000000005, 0.26964071315852584],
    [0.4435, 0.27014389844530057],
    [0.4645, 0.2702695043177049],
    [0.48500000000000004, 0.27065307195505206],
    [0.5065, 0.2702607160625137],
    [0.5265, 0.2707022454648036],
    [0.5455000000000001, 0.2707396983969789],
    [0.5660000000000001, 0.2710032660343261],
    [0.587, 0.26974887190673047],
    [0.6045, 0.26955021013373415],
    [0.6245, 0.270091739536024],
    [0.6455, 0.2702873454084284],
    [0.6659999999999999, 0.2711609130457756],
    [0.6865000000000001, 0.27047448068312263],
    [0.7035, 0.27029378067506904],
    [0.7255, 0.271373463017588],
    [0.744, 0.27146887771470607],
    [0.7655000000000001, 0.27174652182216774],
    [0.7855000000000001, 0.27207805122445766],
    [0.8049999999999999, 0.27159754239169026]])

    #LVV 2019 data at 14kHz
    comp = [0.01830538, 0.05386456, 0.04145832, 0.0237838 , 0.03931708,
       0.06042542, 0.0790508 , 0.07607468, 0.05835909, 0.06657535,
       0.03775648, 0.02390032, 0.02662371, 0.04746789, 0.0525121 ,
       0.04585534, 0.06646751, 0.05084494, 0.04207696, 0.02215824,
       0.01780381, 0.01732509, 0.00488723, 0.0086965 , 0.01590828,
       0.01131624, 0.00866224, 0.01029487, 0.00647845, 0.00487716,
       0.00524228, 0.00330431, 0.00254167, 0.00423263]
    
    sourceFreq = 18_650

    # Real data
    truescatter = comp

    waves = np.array([0.6       , 0.3       , 0.2       , 0.15      , 0.12      ,
        0.1       , 0.08571429, 0.075     , 0.06666667, 0.06      ,
        0.05454545, 0.05      , 0.04615385, 0.04285714, 0.04      ,
        0.0375    , 0.03529412, 0.03333333, 0.03157895, 0.03      ])

    amps = np.array([-1.85943576e-04,  6.11424589e-04,  4.22465782e-04,  9.94584634e-05,
            1.82869367e-04,  4.62403358e-04, -3.92636879e-05,  5.49616501e-04,
        -3.03324524e-04,  2.22022482e-04,  3.37438019e-04, -1.44359487e-04,
        -5.32544188e-04, -2.11110769e-04,  3.51746289e-04,  1.17038060e-04,
        -4.98759904e-05,  1.56105878e-04,  1.36063225e-04,  4.87574794e-05])

    phases = np.array([3.42407995, 6.21218324, 2.81775226, 5.82009315, 0.89579862,
        5.6362695 , 5.35005385, 2.08078952, 4.86461855, 4.98374707,
        1.70475039, 0.29060533, 2.61582606, 5.19381672, 5.53538759,
        0.87227685, 5.32987222, 1.78663768, 1.66342492, 1.05875783])

    amp2 = []
    phases2 = []
    for i in range(len(amps)):
        if amps[i] < 0:
            amp2.append(np.abs(amps[i]))
            if phases[i]+np.pi > 2*np.pi:
                phases2.append(phases[i]+np.pi-2*np.pi)
            else:
                phases2.append((phases[i]+np.pi))
        else:
            amp2.append(amps[i])
            phases2.append(phases[i])

    amps = amp2.copy()
    phases = phases2.copy()
    phases = np.array(phases) / (2*np.pi) # Phase is in units of radians / 2*pi

    amps_scale = 0.001

    # Proposal stds
    amp_stds = np.linspace(1.0*amps_scale, 0.5*amps_scale, cosine_count)

    sample_count = 20_000
    burn_in_count = 20_000
    run_model = False
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
        
        mcmc.setAmplitudeProposal(amp_stds)
        mcmc.setWavelengths(waves)
        
        # Run the model MCMC sampler
        mcmc.run(kernel=kernel, 
                 chainCount=1, 
                 surfaceFunction=SymCosineSumSurface, 
                 burnInCount=burn_in_count, 
                 sampleCount=sample_count, 
                 scaleTrueScatter=True,
                 targetAccRate=0.9,
                 #truncateInds=True
                 )
        
        mcmc.plotTrace()
    
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
    mean = CosineSumSurface(x,hmm2)
    true = CosineSumSurfaceM(x,amps,waves,phases).copy()

    # Creating mean of all surfaces
    print("Creating individual surfaces")
    surfs = []
    x = np.linspace(0,0.6,500)
    for _ in posterior_samples_grouped:
        surfs.append(CosineSumSurface(x,_))

    print("Taking mean of all combined surfaces")
    mean_surf = np.mean(surfs,axis=0)

    print("Creating HDI interval")
    mins = []
    maxx = []
    for _ in range(500):
        vals = az.hdi(np.array(surfs).T[_],hdi_prob=0.95)
        mins.append(vals[0])
        maxx.append(vals[1])

    plt.figure(figsize = (16,6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)

    # Plot 68% and 95%
    plt.fill_between(x,mins,maxx,color='#ffffbf',label=r'Credible interval (95\%)')
    plt.plot(x, mins, color='black', linewidth=0.75)  # Lower boundary
    plt.plot(x, maxx, color='black', linewidth=0.75)  # Upper boundary

    mins = []
    maxx = []
    for _ in range(500):
        vals = az.hdi(np.array(surfs).T[_],hdi_prob=0.68)
        mins.append(vals[0])
        maxx.append(vals[1])

    plt.fill_between(x,mins,maxx,color='#f1f155',label=r'Credible interval (68\%)')

    plt.plot(x,mean_surf, color="#2c7bb6", linestyle="dashed", label=r'Surface formed from the mean of the model surfaces')
    plt.plot(x,mean, color="#2c7bb6",label=r'Surface formed from the mean of the model parameters')
    plt.plot(x,true, color="#d7191c",label=r'True surface')
    plt.xlabel("x (m)")
    plt.ylabel("Surface elevation (m)")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("Surface elevation")
    plt.savefig("results/" + kernel + " reconstruction.png")

    print("Creating response plot")
    choice_count = 300
    b = np.random.choice(range(sample_count),choice_count)

    # Plot the response
    plt.figure(figsize=(16,6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)
    plt.plot(truescatter, color="#d7191c")

    # Generate the scattered response
    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SurfaceFunctionMulti(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,sourceFreq,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor   
    
    for i in range(len(b)):
        plt.plot(generate_microphone_pressure(posterior_samples_grouped[b[i]]), color="black", alpha=0.04)
    
    plt.xlabel("Microphone index")
    plt.ylabel("Normalized Response")
    plt.savefig("results/" + kernel + " traces.png")
    plt.show()

if __name__ == "__main__":
    modelRun()