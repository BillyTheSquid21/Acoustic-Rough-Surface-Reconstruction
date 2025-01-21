import torch
import numpy as np
import matplotlib.pyplot as plt
from Directed2DVectorized import Directed2DVectorised

def generate_microphone_pressure(parameters,standard=False,sigma=0,factore=True,uSamples=700):
    '''
    Function that takes parameters of the surface and returns scattered
    acoustical pressure. Note that the source location, receiver array,
    frequency, angle, etc. is all handled inside the function. This will
    need to be took out for generalisation.
    '''
    def newFunction(x):
        return parameters[0]*np.cos((2*np.pi/parameters[1])*x +
                                    (2*np.pi/parameters[1])*parameters[2])

    KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,14_000,0.02,np.pi/3,'simp',userMinMax= [-1,1] ,userSamples=uSamples,absolute=False)
    scatter = KA_Object.Scatter(absolute=True,norm=False)

    if standard:
        return (np.array([scatter]).flatten() - np.mean(np.array([scatter]).flatten())) / np.std(np.array([scatter]).flatten())
    elif factore:
        return np.array([scatter]).flatten()/factor
    else:
        return np.array([scatter]).flatten()

# True params
p = [0.0015,0.05,0.00]

# True surface
def trueF(x):
    return p[0]*np.cos((2*np.pi/p[1])*x + p[2])

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

def crit1(position):
  '''
  Function that checks the physical validity of the sampled parameters
  '''
  def newFunction(x):
    return position[0]*np.cos((2*np.pi/position[1])*x +
                                position[2])

  KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,14_000,0.02,np.pi/3,'simp',userMinMax= [-1,1] ,userSamples = 700,absolute=False)
  b = KA_Object.surfaceChecker(True)
  if b == False:
    return False
  else:
    return True

#LVV 2019 data at 14kHz
comp = [0.01260586, 0.03210286, 0.06278351, 0.05601213, 0.03996005, 0.02276676,
    0.03520109, 0.05303403, 0.07947386, 0.06281864, 0.04293401, 0.0486769,
    0.04348036, 0.05091185, 0.05267429, 0.05519047, 0.0744659,  0.06930464,
    0.03456355, 0.02224533, 0.03089714, 0.02583821, 0.03186504, 0.02806474,
    0.05334908, 0.03035023, 0.0342647,  0.0377431,  0.03844866, 0.03600139,
    0.02590109, 0.01360588, 0.00829887, 0.00865361]

# Flat scatter

#Flat plate
def zero(x):
    return 0*x

flat = Directed2DVectorised(SourceLocation,RecLoc,zero,14_000,0.02,np.pi/3,'simp',userMinMax= [-5,5] ,userSamples = 7000,absolute=False)
flatscatter = flat.Scatter(absolute=True,norm=False)
factor=flatscatter.max().copy()

truescatter = comp / factor

import pyro
import pyro.distributions as dist
from scipy.stats import norm as normal

# Define the model
def model(training_data):
    # Priors for the three parameters
    param1 = pyro.sample("amp", dist.HalfNormal(scale=0.01))
    param2 = pyro.sample("wl", dist.HalfNormal(scale=0.08))
    param3 = pyro.sample("phase", dist.HalfNormal(scale=0.01))
    sigma  = pyro.sample("sigma", dist.Uniform(0.01, 5.0))

    # Ensure nonzero as could initially sample
    if param1.less_equal(torch.tensor(0.0)) or param2.less_equal(torch.tensor(0.0)):
        return
    
    # Keep samples in range specified
    if param1.greater(torch.tensor(0.21884 - 3*(343/14_000))) or param2.greater(torch.tensor(0.4)):
        return

    def surfaceFunction(x):
        return param1.item()*np.cos((2*np.pi/param2.item())*x + param3.item())

    # Determine the sampled parameter responses and compare to observed data
    KA_Object = Directed2DVectorised(SourceLocation,RecLoc,surfaceFunction,14_000,0.02,np.pi/3,'simp',userMinMax= [-1,1] ,userSamples=14_000,absolute=False)
    scatter = KA_Object.Scatter(absolute=True,norm=False)
    scatter = np.array([scatter]).flatten()/factor

    # Ensure physically valid (meets kirchoff criteria)
    if not KA_Object.surfaceChecker(True):
        return
    
    # Compute responses for training data
    predicted_response = torch.tensor(scatter)
    
    # Likelihood: Compare the predicted response to the observed training data
    with pyro.plate("data", training_data.shape[0]):
        pyro.sample("obs", dist.Normal(predicted_response, sigma), obs=training_data)

# True function over linspace
xsp = np.linspace(ReceiverLocationsX[0],ReceiverLocationsX[-1], 500)
true = trueF(xsp)

# Run MCMC iterations
from pyro.infer import MCMC, NUTS, RandomWalkKernel
from pyro.infer.mcmc.hmc import HMC

sample_count = 200_000
burn_in_count = 10_000
run_model = True
kernel = "hmc"
posterior_samples = np.array([])

if (run_model):
    print("Running MCMC model")
    if kernel == "rw":
        rw_kernel = RandomWalkKernel(model, init_step_size=0.01, target_accept_prob=0.15)
        mcmc = MCMC(rw_kernel, num_samples=sample_count, warmup_steps=burn_in_count)
        mcmc.run(torch.tensor(truescatter))
        posterior_samples = mcmc.get_samples()
        print(mcmc.summary())
    elif kernel == "nuts":
        nuts_kernel = NUTS(model, target_accept_prob=0.15, adapt_step_size=True, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=sample_count, warmup_steps=burn_in_count)
        mcmc.run(torch.tensor(truescatter))
        posterior_samples = mcmc.get_samples()
        print(mcmc.summary())
    elif kernel == "hmc":
        hmc_kernel = HMC(model, target_accept_prob=0.15, adapt_step_size=True, jit_compile=True)
        mcmc = MCMC(hmc_kernel, num_samples=sample_count, warmup_steps=burn_in_count)
        mcmc.run(torch.tensor(truescatter))
        posterior_samples = mcmc.get_samples()
        print(mcmc.summary())

    posterior_amps = posterior_samples['amp'].detach().numpy()
    posterior_wls  = posterior_samples['wl'].detach().numpy()
    posterior_phase= posterior_samples['phase'].detach().numpy()
    posterior_samples = np.array(list(zip(posterior_amps,posterior_wls,posterior_phase)))

    # Serialize/Deserialize
    np.savetxt(kernel + ".csv", posterior_samples, delimiter=",")
    print("MCMC Results saved!")

posterior_samples = np.loadtxt(kernel + ".csv", delimiter=",")

print_sample_check = False
if (print_sample_check):
    for i in range(len(posterior_samples)):
        param1 = posterior_samples[i][0]
        param2 = posterior_samples[i][1]
        param3 = posterior_samples[i][2]

        def surfaceFunction(x):
            return param1*np.cos((2*np.pi/param2)*x + param3)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,surfaceFunction,14_000,0.02,np.pi/3,'simp',userMinMax= [-1,1] ,userSamples=14_000,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        scatter = np.array([scatter]).flatten()/factor

        plt.plot(scatter)
        plt.plot(truescatter)
        plt.show()

        plt.plot(xsp, surfaceFunction(xsp))
        plt.plot(xsp, trueF(xsp))
        plt.show()

def surfaceFunction(x, p):
        return p[0]*np.cos((2*np.pi/p[1])*x + p[2])

# Creates a surface from each parameter set
print("Creating posterior sample surfaces")
surfs = []
for _ in posterior_samples:
    hmm2 =  _.copy()
    surfs.append(surfaceFunction(xsp,hmm2))

print("Creating mean of parameters surface")
mean = surfaceFunction(xsp,np.mean(posterior_samples,axis=0))

print("Creating mean of all surfaces")
mean_surf = np.mean(surfs, axis=0)

print("Plotting confidence interval")
import arviz as az
mins = []
maxx = []
for _ in range(500):
    print(round((_ / 500) * 100, 2), "%")
    vals = az.hdi(np.array(surfs).T[_],hdi_prob=0.68)
    mins.append(vals[0])
    maxx.append(vals[1])

# Plot reconstruction
print("Plotting reconstruction")
b = np.random.choice(range(0,sample_count),400)
plt.figure(figsize = (16,9))
plt.grid()
plt.fill_between(xsp,mins,maxx,color='grey',alpha=0.5,label='68% Credible interval')
plt.plot(xsp,mean,label='Surface formed from the mean of the' + kernel + 'model parameter')
plt.plot(xsp,true,label='True surface')
plt.plot(xsp,mean_surf, label='Surface formed from the mean of the generated ' + kernel + ' model functions')
plt.legend(loc='upper right')

plt.xlabel("x [m]")
plt.ylabel("Surface elevation")
plt.savefig(kernel + ".png")

# Plot traces
#a.plot_traces()

b = np.random.choice(range(0,sample_count),3000)

plt.figure(figsize=(16,9))
plt.grid()
plt.plot(truescatter)

for i in range(1000):
    lol = posterior_samples[b[i]].copy()
    plt.plot(generate_microphone_pressure(lol),'k',alpha=0.01)
plt.xlabel("Microphone index")
plt.ylabel("Response")
plt.savefig(kernel + " traces.png")
plt.show()