import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pytensor.tensor as pt
import pickle
from Directed2DVectorized import Directed2DVectorised
from Directed2DVectorized import Directed2DVectorisedSymbolic

def modelRun():
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

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,14_000,0.02,np.pi/3,'simp',userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)

        if standard:
            return (np.array([scatter]).flatten() - np.mean(np.array([scatter]).flatten())) / np.std(np.array([scatter]).flatten())
        elif factore:
            return np.array([scatter]).flatten()/factor
        else:
           return np.array([scatter]).flatten()

    np.random.seed(42)

    from pytensor.printing import Print

    with pm.Model() as model:
        # Priors for the three parameters
        param1 = pm.HalfNormal('amp', sigma=0.01)
        param2 = pm.HalfNormal('wl', sigma=0.08)
        param3 = pm.HalfNormal('phase', sigma=0.01)
        sigma = pm.Uniform('sigma', 0.01, 5.0)

        # Surface function with evaluated parameters (if you need it before sampling)
        def surfaceFunction(x):
            return param1 * np.cos((2 * np.pi / param2) * x + param3)

        # Check for physical validity and sample
        p1_constraint1 = param1 > 0
        potential = pm.Potential("p1_c1", pm.math.log(pm.math.switch(p1_constraint1, 1, 1e-10)))

        p2_constraint1 = param2 > 0
        potential = pm.Potential("p2_c1", pm.math.log(pm.math.switch(p2_constraint1, 1, 1e-10)))

        p1_constraint2 = param1 > (0.21884 - 3 * (343 / 14000))
        potential = pm.Potential("p1_c2", pm.math.log(pm.math.switch(p1_constraint2, 1, 1e-10)))

        p2_constraint2 = param2 > 0.4
        potential = pm.Potential("p2_c2", pm.math.log(pm.math.switch(p2_constraint2, 1, 1e-10)))

        # Add check for your surface function's validity
        KA_Object = Directed2DVectorisedSymbolic(
            SourceLocation,
            RecLoc,
            surfaceFunction,
            14000,
            0.02,
            np.pi / 3,
            'simp',
            userSamples=14000,
            absolute=False
        )
        scatter = KA_Object.Scatter(absolute=True, norm=False)
        scatter = scatter / factor

        surface_check = KA_Object.surfaceChecker(True)

        # Likelihood: Compare predicted response to observed data
        likelihood = pm.Normal('obs', mu=scatter, sigma=sigma, observed=truescatter)

    # True function over linspace
    xsp = np.linspace(ReceiverLocationsX[0],ReceiverLocationsX[-1], 500)
    true = trueF(xsp)

    sample_count = 100_000
    burn_in_count = 10_000
    run_model = True
    trace = []
    kernel = "NUTS"
    posterior_samples = []
    if run_model == True:

        if kernel == "metropolis-hastings":
            with model:
                # Define the Metropolis sampler
                step = pm.Metropolis(tune_interval=5, scale=0.01)

            with model:
                # Sample from the posterior
                trace = pm.sample(tune=burn_in_count, draws=sample_count, step=step, chains=4, return_inferencedata=True)
        elif kernel == "NUTS":
            with model:
                # Define the Metropolis sampler
                step = pm.NUTS(target_accept=0.15, step_scale=0.01)

            with model:
                # Sample from the posterior
                trace = pm.sample(tune=burn_in_count, draws=sample_count, step=step, chains=4, return_inferencedata=True)

        print(az.summary(trace))

        # Trace plot
        az.plot_trace(trace)

        # Flatten arrays
        posterior = trace.posterior
        posterior_amps = posterior['amp'].values.flatten()
        posterior_wls = posterior['wl'].values.flatten()
        posterior_phase = posterior['phase'].values.flatten()

        # Combine into a 2D array (rows = samples, columns = variables)
        posterior_samples = np.column_stack((posterior_amps, posterior_wls, posterior_phase))

        # Save as CSV
        np.savetxt(kernel + ".csv", posterior_samples, delimiter=",")
        print("MCMC Results saved!")

    posterior_samples = np.loadtxt(kernel + ".csv", delimiter=",")

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

    # Create the response array
    print("Generating responses")
    trace_count = 50
    trace_index = len(posterior_samples) - trace_count
    posterior_responses = []
    for i in range(trace_index, len(posterior_samples)):

        param = posterior_samples[i]

        def surfaceFunction(x):
            return param[0] * np.cos((2 * np.pi / param[1]) * x + param[2])
        
        # Add check for your surface function's validity
        KA_Object = Directed2DVectorised(
            SourceLocation,
            RecLoc,
            surfaceFunction,
            14000,
            0.02,
            np.pi / 3,
            'simp',
            userSamples=14000,
           absolute=False
        )
        scatter = KA_Object.Scatter(absolute=True, norm=False)
        scatter = np.array([scatter]).flatten() / factor
        posterior_responses.append(scatter)

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