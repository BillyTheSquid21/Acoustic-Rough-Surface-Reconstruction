import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import pandas as pd
import pytensor.tensor as pt
from tqdm import tqdm

from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # Create the neural network model
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

    from src.AcousticParameterMCMC import AcousticParameterMCMC
    from src.Directed2DVectorized import Directed2DVectorised
    from src.SymbolicMath import SymCosineSurfaceM

    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, 14_000, 700)
    def generate_microphone_pressure(parameters,uSamples=700):
        def newFunction(x):
            return SymCosineSurfaceM(x, parameters[0], parameters[1], parameters[2])

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,14_000,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor

    p_count = 10_000
    train_count = p_count
    sample_count = 100_000 # Hard coded for now
    
    params = []
    scatters = []
    ampspace = np.linspace(0.0005, 0.01, p_count)
    for a in tqdm (range(p_count), desc="Generating param responses"):
        params.append((ampspace[a]))
        scatters.append(generate_microphone_pressure((ampspace[a], 0.05, 0.0)))

    params = np.array(params)
    scatters = np.array(scatters)

    training_data = np.column_stack((params, scatters))
    print("Combined params and scatter responses: ", training_data.shape)

    # Setup training data
    # Create pandas dataframe (currently hard coded to 3 param recovery! TODO: maybe make more generic)
    # Split into test and training sets
    columns = ['amp']
    for i in range(0, scatters.shape[1]):
        columns.append('r' + str(i))

    dataframe = pd.DataFrame(training_data,
                    columns=columns)

    # X3 contains the receiver data
    # Y3 contains the parameter data
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(
        dataframe[columns[1*cosine_count:]], dataframe[columns[:1*cosine_count]], test_size=0.25, random_state=11)

    print("Training params view ", Y3_train.shape, ":\n", Y3_train, "\n")
    print("Training scatter view ", X3_train.shape, ":\n", X3_train, "\n")
    print("Testing params view:", Y3_test.shape, "\n", Y3_test, "\n")
    print("Testing scatter view:", X3_test.shape, "\n", X3_test, "\n")

    # Initialize weights for network
    n_hidden_1 = scatters.shape[1] # Should be 34 for receiver count
    n_hidden_2 = n_hidden_1

    np.random.seed(42)
    floatX = pytensor.config.floatX

    init_1 = np.random.randn(X3_train.shape[1], n_hidden_1).astype(floatX)
    init_2 = np.random.randn(n_hidden_1, n_hidden_2).astype(floatX)
    init_out = np.random.randn(n_hidden_2,Y3_train.shape[1]).astype(floatX)

    init_out_amp = np.random.randn(n_hidden_2, 1).astype(floatX)

    # Initialize random biases in each layer
    init_b_1 = np.random.randn(n_hidden_1).astype(floatX)
    init_b_2 = np.random.randn(n_hidden_2).astype(floatX)
    init_b_out = np.random.randn(cosine_count*1).astype(floatX)

    amp_sd = 1.0
    print("Amp std: ", amp_sd)

    # Initialize shared
    model3_input = pytensor.shared(np.array(X3_train).astype(floatX))
    model3_output = pytensor.shared(np.array(Y3_train/amp_sd).astype(floatX))

    with pm.Model() as neural_network:
        # Input -> Layer 1
        weights_1 = pm.Normal('w_1', mu=0, sigma=1,
                                shape=(model3_input.shape[1], n_hidden_1),
                                initval=init_1)
        
        acts_1 = pm.math.tanh(pm.math.dot(model3_input, weights_1))

        # Layer 1 -> Layer 2
        weights_2 = pm.Normal('w_2', mu=0, sigma=1,
                                shape=(n_hidden_1, n_hidden_2),
                                initval=init_2)
        
        acts_2 = pm.math.tanh(pm.math.dot(acts_1, weights_2))

        # Layer 2 -> Output Layer
        weights_out = pm.Normal('w_out', mu=0, sigma=1,
                                    shape=(n_hidden_2, 1),
                                    initval=init_out)
        
        acts_out = pm.math.softmax(pm.math.dot(acts_2, weights_out)) # noqa

        # Define likelihood
        #out = pm.Multinomial('likelihood', n=1, p=acts_out,
        #                       observed=ann_output)
        out = pm.Normal('amp', sigma=0.02, mu=acts_out[0], observed=model3_output)

    run_model = True
    if run_model:
        with neural_network:

            #inference = pm.ADVI()
            #tracker = pm.callbacks.Tracker(
            #    mean=inference.approx.mean.eval,  # callable that returns mean
            #    std=inference.approx.std.eval,    # callable that returns std
            #)

            #approx = inference.fit(n=100_000, callbacks=[tracker])
            step = pm.NUTS(target_accept=0.9)
            nn_trace = pm.sample(tune=1_000, draws=5_000, step=step, chains=1, cores=1, nuts_sampler="numpyro")
            nn_trace.to_netcdf("results/nn_trace.nc")

            #fig = plt.figure(figsize=(16, 9))
            #mu_ax = fig.add_subplot(221)
            #std_ax = fig.add_subplot(222)
            #hist_ax = fig.add_subplot(212)
            #mu_ax.plot(tracker["mean"])
            #mu_ax.set_title("Mean track")
            #std_ax.plot(tracker["std"])
            #std_ax.set_title("Std track")
            #hist_ax.plot(inference.hist)
            #hist_ax.set_title("Negative ELBO track")
            #plt.savefig("results/nn_elbo.png")

            summary = az.summary(nn_trace)
            print(summary)
            with open("results/output.txt", "w") as txt:
                txt.write(summary.to_string())

    nn_trace = az.from_netcdf("results/nn_trace.nc")

    # Replace shared variables with testing set
    # Need to set as shapes change when test and training sets differ in shape
    model3_input.set_value(np.array(X3_test).astype(floatX))
    model3_output.set_value(np.array(Y3_test/amp_sd).astype(floatX))
    ppc_nn = pm.sample_posterior_predictive(nn_trace, var_names=["amp"], model=neural_network)["posterior_predictive"]

    from src.SymbolicMath import SymAngularMean
    amps = np.array(ppc_nn['amp']) * amp_sd
    amps_index = np.array(amps).squeeze()
    pred_amp = amps_index.mean(axis=0)

    plt.scatter(np.array(pred_amp), np.array(Y3_test))
    plt.show()
