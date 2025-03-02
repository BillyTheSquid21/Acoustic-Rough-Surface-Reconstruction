import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import pandas as pd
import pytensor.tensor as pt

from sklearn.model_selection import train_test_split

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

train_count = 500
params = np.array(AcousticParameterMCMC.LoadCSVData("results/examples/nuts-gpu-solver-100K_IT/GPU_Test.csv")[:train_count]).reshape(train_count, -1)
print("Loaded previous trace: ", params.shape)

factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, 14_000, 700)
def generate_microphone_pressure(parameters,uSamples=700):
    def newFunction(x):
        return SymCosineSurfaceM(x, parameters[0], parameters[1], parameters[2])

    KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,14_000,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
    scatter = KA_Object.Scatter(absolute=True,norm=False)
    return scatter/factor   

# Generating responses
print("Generating responses")
scatters = []
for i in range(train_count):
    scatters.append(generate_microphone_pressure(params[i]))
scatters = np.array(scatters)

training_data = np.concatenate((params, scatters), axis=1)
print("Combined params and scatter responses: ", training_data.shape)

# Setup training data
print("Setting up training data")

# Create pandas dataframe (currently hard coded to 3 param recovery! TODO: maybe make more generic)
# Split into test and training sets
columns = ['amp', 'wl', 'phase']
for i in range(0, scatters.shape[1]):
  columns.append('r' + str(i))

dataframe = pd.DataFrame(training_data,
                   columns=columns)

# X3 contains the receiver data
# Y3 contains the parameter data
X3_train, X3_test, Y3_train, Y3_test = train_test_split(
    dataframe[columns[3*cosine_count:]], dataframe[columns[:3*cosine_count]], test_size=0.25, random_state=11)

# Initialize weights for network
n_hidden_1 = scatters.shape[1] # Should be 34 for receiver count
n_hidden_2 = n_hidden_1

np.random.seed(42)
floatX = pytensor.config.floatX

init_1 = np.random.randn(X3_train.shape[1], n_hidden_1).astype(floatX)
init_2 = np.random.randn(n_hidden_1, n_hidden_2).astype(floatX)
init_out = np.random.randn(n_hidden_2,Y3_train.shape[1]).astype(floatX)

init_out_amp = np.random.randn(n_hidden_2, 1).astype(floatX)
init_out_wl = np.random.randn(n_hidden_2, 1).astype(floatX)
init_out_phase = np.random.randn(n_hidden_2, 1).astype(floatX)

# Initialize random biases in each layer
init_b_1 = np.random.randn(n_hidden_1).astype(floatX)
init_b_2 = np.random.randn(n_hidden_2).astype(floatX)
init_b_out = np.random.randn(cosine_count*3).astype(floatX)

# Initialize shared
model3_input = pytensor.shared(np.array(X3_train))
model3_amp_output = pytensor.shared(np.array(Y3_train['amp']))
model3_wl_output = pytensor.shared(np.array(Y3_train['wl']))
model3_phase_output = pytensor.shared(np.array(Y3_train['phase']))
model3_output = pytensor.shared(np.array(Y3_train))

with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', mu=0, sigma=1, shape=(model3_input.shape[1], n_hidden_1), initval=init_1)
    # Bias at 1st hidden layer
    bias_1 = pm.Normal('b_1', mu=0, sigma=1, shape=(n_hidden_1), initval=init_b_1)

    # Weights from 1st to 2nd hidden layer
    weights_1_2 = pm.Normal('w_1_2', mu=0, sigma=1, shape=(n_hidden_1, n_hidden_2), initval=init_2)
    # Bias at 2nd hidden layer
    bias_2 = pm.Normal('b_2', mu=0, sigma=1, shape=(n_hidden_2), initval=init_b_2)

    # Weights from hidden layer to output (3 outputs)
    weights_2_out = pm.Normal('w_2_out', mu=0, sigma=1, shape=(n_hidden_2, 3), initval=init_out)
    # Bias at output hidden layer
    bias_out = pm.Normal('b_out', mu=0, sigma=1, shape=(3), initval=init_b_out)


    # Build neural-network using tanh activation function
    act_1 = pm.math.tanh(pm.math.dot(model3_input, weights_in_1) + bias_1)
    act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + bias_2)
    #act_out = pm.math.dot(act_2, weights_2_out) + bias_out
    act_out_amp = pm.math.dot(act_2, weights_2_out[:,0]) + bias_out[0]
    act_out_wl = pm.math.dot(act_2, weights_2_out[:,1]) + bias_out[1]
    act_out_phase = pm.math.dot(act_2, weights_2_out[:,2]) + bias_out[2]

    sd_amp = pm.HalfNormal('sd_amp', sigma=1)
    out_amp = pm.Normal('amp', mu=act_out_amp, sigma=sd_amp, observed=model3_amp_output)

    sd_wl = pm.HalfNormal('sd_wl', sigma=1)
    out_wl = pm.Normal('wl', mu=act_out_wl, sigma=sd_wl, observed=model3_wl_output)

    sd_phase = pm.HalfNormal('sd_phase', sigma=1)
    out_phase = pm.Normal('phase', mu=act_out_phase, sigma=sd_phase, observed=model3_phase_output)

with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=50000, method=inference)
    nn_trace = approx.sample(draws=5000)

    plt.plot(approx.hist, label="old ADVI", alpha=0.3)
    plt.legend()
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    #print(az.summary(nn_trace))
    #az.plot_trace(nn_trace, combined=False)
    plt.show()

    #az.plot_forest(nn_trace)
    #plt.show()

# Replace shared variables with testing set
model3_input.set_value(np.array(X3_test))
model3_amp_output = pytensor.shared(np.array(Y3_test['amp']))
model3_freq_output = pytensor.shared(np.array(Y3_test['wl']))
model3_phase_output = pytensor.shared(np.array(Y3_test['phase']))
# Create posterior predictive samples
ppc_nn = pm.sample_posterior_predictive(nn_trace, model=neural_network)

pred_amp = ppc_nn['amp'].mean(axis=0)
pred_wl = ppc_nn['wl'].mean(axis=0)
pred_phase = ppc_nn['phase'].mean(axis=0)

print(pred_amp, " ", pred_wl, " ", pred_phase)