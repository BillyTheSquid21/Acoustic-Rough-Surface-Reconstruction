import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
import pandas as pd
import pytensor.tensor as pt
from tqdm import tqdm
from src.AcousticParameterBNN import BayesianNN

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(42)

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

    frequency = 14_000

    from src.AcousticParameterMCMC import AcousticParameterMCMC
    from src.Directed2DVectorized import Directed2DVectorised
    from src.SymbolicMath import SymCosineSurface
    
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, frequency)
    def generate_microphone_pressure(parameters,uSamples=700):
        def newFunction(x):
            return SymCosineSurface(x,parameters).copy()

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor 
    
    def get_spec_point(sourceloc, recloc):
        '''
        Zero index is source, One index is reciever
        '''
        return (recloc[0]*sourceloc[1])/(sourceloc[1]+recloc[1])
    
    # Get point where directivity intersects
    specpoint = SourceLocation[0] + SourceLocation[1]*np.tan((np.pi/2)-SourceAngle)
    specspread = 0.1 # 10 cm each side

    indices = []
    for i in range(len(RecLoc)):
        rspec = get_spec_point(SourceLocation, RecLoc[i])
        if rspec > specpoint - specspread and rspec < specpoint + specspread:
            indices.append(i)

    generate_data = True
    p_count = 250
    train_count = p_count
    
    if generate_data:
        params = []
        scatters = []
        ampspace = np.random.uniform(0.0, 0.01, p_count)
        for i in range(p_count):
            params.append((ampspace[i], 0.05, 0.0))
            
        params = np.array(params)
        np.savetxt("results/amp_only_params_14KHz.csv", params, delimiter=",")
            
        for p in tqdm (range(int(train_count)), desc="Generating param responses with noise"):
            s = generate_microphone_pressure(params[p])
            s += np.random.normal(loc=0.0, scale=0.05, size=(34,))
            s = np.abs(s)
            scatters.append(s[indices[0]:indices[-1]])

        params = np.array(params)
        scatters = np.array(scatters)

        plt.hist(params[:, 0], bins=50, alpha=0.5, label="amp")
        plt.legend()
        plt.show()
        np.savetxt("results/amp_only_scatter_14KHz.csv", scatters, delimiter=",")

    params = np.loadtxt("results/amp_only_params_14KHz.csv", delimiter=",")
    scatters = np.loadtxt("results/amp_only_scatter_14KHz.csv", delimiter=",")
    training_data = np.column_stack((params[:,:1], scatters))
    print("Combined params and scatter responses: ", training_data.shape)

    # Setup training data
    # Create pandas dataframe (currently hard coded to 2 param recovery! TODO: maybe make more generic)
    # Split into test and training sets
    columns = ['amp']
    for i in range(0, scatters.shape[1]):
        if i in indices:
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

    # Linear regression first
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X3_train, Y3_train['amp'])
    print("Linear score: ", lr.score(X3_test, Y3_test['amp']))
    plt.scatter(Y3_train['amp'], Y3_train['amp'], label="True values")
    plt.scatter(Y3_train['amp'], lr.predict(X3_train), color="red", label="Model predictions")
    plt.legend()
    plt.show()

    # Initialize and train model
    bnn = BayesianNN(n_hidden=40)
    bnn.setName("amp_only_bnn")
    bnn.train(X3_train, Y3_train, sampleCount=5000, burnInCount=2000)
    amps = np.array(bnn.predict(X3_test, Y3_test)['param'])

    amps_index = np.array(amps).squeeze()
    pred_amp = amps_index.mean(axis=0)

    plt.figure(figsize=(16, 9))
    plt.scatter(np.array(Y3_test["amp"]), np.array(pred_amp), label="Predicted vs test")
    plt.scatter(np.array(Y3_test["amp"]), np.array(Y3_test["amp"]), label="True test set")
    plt.legend()
    plt.show()

    from sklearn.metrics import r2_score
    print("R2 score of BNN: ", r2_score(np.array(Y3_test["amp"]), pred_amp))