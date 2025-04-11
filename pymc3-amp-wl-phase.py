import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
import pandas as pd
import pytensor.tensor as pt
import scienceplots
from tqdm import tqdm
from src.AcousticParameterBNN import BayesianNN

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    plt.style.use('science')
    plt.rcParams["font.family"] = "Bitstream Charter"

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
    a = 0.02
    frequency = 14_000

    from src.AcousticParameterMCMC import AcousticParameterMCMC
    from src.Directed2DVectorized import Directed2DVectorised
    from src.SymbolicMath import SymCosineSurface
    
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, frequency)
    def generate_microphone_pressure(parameters,uSamples=700,a=0.02):
        def newFunction(x):
            return SymCosineSurface(x,parameters).copy()

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,a,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor 
    
    from src.SymbolicMath import GetSpecularIndices
    indices = GetSpecularIndices(SourceLocation, SourceAngle, frequency, RecLoc)
    last_indice = indices[-1]
    for i in range(5):
        indices.append(last_indice + i)

    print("Indices used: ", indices)

    generate_data = False
    p_count = 25_000
    
    from src.SymbolicMath import SymRMS
    def noise_sigma(s, pc):
        '''
        Get scale of noise from rms average of the signal
        '''
        return (SymRMS(np.array(s)))*pc

    amp_bounds = [0.0, 0.01]
    wl_bounds = [0.04, 0.2]
    if generate_data:
        params = []
        scatters = []
        ampspace = np.random.uniform(amp_bounds[0], amp_bounds[1], p_count)
        wlspace = np.random.uniform(wl_bounds[0], wl_bounds[1], p_count)
        pc_noise = 0.2 # 20% noise level

        for i in range(p_count):
            params.append((ampspace[i], wlspace[i], 0.0))
        
        pbar = tqdm (range(len(params)), desc="Generating param responses with noise")
        for p in pbar:
            # Smoothness check
            def newFunction(x):
                return SymCosineSurface(x,params[p]).copy()

            an = a + np.random.normal(loc=0.0, scale=a*0.01) # Small 1% uncertainty to the aperture

            KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,a,SourceAngle,'simp',userMinMax=[-1,1],userSamples=700,absolute=False)
            if not KA_Object.surfaceChecker(True, True):
                # While the surface is not valid, keep rerolling until it is
                while not KA_Object.surfaceChecker(True, True):
                    params[p] = (np.random.uniform(amp_bounds[0], amp_bounds[1]), np.random.uniform(wl_bounds[0], wl_bounds[1]), 0.0)
                    KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,an,SourceAngle,'simp',userMinMax=[-1,1],userSamples=700,absolute=False)

            s = KA_Object.Scatter(absolute=True,norm=False) / factor
            s += np.random.normal(loc=0.0, scale=noise_sigma(s, pc_noise), size=(34,))
            s = np.abs(s)
            scatters.append(s[indices[0]:indices[-1]])

        params = np.array(params)
        scatters = np.array(scatters)

        plt.hist(params[:, 0], bins=50, alpha=0.5, label="amp")
        plt.legend()
        plt.show()

        plt.hist(params[:, 1], bins=50, alpha=0.5, label="wl")
        plt.legend()
        plt.show()

        np.savetxt("results/amp_wl_scatter_14KHz.csv", scatters, delimiter=",")
        np.savetxt("results/amp_wl_params_14KHz.csv", params, delimiter=",")

    params = np.loadtxt("results/amp_wl_params_14KHz.csv", delimiter=",")[:1000]
    scatters = np.loadtxt("results/amp_wl_scatter_14KHz.csv", delimiter=",")[:1000]

    training_data = np.column_stack((params[:,:2], scatters))
    print("Combined params and scatter responses: ", training_data.shape)

    # Setup training data
    # Split into test and training sets
    columns = ['amp','wl']
    for i in range(0, scatters.shape[1]):
        if i in indices:
            columns.append('r' + str(i))
    dataframe = pd.DataFrame(training_data,
                    columns=columns)

    # X3 contains the receiver data
    # Y3 contains the parameter data
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(
        dataframe[columns[2*cosine_count:]], dataframe[columns[:2*cosine_count]], test_size=0.2, random_state=11)

    print("Training params view ", Y3_train.shape, ":\n", Y3_train, "\n")
    print("Training scatter view ", X3_train.shape, ":\n", X3_train, "\n")
    print("Testing params view:", Y3_test.shape, "\n", Y3_test, "\n")
    print("Testing scatter view:", X3_test.shape, "\n", X3_test, "\n")

    # Linear regression first
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X3_train, Y3_train)
    print("Linear score: ", lr.score(X3_test, Y3_test))
    plt.scatter(Y3_train['amp'], Y3_train['amp'], label="True values")
    plt.scatter(Y3_train['amp'], lr.predict(X3_train)[:,0], color="red", label="Model predictions")
    plt.legend()
    plt.show()

    plt.scatter(Y3_train['wl'], Y3_train['wl'], label="True values")
    plt.scatter(Y3_train['wl'], lr.predict(X3_train)[:,1], color="red", label="Model predictions")
    plt.legend()
    plt.show()

    # Initialize and train model
    bnn = BayesianNN(n_hidden=indices[-1]*1.5)
    bnn.setName("amp_wl_bnn")
    #bnn.train(X3_train, Y3_train, sampleCount=25_000, burnInCount=5_000)
    params = np.array(bnn.predict(X3_test, Y3_test)['param']).squeeze()
    amps = np.array(params[:,:,0])
    wls = np.array(params[:,:,1])

    amps_index = np.array(amps).squeeze()
    wls_index = np.array(wls).squeeze()

    # Convert to NumPy arrays
    true_amp = np.array(Y3_test["amp"])
    true_wl = np.array(Y3_test["wl"])
    amps_sorted = np.array(amps_index)
    wls_sorted = np.array(wls_index)

    # Step 1: Sort indices
    sort_idx_amp = np.argsort(true_amp)
    sort_idx_wl = np.argsort(true_wl)

    # Step 2: Sort true test
    true_amp_sorted = true_amp[sort_idx_amp]
    true_wl_sorted = true_wl[sort_idx_wl]

    # Step 3: Sort index (reordering the posterior samples)
    amps_sorted = amps_index[:, sort_idx_amp]
    wls_sorted = wls_index[:, sort_idx_wl]

    # Step 4: Compute HDI and mean again after sorting
    amps_hdi_sorted = az.hdi(amps_sorted, hdi_prob=0.68).T
    wls_hdi_sorted = az.hdi(wls_sorted, hdi_prob=0.68).T

    from scipy.signal import savgol_filter
    window_length = 13  # Must be odd (adjust for smoothness)
    polyorder = 2  # Polynomial order for fitting

    # Amps first
    amp_lower_hdi_sorted, amp_upper_hdi_sorted = amps_hdi_sorted
    amp_lower_hdi_smooth = savgol_filter(amp_lower_hdi_sorted, window_length, polyorder)
    amp_upper_hdi_smooth = savgol_filter(amp_upper_hdi_sorted, window_length, polyorder)
    pred_amp_sorted = amps_sorted.mean(axis=0)

    lower_err = pred_amp_sorted - amp_lower_hdi_sorted
    upper_err = amp_upper_hdi_sorted - pred_amp_sorted

    # Step 5: Plot with correctly sorted data
    plt.figure(figsize=(16, 9))

    # Scatter plot of predicted vs true values
    plt.errorbar(
        true_amp_sorted, pred_amp_sorted, 
        yerr=[lower_err, upper_err], 
        fmt='o', label="Predicted with 68%% HDI", 
        capsize=4, capthick=1, alpha=0.7, markersize=5
    )

    # Scatter plot of true test values
    #plt.scatter(true_amp_sorted, pred_amp_sorted, label="amps")
    plt.scatter(true_amp_sorted, true_amp_sorted, color='orange', label="True test set")

    # Fill HDI region correctly
    plt.fill_between(true_amp_sorted, amp_lower_hdi_smooth, amp_upper_hdi_smooth, 
                    color='gray', alpha=0.3, label="68%% HDI")

    plt.legend()
    plt.xlabel("True Amplitude")
    plt.ylabel("Predicted Amplitude")
    plt.title("Predicted vs True Amplitudes with 68% HDI")
    plt.show()

    from sklearn.metrics import r2_score
    print("R2 score of amp in BNN: ", r2_score(true_amp_sorted, pred_amp_sorted))

    #WLs next
    wl_lower_hdi_sorted, wl_upper_hdi_sorted = wls_hdi_sorted
    wl_lower_hdi_smooth = savgol_filter(wl_lower_hdi_sorted, window_length, polyorder)
    wl_upper_hdi_smooth = savgol_filter(wl_upper_hdi_sorted, window_length, polyorder)
    pred_wl_sorted = wls_sorted.mean(axis=0)

    lower_err = pred_wl_sorted - wl_lower_hdi_sorted
    upper_err = wl_upper_hdi_sorted - pred_wl_sorted

    # Step 5: Plot with correctly sorted data
    plt.figure(figsize=(16, 9))

    # Scatter plot of predicted vs true values
    plt.errorbar(
        true_wl_sorted, pred_wl_sorted, 
        yerr=[lower_err, upper_err], 
        fmt='o', label="Predicted with 68%% HDI", 
        capsize=4, capthick=1, alpha=0.7, markersize=5
    )

    # Scatter plot of true test values
    #plt.scatter(true_amp_sorted, pred_amp_sorted, label="amps")
    plt.scatter(true_wl_sorted, true_wl_sorted, color='orange', label="True test set")

    # Fill HDI region correctly
    plt.fill_between(true_wl_sorted, wl_lower_hdi_smooth, wl_upper_hdi_smooth, 
                    color='gray', alpha=0.3, label="68%% HDI")

    plt.legend()
    plt.xlabel("True Wavelength")
    plt.ylabel("Predicted Wavelength")
    plt.title("Predicted vs True Wavelengths with 68% HDI")
    plt.show()

    print("R2 score of amp in BNN: ", r2_score(true_wl_sorted, pred_wl_sorted))

    print("R2 score of total BNN: ", r2_score((true_amp_sorted, true_wl_sorted), (pred_amp_sorted, pred_wl_sorted)))