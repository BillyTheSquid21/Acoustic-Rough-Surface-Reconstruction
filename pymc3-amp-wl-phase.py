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
        # Load the trace data
    trace = az.from_netcdf("results/amp_wl_phase_bnn.nc")

    # Initialize a color palette for the forest plot
    sns.set(style="whitegrid")

    # You can control the number of top weights to display here
    top_n = 30  # Change this value to control how many top weights you want to plot
    palette = sns.color_palette("Blues_d", top_n)  # Choose a color palette for the top 20 parameters

    # Rename dictionary if needed
    rename_dict = {"W5": "W4", "b5": "b4"}

    # Iterate over each variable in the posterior
    for var in trace.posterior.data_vars:
        print(f"Plotting {var} with padding...")

        # Get the posterior values for this variable
        values = trace.posterior[var].values

        # If the variable is 1D (like biases), handle it differently
        if values.ndim == 3:  # For 1D variables (e.g., biases: chain, draw, bias_dim_0)
            mean_abs = np.mean(np.abs(values), axis=(0, 1))  # Mean of absolute values across chains and draws
            top_indices = np.argsort(mean_abs)[-top_n:]  # Get indices of top N
            
            var_names = [f"{var}[{i}]" for i in top_indices]  # 1D variable names (e.g., 'b1[0]', 'b1[1]')
            top_coords = [(i,) for i in top_indices]  # For 1D variables, just use the index in the 1D array

            print(f"Top {top_n} {var} indices: {top_indices}")
            
        elif values.ndim == 4:  # For 2D weight matrices (e.g., chain, draw, row, col)
            mean_abs = np.mean(np.abs(values), axis=(0, 1))
            flat_means = mean_abs.reshape(-1)
            top_indices = np.argsort(flat_means)[-top_n:]  # Get indices of top N
            
            rows, cols = mean_abs.shape
            top_coords = [(i // cols, i % cols) for i in top_indices]  # Get 2D coordinates
            var_names = [f"{var}[{i // cols}, {i % cols}]" for i in top_indices]  # 2D variable names

            print(f"Top {top_n} {var} coordinates: {top_coords}")
            
        # Extract mean and credible intervals for each of the top N variables
        means = []
        lower_68 = []
        upper_68 = []
        lower_95 = []
        upper_95 = []
        
        for coord in top_coords:
            if len(coord) == 1:  # For 1D variables (biases)
                row = coord[0]
                param_values = values[:, :, row]  # Get values for this bias
            else:  # For 2D variables (weights)
                row, col = coord
                param_values = values[:, :, row, col]  # Get values for this weight

            # Calculate the mean and the credible intervals (68% and 95%)
            mean = np.mean(param_values)
            ci_lower_68 = np.percentile(param_values, 16)
            ci_upper_68 = np.percentile(param_values, 84)
            ci_lower_95 = np.percentile(param_values, 2.5)
            ci_upper_95 = np.percentile(param_values, 97.5)
            
            means.append(mean)
            lower_68.append(ci_lower_68)
            upper_68.append(ci_upper_68)
            lower_95.append(ci_lower_95)
            upper_95.append(ci_upper_95)

        # Adjust figsize for the top N parameters
        height = max(4, top_n * 0.2)  # Adjust height based on top N
        var_to_plot = rename_dict.get(var, var)

        # Create the forest plot manually using Seaborn palette
        plt.figure(figsize=(8, height))
        
        # Set the palette (it will be reused for each variable)
        palette = sns.color_palette("dark:#5A9_r", n_colors=top_n)  # Choose a palette like "coolwarm"
        
        for i, var_name in enumerate(var_names):
            # Plot the 95% credible interval with a lighter color
            plt.plot([lower_95[i], upper_95[i]], [i, i], color=palette[i], lw=2, label=f"95% CI" if i == 0 else "")  # 95% CI
            # Plot the 68% credible interval with a wider line (overlaid on the 95% CI)
            plt.plot([lower_68[i], upper_68[i]], [i, i], color=palette[i], lw=4, label=f"68% CI" if i == 0 else "")  # 68% CI
            # Plot the mean points with colors from the palette
            plt.scatter(means[i], i, color=palette[i], s=50, zorder=5)  # Mean points

        # Set the plot's labels
        plt.yticks(np.arange(len(var_names)), var_names, fontsize=10)
        plt.xlabel('Parameter Value')
        plt.title(f"Forest plot for top {top_n} {var_to_plot} weights with 68% and 95% CIs", fontsize=14)

        # Add legend
        plt.legend()

        # Tight layout to avoid overlap
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"results/p3forest_top{top_n}_{var_to_plot}.png", bbox_inches='tight')
        plt.close()

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
    for i in range(1,3):
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
    phase_bounds = [-0.5, 0.5]
    if generate_data:
        params = []
        scatters = []
        ampspace = np.random.uniform(amp_bounds[0], amp_bounds[1], p_count)
        wlspace = np.random.uniform(wl_bounds[0], wl_bounds[1], p_count)
        phasespace = np.random.uniform(phase_bounds[0], phase_bounds[1], p_count)
        pc_noise = 0.2 # 20% noise level

        for i in range(p_count):
            params.append((ampspace[i], wlspace[i], phasespace[i]))
        
        pbar = tqdm (range(len(params)), desc="Generating param responses with noise")
        for p in pbar:
            def newFunction(x):
                tp = params[p]
                tp = (tp[0],tp[1],tp[2])
                return SymCosineSurface(x,tp)

            an = a + np.random.normal(loc=0.0, scale=a*0.01) # Small 1% uncertainty to the aperture

            KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,an,SourceAngle,'simp',userMinMax=[-1,1],userSamples=700,absolute=False)
            if not KA_Object.surfaceChecker(True, True):
                # While the surface is not valid, keep rerolling until it is
                while not KA_Object.surfaceChecker(True, True):
                    params[p] = (np.random.uniform(amp_bounds[0], amp_bounds[1]), np.random.uniform(wl_bounds[0], wl_bounds[1]), np.random.uniform(phase_bounds[0], phase_bounds[1]))
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

        plt.hist(params[:, 2], bins=50, alpha=0.5, label="phase")
        plt.legend()
        plt.show()

        np.savetxt("results/amp_wl_phase_scatter_14KHz.csv", scatters, delimiter=",")
        np.savetxt("results/amp_wl_phase_params_14KHz.csv", params, delimiter=",")

    params = np.loadtxt("results/amp_wl_phase_params_14KHz.csv", delimiter=",")[:1000]
    scatters = np.loadtxt("results/amp_wl_phase_scatter_14KHz.csv", delimiter=",")[:1000]

    training_data = np.column_stack((params[:,:], scatters))
    print("Combined params and scatter responses: ", training_data.shape)

    # Setup training data
    # Split into test and training sets
    columns = ['amp','wl','phase']
    for i in range(0, scatters.shape[1]):
        if i in indices:
            columns.append('r' + str(i))
    dataframe = pd.DataFrame(training_data,
                    columns=columns)

    # X3 contains the receiver data
    # Y3 contains the parameter data
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(
        dataframe[columns[3*cosine_count:]], dataframe[columns[:3*cosine_count]], test_size=0.2, random_state=11)

    print("Training params view ", Y3_train.shape, ":\n", Y3_train, "\n")
    print("Training scatter view ", X3_train.shape, ":\n", X3_train, "\n")
    print("Testing params view:", Y3_test.shape, "\n", Y3_test, "\n")
    print("Testing scatter view:", X3_test.shape, "\n", X3_test, "\n")

    # Linear regression first
    plt.rcParams.update({'font.size': 18})
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

    plt.scatter(Y3_train['phase'], Y3_train['phase'], label="True values")
    plt.scatter(Y3_train['phase'], lr.predict(X3_train)[:,2], color="red", label="Model predictions")
    plt.legend()
    plt.show()

    # Initialize and train model
    bnn = BayesianNN(n_hidden=indices[-1]*1.5)
    bnn.setName("amp_wl_phase_bnn")
    # Set the penalty mask so phase is allowed to be negative
    #bnn.train(X3_train, Y3_train, sampleCount=20_000, burnInCount=5000, penalty_mask=pt.as_tensor_variable((1.0,1.0,0.0)))
    predict = bnn.predict(X3_test, Y3_test, penalty_mask=pt.as_tensor_variable((1.0,1.0,0.0)))
    params = np.array(predict['param']).squeeze()
    amps = np.array(params[:,:,0])
    wls = np.array(params[:,:,1])
    phases = np.array(params[:,:,2])*(2.0*np.pi)

    amps_index = np.array(amps).squeeze()
    wls_index = np.array(wls).squeeze()
    phases_index = np.array(phases).squeeze()

    # Convert to NumPy arrays
    true_amp = np.array(Y3_test["amp"])
    true_wl = np.array(Y3_test["wl"])
    true_phase = np.array(Y3_test["phase"])*(2.0*np.pi)
    amps_sorted = np.array(amps_index)
    wls_sorted = np.array(wls_index)
    phases_sorted = np.array(phases_index)

    # Step 1: Sort indices
    sort_idx_amp = np.argsort(true_amp)
    sort_idx_wl = np.argsort(true_wl)
    sort_idx_phase = np.argsort(true_phase)

    # Step 2: Sort true test
    true_amp_sorted = true_amp[sort_idx_amp]
    true_wl_sorted = true_wl[sort_idx_wl]
    true_phase_sorted = true_phase[sort_idx_phase]

    # Step 3: Sort index (reordering the posterior samples)
    amps_sorted = amps_index[:, sort_idx_amp]
    wls_sorted = wls_index[:, sort_idx_wl]
    phases_sorted = phases_index[:, sort_idx_phase]

    # Step 4: Compute HDI and mean again after sorting
    amps_hdi_sorted = az.hdi(amps_sorted, hdi_prob=0.68).T
    wls_hdi_sorted = az.hdi(wls_sorted, hdi_prob=0.68).T
    phases_hdi_sorted = az.hdi(phases_sorted, hdi_prob=0.68, circular=True).T

    from scipy.signal import savgol_filter
    window_length = 101  # Must be odd (adjust for smoothness)
    polyorder = 4  # Polynomial order for fitting

    # Amps first
    amp_lower_hdi_sorted, amp_upper_hdi_sorted = amps_hdi_sorted
    amp_lower_hdi_smooth = savgol_filter(amp_lower_hdi_sorted, window_length, polyorder)
    amp_upper_hdi_smooth = savgol_filter(amp_upper_hdi_sorted, window_length, polyorder)
    pred_amp_sorted = amps_sorted.mean(axis=0)

    lower_err = pred_amp_sorted - amp_lower_hdi_sorted
    upper_err = amp_upper_hdi_sorted - pred_amp_sorted

    # Step 5: Plot with correctly sorted data
    plt.figure(figsize=(16, 6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)

    # Scatter plot of predicted vs true values
    amp_inds_sub = np.random.choice(sort_idx_amp, size=250)
    tas_sub = true_amp_sorted[amp_inds_sub]
    pas_sub = pred_amp_sorted[amp_inds_sub]
    le_sub = lower_err[amp_inds_sub]
    ue_sub = upper_err[amp_inds_sub]
    plt.errorbar(
        tas_sub*100.0, pas_sub*100.0, 
        yerr=[le_sub*100.0, ue_sub*100.0], 
        fmt='o', label="Predicted with 68\% HDI", 
        capsize=4, capthick=1, alpha=0.7, markersize=5, color="#2c7bb6"
    )

    # Scatter plot of true test values
    #plt.scatter(true_amp_sorted, pred_amp_sorted, label="amps")
    plt.plot(true_amp_sorted*100.0, true_amp_sorted*100.0, color='#d7191c', label="True test set")

    # Fill HDI region correctly
    plt.fill_between(true_amp_sorted*100.0, amp_lower_hdi_smooth*100.0, amp_upper_hdi_smooth*100.0, 
                    color='#ffffbf', label="68\% HDI")
    
    plt.plot(true_amp_sorted*100.0, amp_lower_hdi_smooth*100.0, color='black', linewidth=0.75)  # Lower boundary
    plt.plot(true_amp_sorted*100.0, amp_upper_hdi_smooth*100.0, color='black', linewidth=0.75)  # Upper boundary

    plt.legend()
    plt.xlabel("True Amplitude (cm)")
    plt.ylabel("Predicted Amplitude (cm)")
    #plt.title("Predicted vs True Amplitudes with 68\% HDI")
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
    plt.figure(figsize=(16, 6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)

    # Scatter plot of predicted vs true values
    wl_inds_sub = np.random.choice(sort_idx_wl, size=250)
    tws_sub = true_wl_sorted[wl_inds_sub]
    pws_sub = pred_wl_sorted[wl_inds_sub]
    lew_sub = lower_err[wl_inds_sub]
    uew_sub = upper_err[wl_inds_sub]
    plt.errorbar(
        tws_sub*100.0, pws_sub*100.0, 
        yerr=[lew_sub*100.0, uew_sub*100.0], 
        fmt='o', label="Predicted with 68\% HDI", 
        capsize=4, capthick=1, alpha=0.7, markersize=5, color="#2c7bb6"
    )

    # Scatter plot of true test values
    #plt.scatter(true_amp_sorted, pred_amp_sorted, label="amps")
    plt.plot(true_wl_sorted*100.0, true_wl_sorted*100.0, color='#d7191c', label="True test set")

    # Fill HDI region correctly
    plt.fill_between(true_wl_sorted*100.0, wl_lower_hdi_smooth*100.0, wl_upper_hdi_smooth*100.0, 
                    color='#ffffbf', label="68\% HDI")
    
    plt.plot(true_wl_sorted*100.0, wl_lower_hdi_smooth*100.0, color='black', linewidth=0.75)  # Lower boundary
    plt.plot(true_wl_sorted*100.0, wl_upper_hdi_smooth*100.0, color='black', linewidth=0.75)  # Upper boundary

    plt.legend()
    plt.xlabel("True Wavelength (cm)")
    plt.ylabel("Predicted Wavelength (cm)")
    plt.show()

    print("R2 score of wl in BNN: ", r2_score(true_wl_sorted, pred_wl_sorted))

    #Phases next
    from src.SymbolicMath import SymAngularMean
    phase_lower_hdi_sorted, phase_upper_hdi_sorted = phases_hdi_sorted
    phase_lower_hdi_smooth = savgol_filter(phase_lower_hdi_sorted, window_length, polyorder)
    phase_upper_hdi_smooth = savgol_filter(phase_upper_hdi_sorted, window_length, polyorder)
    pred_phase_sorted = np.apply_along_axis(SymAngularMean, arr=phases_sorted.copy()/(2.0*np.pi), axis=0)*2.0*np.pi

    # Ensure phases are in radians for polar plot
    true_phase_sorted_rad = true_phase_sorted
    pred_phase_sorted_rad = pred_phase_sorted
    lower_err_rad = np.arccos(np.cos(pred_phase_sorted_rad - phase_lower_hdi_sorted))
    upper_err_rad = np.arccos(np.cos(phase_upper_hdi_sorted - pred_phase_sorted_rad))
    
    phase_inds_sub = np.random.choice(sort_idx_phase, size=250)
    tpsr_sub = true_phase_sorted_rad[phase_inds_sub]
    ppsr_sub = pred_phase_sorted_rad[phase_inds_sub]
    ler_sub = lower_err_rad[phase_inds_sub]
    uer_sub = upper_err_rad[phase_inds_sub]

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')
    plt.thetagrids()

    # Plot predicted vs. true phase with error bars
    ax.errorbar(
        tpsr_sub, ppsr_sub,
        yerr=[ler_sub, uer_sub],
        fmt='o', label="Predicted with 68\% HDI",
        capsize=4, capthick=1, alpha=0.7, markersize=5, color="#2c7bb6"
    )

    # Reference line for true values
    ax.plot(true_phase_sorted_rad, true_phase_sorted_rad, 'o', color='#d7191c', label="True test set")

    # Fill HDI region
    ax.fill_between(
        true_phase_sorted_rad,
        phase_lower_hdi_smooth,
        phase_upper_hdi_smooth,
        color='#ffffbf', label="68\% HDI"
    )
    plt.plot(true_phase_sorted_rad, phase_lower_hdi_smooth, color='black', linewidth=0.75)  # Lower boundary
    plt.plot(true_phase_sorted_rad, phase_upper_hdi_smooth, color='black', linewidth=0.75)  # Upper boundary

    # ---- Set angular (theta) ticks from -2π to +2π ----
    theta_ticks = np.linspace(-2*np.pi, 2*np.pi, 9)  # [-2π, ..., 2π]
    theta_labels = [
        r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$',
        r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'
    ]
    # Wrap values into [0, 2π) range for polar coordinates
    wrapped_ticks = np.mod(theta_ticks, 2*np.pi)
    ax.set_xticks(wrapped_ticks)
    ax.set_xticklabels(theta_labels)

    # ---- Set radial (r) limits and ticks from -2π to 2π ----
    ax.set_rlim(-2*np.pi, 2*np.pi)
    r_ticks = np.linspace(-2*np.pi, 2*np.pi, 9)
    r_labels = [
        r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$',
        r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'
    ]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels(r_labels)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                  labels=['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', 
                          r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'])

    # Add a custom label manually near the radial ticks
    ax.text(
        np.deg2rad(5),  # angle = 90 degrees in radians (i.e., left side of plot)
        ax.get_rmin() + 8.0,  # place it just outside the largest tick
        "Predicted Offset",  # your label text
        ha='left', va='center',
        rotation=0  # optional: vertical text
    )

    # Title and labels
    plt.xlabel("True Offset")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1,-0.1))
    plt.tight_layout()
    plt.show()

    plt.hist(pred_phase_sorted_rad, bins=50, alpha=0.5, label="phase")
    plt.legend()
    plt.show()

    from src.SymbolicMath import SymAngularMean
    def circular_r2_score(true, pred):
        num = (np.arccos(np.cos(true-pred))**2).sum()
        den = (np.arccos(np.cos(true-SymAngularMean(true)))**2).sum()
        return 1.0 - ((num)/(den))

    print("R2 score of phase in BNN: ", circular_r2_score(true_phase_sorted, pred_phase_sorted))

    print("R2 score of total BNN: ", r2_score((true_amp_sorted, true_wl_sorted, true_phase_sorted), (pred_amp_sorted, pred_wl_sorted, pred_phase_sorted)))

    # Reconstruct true scatter
    plt.rcParams.update({'font.size': 18})
    from src.SymbolicMath import SymCosineSumSurface
    from src.SymbolicMath import SymCosineSumSurface as SurfaceFunctionMulti
    comp = [0.01260586, 0.03210286, 0.06278351, 0.05601213, 0.03996005, 0.02276676,
        0.03520109, 0.05303403, 0.07947386, 0.06281864, 0.04293401, 0.0486769,
        0.04348036, 0.05091185, 0.05267429, 0.05519047, 0.0744659,  0.06930464,
        0.03456355, 0.02224533, 0.03089714, 0.02583821, 0.03186504, 0.02806474,
        0.05334908, 0.03035023, 0.0342647,  0.0377431,  0.03844866, 0.03600139,
        0.02590109, 0.01360588, 0.00829887, 0.00865361]
    
    trueScatter = comp/factor
    p = (0.0015, 0.05, 0.0)

    # Creating mean surfaces
    print("Creating mean parameters surface")
    x = np.linspace(0,0.6,500)

    # Get predicted values
    X3_test = np.array([trueScatter[indices[0]:indices[-1]]])
    X3_test = pd.DataFrame(X3_test, columns=columns[3*cosine_count:])
    Y3_test = np.array([p])
    Y3_test = pd.DataFrame(Y3_test, columns=columns[:3*cosine_count])
    print("Testing params view:", Y3_test.shape, "\n", Y3_test, "\n")
    print("Testing scatter view:", X3_test.shape, "\n", X3_test, "\n")
    predict = bnn.predict(X3_test, Y3_test, penalty_mask=pt.as_tensor_variable((1.0,1.0,0.0)))
    params = np.array(predict['param']).squeeze()
    amps = np.array(params[:,0])
    wls = np.array(params[:,1])
    phases = np.array(params[:,2])
    posterior_samples_grouped = np.column_stack((amps, wls, phases))[:,np.newaxis,:]

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
    b = np.random.choice(range(posterior_samples_grouped.shape[0]),choice_count)

    plt.xlabel("x (m)")
    plt.ylabel("Surface elevation (m)")
    plt.savefig("results/" + "bnn" + " reconstruction.png")

    print("Creating response plot")
    plt.figure(figsize=(16,6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)
    plt.plot(trueScatter, color="#d7191c")

    def generate_microphone_pressure(parameters,uSamples=700):
        def newFunction(x):
            return SurfaceFunctionMulti(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter/factor   
    
    for i in range(len(b)):
        plt.plot(generate_microphone_pressure(posterior_samples_grouped[b[i]]), 'k', alpha=0.04)
    
    plt.xlabel("Microphone index")
    plt.ylabel("Response")
    plt.savefig("results/" + "bnn" + " traces.png")

    import corner
    fig = corner.corner(np.array(posterior_samples_grouped)*np.array([100.0,100.0,2.0*np.pi]),bins=200,
              quantiles=[0.16, 0.5, 0.84],labels=[r"A (cm)", r"$\lambda$ (cm)", r"$\phi$ (rad)"],
              show_titles=True, title_fmt = ".4f", figsize=(16, 8))
    
    for ax in fig.axes:
        ax.grid(which='both')  # Enable both major and minor grid lines
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=0.75)

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

    plt.figure(figsize=(8,6))
    plt.grid()
    plt.hist(amps*100.0, bins=50, alpha=0.5, label="$A$", color='#2c7bb6')
    plt.xlabel("Predicted Amplitude (cm)")
    plt.show()

    plt.figure(figsize=(8,6))
    plt.grid()
    plt.hist(wls*100.0, bins=50, alpha=0.5, label="$\lambda$", color='#2c7bb6')
    plt.xlabel("Predicted Wavelength (cm)")
    plt.show()