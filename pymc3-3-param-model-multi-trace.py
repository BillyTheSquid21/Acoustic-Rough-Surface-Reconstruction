import numpy as np
import matplotlib.pyplot as plt
import jax
import arviz as az
import corner
import scienceplots
import os
from tqdm import tqdm

from src.Directed2DVectorized import Directed2DVectorised
from src.AcousticParameterMCMC import AcousticParameterMCMC

from src.SymbolicMath import SymCosineSumSurfaceVectorized
from src.SymbolicMath import SymCosineSurface
from src.SymbolicMath import SymRMS

def modelRun():
    plt.style.use('science')
    plt.rcParams["font.family"] = "Bitstream Charter"

    # Check jax backend
    print("jax device: ", jax.default_backend(), " ", jax.device_count())

    cosine_count = 1

    output_folder = "results/noisy-noisy"

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
    sourceFreq = 14_000
    userSamples = 700

    # Scatter data requirements
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)

    def generate_microphone_pressure(parameters,uSamples=userSamples):
        def newFunction(x):
            return SymCosineSurface(x, parameters)

        KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,sourceFreq,0.02,SourceAngle,'simp',userMinMax=[-1,1],userSamples=uSamples,absolute=False)
        scatter = KA_Object.Scatter(absolute=True,norm=False)
        return scatter

    # Generate scatter for varying amps and wavelengths
    # Only vary two for now
    p_count = 50
    pc_noise = 0.05

    def noise_sigma(s, pc):
        '''
        Get scale of noise from rms average of the signal
        '''
        return (SymRMS(np.array(s)))*pc

    amp_bounds = [0.001, 0.00475]
    wl_bounds = [0.05, 0.15]
    phase_bounds = [-0.5, 0.5]
    frequency=sourceFreq
    generate_data = False
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

            an = 0.02

            KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,an,SourceAngle,'simp',userMinMax=[-1,1],userSamples=700,absolute=False)
            if not KA_Object.surfaceChecker(True, True):
                # While the surface is not valid, keep rerolling until it is
                while not KA_Object.surfaceChecker(True, True):
                    params[p] = (np.random.uniform(amp_bounds[0], amp_bounds[1]), np.random.uniform(wl_bounds[0], wl_bounds[1]), np.random.uniform(phase_bounds[0], phase_bounds[1]))
                    KA_Object = Directed2DVectorised(SourceLocation,RecLoc,newFunction,frequency,an,SourceAngle,'simp',userMinMax=[-1,1],userSamples=700,absolute=False)

            s = KA_Object.Scatter(absolute=True,norm=False)
            s += np.random.normal(loc=0.0, scale=noise_sigma(s, pc_noise), size=(34,))
            s = np.abs(s)
            scatters.append(s)

        params = np.array(params)
        scatters = np.array(scatters)

        plt.hist(params[:, 0], bins=50, alpha=0.5, label="amp")
        plt.legend()
        plt.show()

        plt.scatter(params[:, 0], params[:, 0])
        plt.show()

        plt.hist(params[:, 1], bins=50, alpha=0.5, label="wl")
        plt.legend()
        plt.show()

        plt.hist(params[:, 2], bins=50, alpha=0.5, label="phase")
        plt.legend()
        plt.show()

        np.savetxt(output_folder + "/data/amp_wl_phase_params_14KHz.csv", params, delimiter=",")
        np.savetxt(output_folder + "/data/amp_wl_phase_scatter_14KHz.csv", scatters, delimiter=",")

    params = np.loadtxt(output_folder + "/data/amp_wl_phase_params_14KHz.csv", delimiter=",")
    scatters = np.loadtxt(output_folder + "/data/amp_wl_phase_scatter_14KHz.csv", delimiter=",")

    # Update flat response factor here to get in model scale
    factor = AcousticParameterMCMC.GenerateFactor(SourceLocation, SourceAngle, RecLoc, 0.02, sourceFreq)

    # Run the samplers and generate traces for each one
    run_samplers = False
    kernel = "NUTS"
    userSamples = 700
    current_p = ""

    sample_count = 20_000
    burn_in_count = 20_000
    if run_samplers:

        pbar = tqdm(range(len(params)))
        for i in pbar:
            p = params[i]
            s = scatters[i]
            current_p = "AMP_" + str(p[0]) + "_WL_" + str(p[1]) + "_PHASE_" + str(p[2])
            pbar.set_postfix({'Current Params': current_p})
            filename = output_folder + "/" + current_p

            # Create the model MCMC sampler
            mcmc = AcousticParameterMCMC(cosineCount=cosine_count, 
                                            sourceLocation=SourceLocation, 
                                            sourceAngle=SourceAngle,
                                            receiverLocations=RecLoc, 
                                            truescatter=s, 
                                            userSampleDensity=userSamples, 
                                            sourceFrequency=sourceFreq)
            
            mcmc.setFileName(filename)
            mcmc.setAmplitudeProposal(np.array([0.01]))
            mcmc.setWavelengthProposal(np.array([0.2]))
            mcmc.setError(0.15**2)
            
            # Run the model MCMC sampler
            mcmc.run(kernel=kernel, 
                        chainCount=1, 
                        surfaceFunction=SymCosineSumSurfaceVectorized, 
                        burnInCount=burn_in_count, 
                        sampleCount=sample_count, 
                        scaleTrueScatter=True,
                        targetAccRate=0.9)
            
    # Read in all CSV's from the large set group
    def list_csv_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    
    def tokenize_filename(filename):
        return filename.rsplit('/', 1)[-1].split('_')

    csv_files = list_csv_files(output_folder)
    true_amps = []
    true_wls = []
    true_phases = []
    posterior_samples_grouped = []
    for file in csv_files:
        posterior_samples_grouped.append(np.array(AcousticParameterMCMC.LoadCSVData(file)))
        
        tokens = tokenize_filename(file)
        true_amps.append(float(tokens[1]))
        true_wls.append(float(tokens[3]))
        true_phases.append(float(tokens[5].split('.')[0] + "." + tokens[5].split('.')[1]))

    posterior_samples_grouped = np.array(posterior_samples_grouped).squeeze()

    amps = np.array(posterior_samples_grouped[:,:,0])
    wls = np.array(posterior_samples_grouped[:,:,1])
    phases = np.array(posterior_samples_grouped[:,:,2])*2.0*np.pi

    # Convert to NumPy arrays
    true_amp = np.array(true_amps)
    true_wl = np.array(true_wls)
    true_phase = np.array(true_phases)*2.0*np.pi

    amps_hdi = az.hdi(amps.T, hdi_prob=0.68).T
    wls_hdi = az.hdi(wls.T, hdi_prob=0.68).T
    phases_hdi = az.hdi(phases.T, hdi_prob=0.68, circular=True).T

    # Amps first
    amp_lower_hdi, amp_upper_hdi = amps_hdi
    pred_amp_sorted = amps.mean(axis=1)

    lower_err = pred_amp_sorted - amp_lower_hdi
    upper_err = np.maximum(amp_upper_hdi - pred_amp_sorted,0)

    # Step 5: Plot with correctly sorted data
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(16, 6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)

    # Scatter plot of predicted vs true values
    plt.errorbar(
        true_amp*100.0, pred_amp_sorted*100.0, 
        yerr=[lower_err*100.0, upper_err*100.0], 
        fmt='o', label="Predicted with 68\% HDI", 
        capsize=4, capthick=1, alpha=0.7, markersize=5, color="#2c7bb6"
    )

    # Scatter plot of true test values
    #plt.scatter(true_amp_sorted, pred_amp_sorted, label="amps")
    plt.scatter(true_amp*100.0, true_amp*100.0, color='#d7191c', label="True test set")

    # Fill HDI region correctly
    #plt.fill_between(true_amp, amp_lower_hdi_smooth, amp_upper_hdi_smooth, 
    #                color='gray', alpha=0.3, label="68%% HDI")

    plt.legend()
    plt.xlabel("True Amplitude (cm)")
    plt.ylabel("Predicted Amplitude (cm)")
    #plt.title("Predicted Amplitude vs True Amplitude with 68\% HDI")
    plt.show()

    from sklearn.metrics import r2_score
    print("R2 score of amp in BNN: ", r2_score(true_amp, pred_amp_sorted))

    # WLs next
    wl_lower_hdi, wl_upper_hdi = wls_hdi
    pred_wl_sorted = wls.mean(axis=1)

    lower_err = np.maximum(pred_wl_sorted - wl_lower_hdi,0)
    upper_err = np.maximum(wl_upper_hdi - pred_wl_sorted,0)

    # Step 5: Plot with correctly sorted data
    plt.figure(figsize=(16, 6))
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.75)

    # Scatter plot of predicted vs true values
    plt.errorbar(
        true_wl*100.0, pred_wl_sorted*100.0, 
        yerr=[lower_err*100.0, upper_err*100.0], 
        fmt='o', label="Predicted with 68\% HDI", 
        capsize=4, capthick=1, alpha=0.7, markersize=5, color="#2c7bb6"
    )

    # Scatter plot of true test values
    #plt.scatter(true_amp, pred_wl_sorted, label="wls")
    plt.scatter(true_wl*100.0, true_wl*100.0, color='#d7191c', label="True test set")

    # Fill HDI region correctly
    #plt.fill_between(true_amp, amp_lower_hdi_smooth, amp_upper_hdi_smooth, 
    #                color='gray', alpha=0.3, label="68%% HDI")

    plt.legend()
    plt.xlabel("True Wavelength (cm)")
    plt.ylabel("Predicted Wavelength (cm)")
    #plt.title("Predicted Wavelength vs True Amplitude with 68\% HDI")
    plt.show()

    print("R2 score of amp in BNN: ", r2_score(true_wl, pred_wl_sorted))

    #Phases next
    from src.SymbolicMath import SymAngularMean
    phase_lower_hdi, phase_upper_hdi = phases_hdi
    pred_phase_sorted = np.apply_along_axis(SymAngularMean, arr=phases.copy()/(2.0*np.pi), axis=1)*2.0*np.pi

    lower_err = np.arccos(np.cos(pred_phase_sorted - phase_lower_hdi))
    upper_err = np.arccos(np.cos(phase_upper_hdi - pred_phase_sorted))

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')
    plt.thetagrids()

    # Plot predicted vs. true phase with error bars
    ax.errorbar(
        true_phase, pred_phase_sorted,
        yerr=[lower_err, upper_err],
        fmt='o', label="Predicted with 68\% HDI",
        capsize=4, capthick=1, alpha=0.7, markersize=5, color="#2c7bb6"
    )

    # Reference line for true values
    ax.plot(true_phase, true_phase, 'o', color='#d7191c', label="True test set")

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
        np.deg2rad(0),  # angle = 90 degrees in radians (i.e., left side of plot)
        ax.get_rmin() + 6.75,  # place it just outside the largest tick
        "Predicted Offset",  # your label text
        ha='left', va='center',
        rotation=0  # optional: vertical text
    )

    # Title and labels
    plt.xlabel("True Offset")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1,-0.1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    modelRun()