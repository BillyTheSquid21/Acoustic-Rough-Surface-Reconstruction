import numpy as np
import scipy as sp

def FourierRes(signal, timeIncrement,axis = 0):
    '''Just give us the fourier transform of a signal'''
    import scipy as sp
    sampleFrequency = 1/timeIncrement
    n = signal.shape[0]
    yf = np.fft.fft(signal,axis=axis)
    #xf = sp.fft.fftshift(sp.fft.fftfreq(n,timeIncrement))
    xf = np.fft.fftfreq(n,timeIncrement)
    return np.fft.fftshift(xf),np.fft.fftshift(yf)

def SpatialRes(signal):
    '''Merely a FFT.'''
    y = sp.fft.fft(signal,axis=0)
    return y

def Doppler(signals,dt,axis = 0,real=True):
    '''Takes signals of shape (number of signals, number of samples, number of receivers ).
    Returns the Doppler spectrum of all the receivers.'''
    n = np.array(signals[0]).size
    fSig = []

    for s in signals:
        x,Fsignals = FourierRes(np.array(s),dt,axis=axis)
        fSig.append(Fsignals)
    freq = x
    if real:
        return freq,1/n*np.mean(np.abs(np.array(fSig))**2,axis=0)
    else:
        return freq,1/n*np.mean(np.real(np.array(fSig)),axis=0),1/n*np.mean(np.imag(np.array(fSig)),axis=0)

def FourierCoefs(signal, x):
    '''Return the Fourier transform from a signal.'''
    dx = x[1] - x[0] #Dx
    N = len(x)
    freqs = np.fft.fftfreq(N, dx)[:N//2]
    fft = (np.fft.fft(signal)/N)[:N//2]
    return freqs, fft

def Decompose(freqs, fft, ranges, x):
    '''Decompose a Fourier transform and the frequencies into coefficients. Returns
       the full decomposed surface, the coefficients, and the frequency.'''
    c = fft[0]
    a_s = -2*np.real(fft[1:ranges])
    b_s = -2*np.imag(fft[1:ranges])
    f_comp = 2*np.pi*freqs[1:ranges]
    temp_a = np.reshape(a_s,(-1,1)) + np.zeros((ranges-1,len(x)))
    temp_b = np.reshape(b_s,(-1,1)) + np.zeros((ranges-1,len(x)))
    temp_f = np.reshape(f_comp,(-1,1)) + np.zeros((ranges-1,len(x)))
    summies = (temp_a * np.cos((temp_f * x)) + temp_b * np.sin((temp_f * x)))
    summation = np.sum(summies, axis = 0)
    summation = summation + c
    return np.real(summation), a_s, b_s,f_comp

def ExtractComponents(a_coef, b_coef, freqs):
    '''Extract the fouerier components that aren't extremely small. '''
    surface_parameters = []
    for i in range(len(a_coef)):
        if a_coef[i] > 0.000001 or b_coef[i] > 0.000001:
            surface_parameters.append([a_coef[i], b_coef[i], freqs[i]])
    return surface_parameters