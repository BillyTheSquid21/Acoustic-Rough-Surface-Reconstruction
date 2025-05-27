import pytensor
import pytensor.graph
import pytensor.graph.op
import pytensor.tensor as pt
import numpy as np
import scipy as sp

'''
A collection of important maths functions for the project. 

Some are vectorized to maintain symbolic graph in pymc.
'''

def AngularMean(phases):
    '''
    Calculates the angular mean of the phases which accounts for phase wrapping.
    Described in Johnson et al (2024).

    Parameters:
        phases: An array of phases for one cosine wave

    Returns:
        float: The angular mean of the phases, in range [-0.5,0.5]
    '''
    adjusted_phases = 2.0*np.pi*phases
    sin_sum = np.sum(np.sin(adjusted_phases))
    cos_sum = np.sum(np.cos(adjusted_phases))
    return np.arctan2(sin_sum, cos_sum)/(2.0*np.pi)

def RMS(x):
    '''
    Calculates the Root Mean Squared (RMS) values of an input.

    Parameters:
        x (np.array): The values to calculate the RMS of
    
    Returns:
        float: The RMS value
    '''
    return np.sqrt(np.mean(x**2.0))

def GetSpecularIndices(sourceLocation, sourceAngle, sourceFreq, receiverLocations):
    '''
    Get the indices of the receivers within the specular region, based on the half power beam width.

    Parameters:
        sourceLocation (tuple): The location in world space of the acoustic source.
        sourceAngle (float): Angle the acoustic source is at
        sourceFrequency (float): The frequency of the source acoustic wave
        recieverLocations (Array or List): The locations in world space of the recievers. Must be an array of (x,y) coordinates.

    Returns:
        List: The indices of the receivers within the specular region.
    '''

    def get_spec_point(sourceloc, recloc):
        '''
        Zero index is source, One index is reciever
        '''
        return (recloc[0]*sourceloc[1])/(sourceloc[1]+recloc[1])
    
    # Get point where directivity intersects
    specpoint = sourceLocation[0] + sourceLocation[1]*np.tan((np.pi/2)-sourceAngle)

    # Work out spread based on width of node
    # 1st method: from angle work out width of lobe at spec point
    specdist = (np.array(sourceLocation) - specpoint)
    specdist = np.sqrt(specdist.dot(specdist))

    # Get Half Power Beam Width
    k = (2*np.pi*sourceFreq)/343
    a = 0.02
    hpbw = HalfPowerBeamWidth(k, a, specdist)

    specspread_left = (specdist * (np.sin(hpbw/2))) / (np.sin(np.pi-sourceAngle-(hpbw/2)))
    specspread_right = (specdist * (np.sin(hpbw/2))) / (np.sin(sourceAngle-(hpbw/2)))

    indices = []
    for i in range(len(receiverLocations)):
        rspec = get_spec_point(sourceLocation, receiverLocations[i])
        if rspec > specpoint - specspread_left and rspec < specpoint + specspread_right:
            indices.append(i)
    return indices

def AcousticSourceDirectivity(theta, k, a, dist):
    '''
    Directivity function for the microphones

    Parameters:
        theta (float): The angle from 0 within the directivity region.
        k (float): The source wavenumber.
        a (float): The piston aperture.
        dist (float): The distance from the source.

    Returns:
        float: The directivity magnitude.
    '''
    # https://homepages.uc.edu/~masttd/papers/2005_jasa_piston.pdf equation 20
    d = -1j*k*(a**2)*(sp.special.jn(1,k*a*np.sin(theta)))/(k*a*np.sin(theta)) * ((np.e**(1j*k*dist))/dist)
    return np.abs(d*(dist*(np.e**(-1j*k*dist))))

def HalfPowerBeamWidth(k, a, dist):
    '''
    Half Power Beam Width (HPBW) function for the lobe width
    
    Parameters:
        k (float): The source wavenumber.
        a (float): The piston aperture.
        dist (float): The distance from the source.

    Returns: The HPBW in degrees
    '''
    max_directivity = AcousticSourceDirectivity(0.001, k, a, dist)
    half_power = max_directivity * (1/np.sqrt(2)) # Power is square of intensity, so when is down by 1/root2

    # Move out from center until directivity <= half_power
    # Only do 180 degree sweep
    hpbw = np.pi
    for i in range(1,180):
        theta = float(i)*(np.pi/180.0)
        d = AcousticSourceDirectivity(theta, k, a, dist)
        if d <= half_power:
            hpbw = float(theta)*2.0
            break
    return hpbw

def SymRandomSurface(beta, x, t, velocity, depth, lM, lm, aw = 1e-3):
  '''
  Generates a random complex surface that is kirchoff valid
  '''
  g = 9.81
  surface_tension = 72.75e-03
  density = 998.2

  Dx = x[1] - x[0]
  Dt = t[1] - t[0]

  ksx = 2*np.pi / Dx
  omega = 2*np.pi / Dt

  omega = 2*np.pi / Dt

  Dkx = ksx / len(x)
  Domega = omega / len(t)

  Nt = np.arange(-omega/2, omega/2, Domega)

  Kx = np.arange(-ksx/2, ksx/2, Dkx)
  Kx = np.tile(Kx,[int(len(Nt)),1])

  t = np.tile(np.array(t), [Kx.shape[1],1])
  t = np.swapaxes(t, 0, 1)

  A = (np.abs(Kx)**(-beta/2)) #here
  A[np.where(np.abs(Kx) > 2*np.pi/lm)] = 0
  A[np.where(np.abs(Kx) < 2*np.pi/lM)] = (2*np.pi/lM)**(-beta/2)

  turb = Kx*velocity
  gw1 = turb + np.sqrt((g + surface_tension/density*Kx**2)*Kx*np.tanh(Kx*depth))
  gw2 = turb - np.sqrt((g + surface_tension/density*Kx**2)*Kx*np.tanh(Kx*depth))

  rand1 = np.tile(np.random.randn(A.shape[1]) + 1j *np.random.randn(A.shape[1]),[A.shape[0],1])
  rand2 = np.tile(np.random.randn(A.shape[1]) + 1j *np.random.randn(A.shape[1]),[A.shape[0],1])
  rand3 = np.tile(np.random.randn(A.shape[1]) + 1j *np.random.randn(A.shape[1]),[A.shape[0],1])

  spec1 = np.fft.ifftshift(rand1 * A * np.exp(1j * turb * t), axes=1)
  spec2 = np.fft.ifftshift(rand2 * A *np.exp(1j * gw1 * t), axes=1)
  spec3 = np.fft.ifftshift(rand3 * A * np.exp(1j * gw2 * t),axes=1)

  surf1 = np.fft.ifft(spec1, axis = 1)
  surf2 = np.fft.ifft(spec2, axis = 1)
  surf3 = np.fft.ifft(spec3, axis = 1)
  
  total = (surf1 + surf2 + surf3)
  total = (np.real(total)/np.std(total)*aw)
  return total

def CosineSurfaceM(x, amp, wl, p):
    '''
    Generates a single cosine surface from amplitude, wavelength and phase.

    Parameters:
        x (np.array): The domain over which to compute the surface.
        amp (float): The amplitude of the surface.
        wl (float): The wavelength of the surface.
        p (float): The phase of the surface

    Returns:
        np.array: The surface amplitudes over the domain.
    '''

    # Is equal to Amp*Cos(2*pi*(x/WL + phase))
    # Add a tiny offset to wavelength as otherwise could div by 0!
    return amp*np.cos((2*np.pi*((x/(wl+1e-10)) + p)))

def CosineSumSurfaceM(x, amps, wls, ps):
    '''
    Generates a multiple cosine surface from amplitudes, wavelengths and phases.

    Parameters:
        x (np.array): The domain over which to compute the surface.
        amps (Array or List): The amplitudes of the surface.
        wls (Array or List): The wavelengths of the surface.
        ps (Array or List): The phases of the surface

    Returns:
        np.array: The surface amplitudes over the domain.
    '''
    surface = np.zeros(x.shape)
    for i in range(0, len(amps)):
        surface += CosineSurfaceM(x,amps[i], wls[i], ps[i])
    return surface

def CosineSurface(x, params):
    '''
    Generates a single cosine surface from amplitude, wavelength and phase packed into a tuple.

    Parameters:
        x (np.array): The domain over which to compute the surface.
        params (tuple): The surface parameters (amp,wl,phase)

    Returns:
        np.array: The surface amplitudes over the domain.
    '''
    return params[0]*np.cos((2*np.pi*((x/(params[1]+1e-10)) + params[2])))

def CosineSumSurface(x, paramsArray):
    '''
    Generates a multiple cosine surface from amplitudes, wavelengths and phases packed into an array.

    Parameters:
        x (np.array): The domain over which to compute the surface.
        paramsArray (np.array): The surface parameters ((amp1,wl1,phase1), (amp2,wl2,phase2), ...)

    Returns:
        np.array: The surface amplitudes over the domain.
    '''
    surface = np.zeros(x.shape)
    for p in paramsArray:
        surface += CosineSurface(x,p)
    return surface

def SymCosineSurface(x, amp, wl, p):
    """
    Vectorized version of a cosine surface function

    Parameters:
        x: The domain over which to calculate the cosine values
        amp: A tensor value of the amplitude for the surface
        wl: A tensor value of the wavelength for the surface
        p: A tensor value of the phase for the surface

    Returns:
        pytensor.Tensor: A single cosine surface contribution
    
    """
    return amp * pt.cos(2 * pt.pi * ((x / (wl + 1e-10)) + p))

def SymCosineSumSurface(x, amps, wls, ps):
    """
    Vectorized version of a cosine sum surface function

    Parameters:
        x: The domain over which to calculate the cosine contributions
        amps: A tensor containing all the amplitude contributions
        wls: A tensor containing all the wavelength contributions
        ps: A tensor containing all the phase contributions

    Returns:
        pytensor.Tensor: The sum of the surface contributions
    """
    
    # Calculate surface contributions without a Python function call in scan
    surface_contributions, _ = pytensor.scan(
        fn=lambda amp, wl, p: amp * pt.cos(2 * pt.pi * ((x / (wl + 1e-10)) + p)),
        sequences=[amps, wls, ps]
    )

    # Sum the contributions along the correct axis
    return surface_contributions.sum(axis=0)

def SymBessel(n, x):
    """
    Computes the bessel function of the first kind of x

    Parameters:
        n: The order of the bessel function
        x: The tensor to compute over

    Returns:
        pytensor.Tensor: The bessel function values

    """

    # PyTensor doesn't have a built-in Bessel function, so we use a series expansion
    bessel_sum = 0
    for k in range(10):  # 10 terms of the series
        term = ((-1)**k) * ((x / 2)**(2 * k + n)) / (pt.gamma(k + 1) * pt.gamma(k + n + 1))
        bessel_sum += term
    return bessel_sum

def SymIntegral(y, x, axis=-1):
    """
    Compute the integral of y(x) using the composite Simpson's rule for symbolic tensors.

    Parameters:
        y: The tensor to integrate.
        x: The sample points corresponding to y. Must be uniformly spaced.
        axis: The axis along which to integrate. Default is the last axis.

    Returns:
        pytensor.Tensor: The integral computed using Simpson's rule.
    """
    # Ensure the axis is valid
    if axis < 0:
        axis += y.ndim

    # Move the integration axis to the end for easier manipulation
    y = pt.moveaxis(y, axis, -1)
    x = pt.moveaxis(x, 0, -1)

    # Create the Simpson's rule weights
    weights = pt.ones_like(y)
    weights = pt.set_subtensor(weights[..., 1:-1:2], 4)  # Odd indices
    weights = pt.set_subtensor(weights[..., 2:-1:2], 2)  # Even indices

    # Compute the spacing (h) between x values
    h = x[1] - x[0]

    # Apply Simpson's rule
    integral = (h / 3) * pt.sum(weights * y, axis=-1, keepdims=False)
    return integral

def SymGradient(y, x):
    '''
    Compute the gradient of y(x) using the central difference rule for symbolic tensors.

    Parameters:
        y: The tensor to compute the gradient over. Must be uniformly spaced.
        x: The sample points corresponding to y.

    Returns:
        pytensor.Tensor: The gradient computed using the central difference rule
    '''
    # Compute dx (assume uniform spacing)
    dx = x[1] - x[0]  # Symbolic spacing

    # Forward difference for the first point
    dy_start = (y[1] - y[0]) / dx

    # Central differences for inner points
    dy_inner = (y[2:] - y[:-2]) / (2 * dx)

    # Backward difference for the last point
    dy_end = (y[-1] - y[-2]) / dx

    # Combine into a single array
    dy_full = pt.concatenate([[dy_start], dy_inner, [dy_end]])

    return dy_full