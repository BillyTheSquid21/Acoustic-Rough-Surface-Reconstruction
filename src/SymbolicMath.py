import pytensor
import pytensor.graph
import pytensor.graph.op
import pytensor.tensor as pt
import numpy as np

def SymRandomSurface(beta, x, t, velocity, depth, lM, lm, aw = 1e-3):
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

def SymCosineSurfaceM(x, amp, wl, p):
    # Is equal to Amp*Cos(2*pi*(x/WL + phase))
    # Add a tiny offset to wavelength as otherwise could div by 0!
    return amp*np.cos((2*np.pi*((x/(wl+1e-10)) + p)))

def SymCosineSumSurfaceM(x, amps, wls, ps):
    surface = np.zeros(x.shape)
    for i in range(0, len(amps)):
        surface += SymCosineSurfaceM(x,amps[i], wls[i], ps[i])
    return surface

def SymCosineSurface(x, params):
    # Is equal to Amp*Cos(2*pi*(x/WL + phase))
    # Add a tiny offset to wavelength as otherwise could div by 0!
    return params[0]*np.cos((2*np.pi*((x/(params[1]+1e-10)) + params[2])))

def SymCosineSumSurface(x, params_array):
    surface = np.zeros(x.shape)
    for p in params_array:
        surface += SymCosineSurface(x,p)
    return surface

def SymCosineSumSurface(x, params_array):
    surface = np.zeros(x.shape)
    for p in params_array:
        surface += SymCosineSurface(x,p)
    return surface

def SymCosineSurfaceVectorized(x, amp, wl, p):
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

def SymCosineSumSurfaceVectorized(x, amps, wls, ps):
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