import pytensor
import pytensor.graph
import pytensor.graph.op
import pytensor.tensor as pt
import numpy as np

def SymCosineSurface(x, params):


    # Is equal to Amp*Cos(2*pi*(x/WL + phase))
    # Add a tiny offset to wavelength as otherwise could div by 0!
    return params[0]*np.cos((2*np.pi*((x/(params[1]+1e-10)) + params[2])))

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
    return amp * np.cos(2 * np.pi * ((x / (wl + 1e-10)) + p))

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