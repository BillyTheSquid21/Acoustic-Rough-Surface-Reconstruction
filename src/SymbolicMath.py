import pytensor.tensor as pt

def SymIntegral(y, x, axis=-1):
    """
    Compute the integral of y(x) using the composite Simpson's rule for symbolic tensors.

    Parameters:
        y: The tensor to integrate.
        x: The sample points corresponding to y. Must be uniformly spaced.
        axis: The axis along which to integrate. Default is the last axis.

    Returns:
        pt.Tensor: The integral computed using Simpson's rule.
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
        pt.Tensor: The gradient computed using the central difference rule
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