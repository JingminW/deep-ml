import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    norm = np.linalg.norm(gradient)

    if norm == 0.0:
        direction = np.zeros_like(gradient)
        descent_direction = np.zeros_like(gradient)
    else:
        direction = gradient / norm
        descent_direction = -direction

    return {"magnitude": norm, "direction": direction, "descent_direction": descent_direction}