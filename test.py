import numpy as np
import matplotlib.pyplot as plt

def calculate_variance(scores):
    return np.var(scores)

def calculate_closeness(variance):
    return 1 / (1 + variance)

def gaussian_scale(value, min_output=50, max_output=500, mean=0.5, std_dev=0.5):

    # Calculate Gaussian probability density
    gaussian_value = np.exp(-0.5 * ((value - mean) / std_dev) ** 2)
    
    # Scale to desired output range
    scaled_value = min_output + (max_output - min_output) * gaussian_value
    
    return scaled_value

def process_scores(scores):
    # Calculate variance
    variance = calculate_variance(scores)
    
    # Calculate closeness
    closeness = calculate_closeness(variance)
    
    # Scale using Gaussian distribution
    scaled_closeness = gaussian_scale(closeness)
    
    return scaled_closeness

print(process_scores([-0.2675, -0.2674, -0.2673, -0.2676, -0.2675]))