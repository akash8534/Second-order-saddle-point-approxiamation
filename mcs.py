import numpy as np

# Define the limit-state function G2(U)
def G2(U):
    return ((U[0] - 3)**2 / 2**2) + ((U[1] - 2)**2 / 1**2) - 1

# Monte Carlo Simulation function
def monte_carlo_simulation(num_samples, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random samples from the standard normal distribution
    samples = np.random.randn(num_samples, 2)
    
    # Evaluate the limit-state function for each sample
    G2_values = np.apply_along_axis(G2, 1, samples)
    
    # Estimate the probability of failure
    num_failures = np.sum(G2_values < 0)
    pf_estimate = num_failures / num_samples
    
    return pf_estimate

# Example usage:
if __name__ == "__main__":
    num_samples = 100000  # Number of samples for Monte Carlo Simulation
    
    # Compute the probability of failure using MCS
    pf_mcs = monte_carlo_simulation(num_samples, seed=42)  # Using seed for reproducibility
    
    print("Estimated probability of failure using MCS:", pf_mcs)
