import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar, approx_fprime

# Define the limit-state function G2(U)
def G2(U):
    return ((U[0] - 3)**2 / 2**2) + ((U[1] - 2)**2 / 1**2) - 1

# Define a function to compute the gradient of G2(U) using finite differences
def gradient_G2(U, eps=1e-6):
    return approx_fprime(U, G2, eps)

# Function to compute the second derivative of K_Q(t) at t_s
def K_Q_second_derivative(Hessian, t_s):
    r_Z_i = np.sqrt(np.diag(Hessian))
    second_deriv = np.sum([
        (2 * r_Z_i[i] ** 2 / (1 - 2 * r_Z_i[i] ** 2 * t_s)**2)
        for i in range(len(r_Z_i))
        if 1 - 2 * r_Z_i[i] ** 2 * t_s > 0
    ])
    return second_deriv

# Obtain the CGF of Q(U)
def K_Q(t, Hessian, u_star):
    r_Z_i = np.sqrt(np.diag(Hessian))
    K_Q_t = np.sum([
        r_Z_i[i] ** 2 * t / (1 - 2 * r_Z_i[i] ** 2 * t) - 0.5 * np.log(1 - 2 * r_Z_i[i] ** 2 * t)
        for i in range(len(u_star))
        if 1 - 2 * r_Z_i[i] ** 2 * t > 0
    ])
    return K_Q_t

# Function to compute the saddlepoint t_s and probability of failure p_f
def compute_saddlepoint(u_star, hessian):
    # Compute the gradient vector at the specified MPP
    gradient = gradient_G2(u_star)

    # Use the provided Hessian matrix
    Hessian = np.array(hessian)

    # Transform into a linear combination of chi-square variables (assuming unchanged)
    def transform_to_chi_square(U):
        Q_U = np.sum((U - u_star) ** 2 / np.diag(Hessian))
        return Q_U

    # Solve for the saddlepoint t_s
    def saddlepoint_eq(t):
        return K_Q(t, Hessian, u_star) - t

    # Improved bracket search to ensure valid points
    def find_valid_bracket():
        lower, upper = -10, 10
        step = 0.1
        for t in np.arange(lower, upper, step):
            try:
                if np.isfinite(K_Q(t, Hessian, u_star)):
                    return t, upper
            except:
                continue
        return -10, 10  # Fallback to original bracket if valid bracket not found

    bracket = find_valid_bracket()
    saddlepoint_result = root_scalar(saddlepoint_eq, bracket=bracket)
    t_s = saddlepoint_result.root

    # Compute w and t values, handling invalid cases
    K_Q_t_s = K_Q(t_s, Hessian, u_star)
    if K_Q_t_s < 0:
        w = np.sign(t_s) * np.sqrt(-2 * K_Q_t_s)
    else:
        w = np.sign(t_s) * np.sqrt(2 * K_Q_t_s)
    
    K_Q_second_t_s = K_Q_second_derivative(Hessian, t_s)
    t = t_s * np.sqrt(K_Q_second_t_s)

    # Calculate the probability of failure p_f, handling edge cases
    if w == 0:
        p_f = norm.cdf(w)
    else:
        p_f = norm.cdf(w) + (norm.pdf(w) / w) * (-1 / t + 1 / w)

    return u_star, gradient, Hessian, t_s, p_f, w, t, K_Q_t_s, K_Q_second_t_s

# Example usage:
if __name__ == "__main__":
    u_star_input = np.array([1.17, 1.59])
    hessian_input = [[0.5, 0], [0, 2]]  # User-provided Hessian matrix

    # Compute results
    u_star_result, gradient_result, hessian_result, t_s_result, p_f_result, w, t, K_Q_t_s, K_Q_second_t_s = compute_saddlepoint(u_star_input, hessian_input)
    
    print("Hessian matrix:\n", hessian_result)
    print("Saddlepoint t_s:", t_s_result)
    print("K_Q(t_s):", K_Q_t_s)
    print("K_Q''(t_s):", K_Q_second_t_s)
    print("w:", w)
    print("t:", t)
    print("Probability of failure p_f:", p_f_result)
