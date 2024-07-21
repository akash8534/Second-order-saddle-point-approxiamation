import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize, approx_fprime

# Define the limit-state function G2(U)
def G2(U):
    return ((U[0] - 3)**2 / 2**2) + ((U[1] - 2)**2 / 1**2) - 1

# Define a function to compute the gradient of G2(U) using finite differences
def gradient_G2(U, eps=1e-6):
    return approx_fprime(U, G2, eps)

# Define a function to compute the Hessian of G2(U) using finite differences
def hessian_G2(U, eps=1e-6):
    n = len(U)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                hessian[i][j] = (G2(U + eps * (np.array([1 if k == i else 0 for k in range(n)]) + np.array([1 if k == j else 0 for k in range(n)]))) - 2 * G2(U) + G2(U - eps * (np.array([1 if k == i else 0 for k in range(n)]) + np.array([1 if k == j else 0 for k in range(n)])))) / eps**2
            else:
                hessian[i][j] = (G2(U + eps * (np.array([1 if k == i else 0 for k in range(n)]) + np.array([1 if k == j else 0 for k in range(n)]))) - G2(U + eps * np.array([1 if k == i else 0 for k in range(n)])) - G2(U + eps * np.array([1 if k == j else 0 for k in range(n)])) + G2(U)) / eps**2
    return hessian

# Function to compute the saddlepoint t_s and probability of failure p_f
def compute_saddlepoint(u_star):
    # Compute the gradient vector at the specified MPP
    gradient = gradient_G2(u_star)

    # Compute the Hessian matrix at the specified MPP
    Hessian = hessian_G2(u_star)

    # Transform into a linear combination of chi-square variables (assuming unchanged)
    def transform_to_chi_square(U):
        k = np.array([1, 1])  # Assuming two independent standard normal variables
        r_Z = np.sqrt(np.diag(Hessian))
        Q_U = np.sum((U - u_star) ** 2 / np.diag(Hessian))
        return Q_U

    # Obtain the CGF of Q(U)
    def K_Q(t):
        k_i = np.array([1, 1])
        r_Z_i = np.sqrt(np.diag(Hessian))
        K_Q_t = np.sum([
            k_i[i] * r_Z_i[i] ** 2 * t / (1 - 2 * r_Z_i[i] ** 2 * t) - 0.5 * np.log(1 - 2 * r_Z_i[i] ** 2 * t)
            for i in range(len(u_star))
            if 1 - 2 * r_Z_i[i] ** 2 * t > 0  # Ensure valid log argument
        ])
        return K_Q_t

    # Solve for the saddlepoint t_s
    def saddlepoint_eq(t):
        return K_Q(t) - t

    # Improved bracket search to ensure valid points
    def find_valid_bracket():
        lower, upper = -10, 10
        step = 0.1
        for t in np.arange(lower, upper, step):
            try:
                if np.isfinite(K_Q(t)):
                    return t, upper
            except:
                continue
        return -10, 10  # Fallback to original bracket if valid bracket not found

    bracket = find_valid_bracket()
    saddlepoint_result = root_scalar(saddlepoint_eq, bracket=bracket)
    t_s = saddlepoint_result.root

    # Compute w and t values, handling invalid cases
    K_Q_t_s = K_Q(t_s)
    if K_Q_t_s < 0:
        w = np.sign(t_s) * np.sqrt(-2 * K_Q_t_s)
    else:
        w = np.sign(t_s) * np.sqrt(2 * K_Q_t_s)
    t = t_s * np.sqrt(abs(K_Q_t_s))

    # Calculate the probability of failure p_f, handling edge cases
    if w == 0:
        p_f = norm.cdf(w)
    else:
        p_f = norm.cdf(w) + (norm.pdf(w) / w) * (-1 / t +1 / w)

    return u_star, gradient, Hessian, t_s, p_f

# Example usage:
if __name__ == "__main__":
    u_star_input = np.array([1.27, 1.59])
    
    # Compute results
    u_star_result, gradient_result, hessian_result, t_s_result, p_f_result = compute_saddlepoint(u_star_input)
    
    print("Probability of failure p_f:", p_f_result)
