""" Define Utility Functions """
import random
import numpy as np
from scipy.stats import norm

#set random seed for replication
def set_seed():
    random.seed(20250305)

#simulate gbm
def GBM_simulate(S0,mu,sigma,n_steps,n_paths,dt):
    set_seed()
    t = np.linspace(0, n_steps*dt, n_steps)  
    dwt = np.random.normal(0,np.sqrt(dt),size=(n_paths,n_steps))
    wt = np.cumsum(dwt,axis=1)
    St = S0*np.exp((mu-0.5*sigma**2)*t + sigma*wt)
    return St

#simulate BS Call option pricing formula and delta formula
def BSM_call(S, K, T, r, sigma, q):
    set_seed()
    epsilon = 1e-8  # 避免除零问题
    sigma_safe = np.maximum(sigma, epsilon)
    T_safe = np.maximum(T, epsilon)

    d1 = (np.log(S/K) + (r-q+0.5*sigma_safe**2)*T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma*np.sqrt(T)
    BS_call = norm.cdf(d1)*S*np.exp(-q*T) - K*np.exp(-r*T)*norm.cdf(d2)
    BS_delta = norm.cdf(d1)*np.exp(-q*T)
    BS_vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    return BS_call, BS_delta, BS_vega

#generate asset price, option price and delta
def get_sim_path(tau,n_paths,freq):
    #tau: initial time to maturity
    #freq:trading frequency. for example:if freq=2,means trade every 2 days
    set_seed()
    T = 252 #252 trading days one year
    dt_orig = 1/T # if trade every day (unit:year)
    dt = dt_orig*freq #multiply with frequency
    n_steps = int((tau/T)/dt) #int(tau/freq)
    mu = 0.05
    sigma = 0.2
    S0 = 100
    K = 100
    r = 0
    q = 0

    GBM_price = GBM_simulate(S0,mu,sigma,n_steps+1,n_paths,dt)
    new_t = np.arange(tau/T, -freq/T, -freq/T)
    BSCall_price, BSCall_delta,_ = BSM_call(GBM_price,K,new_t,r,sigma,q)
    print("Simulation Finished!")
    return GBM_price, BSCall_price, BSCall_delta


#generate SABR (搞清楚什么是SABR，什么是bartlett对冲)
def sabr_general_sim(num_paths, num_steps, initial_price, mu, initial_volatility, dt, correlation, vol_of_vol, beta):
    set_seed()
    # Generate correlated random shocks
    standard_normal_asset = np.random.normal(size=(num_paths, num_steps))  # Asset price shocks (dW)
    standard_normal_independent = np.random.normal(size=(num_paths, num_steps))  # Independent shocks for volatility
    correlated_vol_shock = (
        correlation * standard_normal_asset
        + np.sqrt(1 - correlation ** 2) * standard_normal_independent
    )  # Correlated volatility shock (dZ)

    # Initialize arrays for asset prices and volatilities
    volatilities = np.zeros((num_paths, num_steps))
    volatilities[:, 0] = initial_volatility  # Set initial volatility

    asset_prices = np.zeros((num_paths, num_steps))
    asset_prices[:, 0] = initial_price  # Set initial asset price

    for t in range(num_steps - 1):
        # Compute the effective volatility term: σ_t * S_t^(β-1)
        current_volatility = volatilities[:, t] * (asset_prices[:, t] ** (beta - 1))
        
        # Update asset prices
        asset_prices[:, t + 1] = asset_prices[:, t] * np.exp(
            (mu - 0.5 * current_volatility ** 2) * dt
            + current_volatility * np.sqrt(dt) * standard_normal_asset[:, t]
        )

        # Update volatility using a lognormal process
        volatilities[:, t + 1] = volatilities[:, t] * np.exp(
            -0.5 * vol_of_vol ** 2 * dt + vol_of_vol * correlated_vol_shock[:, t] * np.sqrt(dt)
        )
    return asset_prices, volatilities

#参考论文 Hagen et al (2002)
def sabr_implied_volatility(initial_volatility, time_to_maturity, spot_price, strike_price,risk_free_rate, dividend_yield, beta, vol_of_vol, correlation):
    set_seed()
    # Compute forward price (F)
    forward_price = spot_price * np.exp((risk_free_rate - dividend_yield) * time_to_maturity)

    # Handle near ATM cases to avoid division errors
    epsilon = 1e-8  # Small perturbation for numerical stability
    strike_price = np.where(np.abs(forward_price - strike_price) < epsilon, forward_price + epsilon, strike_price)

    # Log-moneyness adjustments
    mean_price_factor = (forward_price * strike_price) ** ((1 - beta) / 2)
    log_moneyness = (1 - beta) * np.log(forward_price / strike_price)

    # Compute the A term
    log_moneyness_squared = log_moneyness * log_moneyness
    A = initial_volatility / (
        mean_price_factor * (1 + log_moneyness_squared / 24 + log_moneyness_squared**2 / 1920)
    )

    # Compute the B term
    mean_price_factor_squared = mean_price_factor * mean_price_factor
    B = 1 + time_to_maturity * (
        ((1 - beta) ** 2) * (initial_volatility**2) / (24 * mean_price_factor_squared)
        + correlation * beta * vol_of_vol * initial_volatility / (4 * mean_price_factor)
        + vol_of_vol**2 * (2 - 3 * correlation**2) / 24
    )

    # Compute Phi and Chi for non-ATM cases
    phi_factor = (vol_of_vol * mean_price_factor / initial_volatility) * np.log(forward_price / strike_price)
    chi_factor = np.log(
        (np.sqrt(1 - 2 * correlation * phi_factor + phi_factor**2) + phi_factor - correlation) / (1 - correlation)
    )
    # Compute implied volatility
    implied_volatility = np.where(
        forward_price == strike_price,
        initial_volatility * B / (forward_price ** (1 - beta)),  # ATM case
        A * B * phi_factor / chi_factor  # General case
    )
    return implied_volatility

#bartlett对冲
def sabr_bartlett_delta(sigma, T, S, K, r, q, beta, vol_of_vol, rho):
    set_seed()
    # Compute implied volatility from SABR model
    implied_vol = sabr_implied_volatility(sigma, T, S, K, r, q, beta, vol_of_vol, rho)

    # Compute Black-Scholes call price and greeks
    bs_price, bs_delta, bs_vega = BSM_call(S, K, T, r, implied_vol, q)

    # Compute d(sigma)/dS in the SABR model
    dsigma_dS = vol_of_vol * rho / (S ** beta)

    # Bartlett Delta correction
    bartlett_delta = bs_delta + bs_vega * dsigma_dS

    return bs_price, bs_delta,bartlett_delta

#simulation for SABR
def get_sim_path_SABR(tau,n_paths,freq):
    #tau: initial time to maturity
    #freq:trading frequency. for example:if freq=2,means trade every 2 days
    set_seed()
    T=252 #252 trading days one year
    dt_orig = 1/T # if trade every day (unit:year)
    dt = dt_orig*freq #multiply with frequency
    n_steps = int((tau/T)/dt) #int(tau/freq)
    mu = 0.05
    sigma = 0.2
    S0 = 100
    K = 100
    r = 0
    q = 0
    # SABR parameters
    beta = 1
    rho = -0.4
    vol_of_vol = 0.6

    print("1. Generate asset price paths (SABR)")
    SABR_price, SABR_vol = sabr_general_sim(n_paths, n_steps+1, S0, mu, sigma, dt, rho, vol_of_vol, beta)
    new_t = np.arange(tau/T, -freq/T, -freq/T)
    print("2. Generate BSCall price, BSCall delta and Bartlett delta")
    BSCall_price, BSCall_delta, Bartlett_delta = sabr_bartlett_delta(SABR_vol, new_t, SABR_price, K, r, q, beta, vol_of_vol, rho)
    print("Simulation Finished!")
    return SABR_price, BSCall_price, BSCall_delta, Bartlett_delta