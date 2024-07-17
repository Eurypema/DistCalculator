import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #FigureCanvasTkAgg is a specific Matplotlib canvas designed to be embedded within a Tkinter application
from itertools import product, combinations
import scipy.special
import scipy.optimize as opt
from scipy.stats import norm, uniform, expon

#factorial function
def factorial(n): 
    m = 1
    if n != 0:
        for i in range (1,n+1):
            m *= i
    return m

#Binomial function
def comb(n,k):
    if n >=k:
        return factorial(n) // (factorial(k)*factorial(n-k))
    else:
        raise ValueError(f"You cannot choose {k} items from a selection of {n} items.") #raise ValueError("You cannot choose "+k+" items from a selection of "+n+" items.")

#Binomial distribution function
def Binomial(n,k,p,w): #probability that k will happen in n trials
    if not (0 <= p <=1): #if p not in range(0, 1) doesnt work
        raise ValueError(p, "is less than 1.")
    q =1-p
    probability = comb(n,k)*(p**k)*(q**(n-k))
    expectation = n*p
    variance = expectation*q
    if w=='p':
        return probability
    elif w == 'e':
        return expectation
    elif w == 'v':
        return variance
    elif w == 'c':
        return sum([Binomial(n,i,p,'p') for i in range(0,k+1)])
    else:
        raise ValueError("Binomial: Invalid parameter.")

#Multinomial coefficient
def Multinomial_coeff(n,categories): #probability that i will get 2 heads on one coin and 6 tails on another coin if i flip them both simultaneously 10 times
    denominator = np.prod([factorial(k) for k in categories])
    return factorial(n) // denominator

#Multinomial distrubution function

def mnom(n, categories, probs, w):
    if sum(categories) != n:
        raise ValueError("The sum of the categories must be equal to n.")
    if len(categories) != len(probs):
        raise ValueError("The length of categories and probs must be equal.")
    if not np.isclose(sum(probs), 1):
        raise ValueError("The probabilities must sum to 1.")

    if w == 'p':
        prod = np.prod([p**k for p, k in zip(probs, categories)])
        probability = Multinomial_coeff(n, categories) * prod
        return round(probability, 15)
    elif w == 'e':
        expectation = [n * p for p in probs]
        return expectation
    elif w == 'v':
        variance = [n * p * (1 - p) for p in probs]
        return variance
    elif w == 'c':
        cdf = 0
        p = np.array(probs)
        for m in product(range(n+1), repeat=len(categories)):
            if sum(m) == n:
                multinom_coeff = factorial(n) / np.prod([factorial(i) for i in m])
                prob = multinom_coeff * np.prod([p[j]**m[j] for j in range(len(categories))])
                cdf += prob
        return round(cdf,15)
    elif w == 'cv':
        cov_matrix = np.zeros((len(probs), len(probs))) #np.zeroes creates a matrix of zeroes that, in this case, is a square matrix of len(prob) dimension
        for i in range(len(probs)):
            for j in range(len(probs)):
                if i ==j:
                    cov_matrix[i][j] = n*probs[i]*(1-probs[j])
                elif i!=j:
                    cov_matrix[i][j] = -n * probs[i] *probs[j]
        cov = []
        for i in range(len(cov_matrix)):
            for j in range(i + 1, len(cov_matrix)):
                cov.append(cov_matrix[i, j])
        return cov
    else:
        raise ValueError('Multinomial: Invalid parameter.')
    
def Geometric(k, p, w):
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    q = 1 - p
    if w == 'p':
        return p * (q ** (k - 1))
    elif w == 'c':
        return 1 - (q ** k)
    elif w == 'e':
        return 1 / p
    elif w == 'v':
        return (1 - p) / (p ** 2)
    else:
        raise ValueError('Invalid parameter')

def negative_Binomial(k, p, r, w):
    if (0 <= p <= 1):
        q = 1 - p
        if w == 'p':
            return comb(k - 1, r - 1) * (q ** (k - r)) * (p ** r)
        elif w == 'c':
            cdf = 0
            for i in range(r, k + 1):
                cdf += comb(i - 1, r - 1) * (q ** (i - r)) * (p ** r)
            return cdf
        elif w == 'e':
            return r / p
        elif w == 'v':
            return (r * (1 - p)) / (p ** 2)
        else:
            raise ValueError('Invalid parameter')
    else:
        raise ValueError('p must be between 0 and 1.')
    
def Poisson(k, l, w):
    if w == 'p':
        return (l**k * (np.e ** -l)) / factorial(k)
    elif w == 'c':
        return np.sum([(l**i / factorial(i)) * (np.e ** -l) for i in range(k + 1)])
    elif w == 'e':
        return l
    elif w == 'v':
        return l
    else:
        raise ValueError('Invalid parameter.')

def uniform_disc(x, n, k, w):
    if w == 'p':
        return 1/n
    elif w == 'c':
        if x < 1:
            return 0
        elif 1 <= x <= n:
            return x/n
        elif x > n:
            return 1
        else:
            raise ValueError()
    elif w == 'e':
        return (n+1)/2
    elif w == 'v':
        return ((n**2)-1)/12
    else:
        raise ValueError('Invalid parameter.')

def Normal(x, m, s, w, y=None):
    z = (x - m) / s
    if w == 'p':
        return (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - m) / s) ** 2)
    elif w == 'c':
        return 0.5 * (1 + scipy.special.erf(z * (1 / np.sqrt(2))))
    elif w == 'cb' and y is not None:
        return Normal(x, m, s, 'c') - Normal(y, m, s, 'c')
    elif w == 'v':
        return s ** 2
    else:
        raise ValueError('Invalid parameter.')

def uniform_cont(x, a, b, w):
    x = np.asarray(x)
    if w == 'p':
        return np.where((a <= x) & (x <= b), 1 / (b - a), 0)
    elif w == 'c':
        return np.where(x < a, 0, np.where(x <= b, (x - a) / (b - a), 1))
    elif w == 'e':
        return (a + b) / 2
    elif w == 'v':
        return ((b - a) ** 2) / 12
    else:
        raise ValueError("Invalid parameter for uniform distribution")

def exp_dist(x, rate, w):
    if rate <= 0:
        raise ValueError('Rate parameter must be positive.')
    if w == 'p':
        return rate * (np.e ** (-rate * x))
    elif w == 'c':
        return 1 - np.e ** (-rate * x)
    elif w == 'e':
        return 1 / rate
    elif w == 'v':
        return 1 / (rate ** 2)
    else:
        raise ValueError('Invalid parameter.')

def Custom(values, probs, val=None, w='c'):
    if val is None:
        val = max(values)
    if w == 'c':
        return sum([p for v, p in zip(values, probs) if v <= val])
    elif w == 'e':
        return sum([v * p for v, p in zip(values, probs)])
    elif w == 'v':
        expectation = sum([v * p for v, p in zip(values, probs)])
        return sum([(v ** 2) * p for v, p in zip(values, probs)]) - expectation ** 2
    else:
        raise ValueError("Invalid parameter, please input 'c' for CDF, 'e' for expectation, or 'v' for variance.")
    
def Normal_quantile(p, mean=0, std_dev=1):
    # Handle edge cases
    if p == 0:
        return float('-inf')
    if p == 1:
        return float('inf')
    
    # This approximation is based on the Beasley-Springer-Moro algorithm
    # Coefficients for the approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.4735109309, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    # Compute the initial approximation for t based on whether p < 0.5
    p = float(p)
    if p < 0.5:
        t = np.sqrt(-2.0 * np.log(p))
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - p))

    # Calculate the numerator and denominator of the rational approximation
    numerator = (((a[3] * t + a[2]) * t + a[1]) * t + a[0])
    denominator = (((b[3] * t + b[2]) * t + b[1]) * t + b[0])
    quantile = t - numerator / (1.0 + denominator * t)

    # Adjust quantile based on whether p < 0.5
    if p < 0.5:
        quantile = -quantile

    # Final refinement using coefficients c
    quantile += (c[0] + c[1] * quantile + c[2] * quantile**2 + c[3] * quantile**3 +
                 c[4] * quantile**4 + c[5] * quantile**5 + c[6] * quantile**6 +
                 c[7] * quantile**7 + c[8] * quantile**8)

    # Return the quantile adjusted by the mean and standard deviation
    return mean + std_dev * quantile


def uniform_quantile(x,a,b):
    x = float(x)
    return a + x * (b-a)

def Exponential_quantile(x, rate):
    x = float(x)
    return -np.log(1-x)/rate

def custom_quantile(x, values, probs):
    cum_probs = np.cumsum(probs)
    idx = np.searchsorted(cum_probs, x)
    return values[idx]

def normalizer(x, m, s):
    return (x-m)/s

#first attempt
def cdf(dist, params, x):
    if dist == 'Normal':
        _, m, s = params
        return Normal(x, m, s, 'c')
    elif dist == 'Uniform Continuous':
        _, a, b = params
        return uniform_cont(x, a, b, 'c')
    elif dist == 'Exponential':
        _, rate = params
        return exp_dist(x, rate, 'c')
    elif dist == 'Custom':
        values, probs = params
        return Custom(values, probs, x, 'c')
    else:
        raise ValueError("CDF: Unsupported distribution.")
def find_intersections(dist1, params1, dist2, params2, xval):
    def cdf_difference(x):
        quant1 = cdf(dist1, params1, x)
        quant2 = cdf(dist2, params2, x)
        print(f"x: {x}, CDF difference: {quant1 - quant2}")
        return quant1 - quant2

    intersections = []
    prev_diff = None
    for i in range(len(xval) - 1):
        x1, x2 = xval[i], xval[i + 1]
        try:
            diff1 = cdf_difference(x1)
            diff2 = cdf_difference(x2)
            print(f"x1: {x1}, x2: {x2}, diff1: {diff1}, diff2: {diff2}")
            if diff1 * diff2 < 0:
                root = opt.brentq(cdf_difference, x1, x2)
                intersections.append(root)
                print(f"Intersection found at: {root}")
            elif diff1 == 0 and (prev_diff is None or prev_diff * diff2 < 0):
                intersections.append(x1)
                print(f"Boundary intersection found at: {x1}")
            elif diff2 == 0 and diff1 * diff2 < 0:
                intersections.append(x2)
                print(f"Boundary intersection found at: {x2}")
            prev_diff = diff2
        except ValueError as e:
            print(f"ValueError at x1: {x1}, x2: {x2} with error: {e}")
            continue

    return intersections


"""
#second attempt
def find_intersections(dist1, params1, dist2, params2, xval):
    def cdf(dist, params, x):
        if dist == 'Normal':
            return norm.cdf(x, loc=params[1], scale=params[2])
        elif dist == 'Uniform Continuous':
            return uniform.cdf(x, loc=params[1], scale=params[2] - params[1])
        elif dist == 'Exponential':
            return expon.cdf(x, scale=1/params[1])
        elif dist == 'Custom':
            return np.interp(x, params[1], np.cumsum(params[2]) / np.sum(params[2]))
        else:
            raise ValueError("Unsupported distribution")

    cdf1 = cdf(dist1, params1, xval)
    cdf2 = cdf(dist2, params2, xval)
    diff = cdf1 - cdf2

    print(f"xval: {xval}")
    print(f"cdf1: {cdf1}")
    print(f"cdf2: {cdf2}")
    print(f"diff: {diff}")

    intersections = []
    for i in range(1, len(diff)):
        if diff[i-1] * diff[i] < 0:  # Crossing zero
            x1, x2 = xval[i-1], xval[i]
            f1, f2 = diff[i-1], diff[i]
            x_intersection = x1 - f1 * (x2 - x1) / (f2 - f1)
            intersections.append(x_intersection)
        elif diff[i-1] == 0 and (i == 1 or diff[i-2] != 0):  # Exactly zero and not a repeated zero
            intersections.append(xval[i-1])
        elif diff[i] == 0 and (i == len(diff)-1 or diff[i+1] != 0):  # Exactly zero and not a repeated zero
            intersections.append(xval[i])

    # Remove duplicates and sort
    intersections = sorted(set(intersections))

    print(f"intersections: {intersections}")
    return intersections

"""
"""
#third attempt

def calculate_cdf(dist, params, x):
    if dist == 'Normal':
        cdf = norm.cdf(x, params[1], params[2])
    elif dist == 'Exponential':
        cdf = expon.cdf(x, scale=1/params[1])
    elif dist == 'Uniform Continuous':
        cdf = uniform.cdf(x, params[1], params[2] - params[1])
    elif dist == 'Custom':
        x_vals, y_vals = params[1], params[2]
        cdf = np.interp(x, x_vals, np.cumsum(y_vals)/np.sum(y_vals))
    else:
        raise ValueError("Unsupported distribution")
    return cdf

def find_intersections(dist1, params1, dist2, params2, xvals):
    cdf1 = calculate_cdf(dist1, params1, xvals)
    cdf2 = calculate_cdf(dist2, params2, xvals)
    diff = np.abs(cdf1 - cdf2)
    intersections = xvals[np.where(np.diff(np.sign(cdf1 - cdf2)))[0]]
    return intersections

"""


def comparator(dist1, dist2, dist1_params, dist2_params, val1=None, val2=None):
    cdf1 = expectation1 = variance1 = cdf2 = expectation2 = variance2 = None
    print(f'Comparator called with parameters {dist1}, {dist2}, {dist1_params}, {dist2_params}, {val1}, {val2}')
    
    if dist1 == 'Normal':
        x1, m1, s1 = dist1_params
        cdf1 = Normal(x1, m1, s1, 'c')
        expectation1 = m1
        variance1 = s1**2
    elif dist1 == 'Uniform Continuous':
        x1, a1, b1 = dist1_params
        cdf1 = uniform_cont(x1, a1, b1, 'c')
        expectation1 = (a1 + b1) / 2
        variance1 = ((b1 - a1) ** 2) / 12
    elif dist1 == 'Exponential':
        x1, rate1 = dist1_params
        cdf1 = exp_dist(x1, rate1, 'c')
        expectation1 = 1 / rate1
        variance1 = 1 / (rate1 ** 2)
    elif dist1 == 'Custom':
        values1, probs1 = dist1_params
        val1 = float(val1) if val1 is not None else max(values1)
        cdf1 = Custom(values1, probs1, val1, 'c')
        expectation1 = Custom(values1, probs1, w='e')
        variance1 = Custom(values1, probs1, w='v')
    
    if dist2 == 'Normal':
        x2, m2, s2 = dist2_params
        cdf2 = Normal(x2, m2, s2, 'c')
        expectation2 = m2
        variance2 = s2**2
    elif dist2 == 'Uniform Continuous':
        x2, a2, b2 = dist2_params
        cdf2 = uniform_cont(x2, a2, b2, 'c')
        expectation2 = (a2 + b2) / 2
        variance2 = ((b2 - a2) ** 2) / 12
    elif dist2 == 'Exponential':
        x2, rate2 = dist2_params
        cdf2 = exp_dist(x2, rate2, 'c')
        expectation2 = 1 / rate2
        variance2 = 1 / (rate2 ** 2)
    elif dist2 == 'Custom':
        values2, probs2 = dist2_params
        val2 = float(val2) if val2 is not None else max(values2)
        cdf2 = Custom(values2, probs2, val2, 'c')
        expectation2 = Custom(values2, probs2, w='e')
        variance2 = Custom(values2, probs2, w='v')
    
    return cdf1, expectation1, variance1, cdf2, expectation2, variance2

def distributor(m, n, k, p, w, mult_params=None, Normal_params=None, uniform_params=None, exp_params=None, custom_params = None):
    print(f"Distributor called with m={m}, n={n}, k={k}, p={p}, w={w}, mult_params={mult_params}, Normal_params={Normal_params}, uniform_params={uniform_params}, exp_params={exp_params}, custom_params={custom_params}")
    
    if w == 'Binomial':
        if m == 'PMF':
            print ([Binomial(n, k, p, 'p'), Binomial(n, k, p, 'e'), Binomial(n, k, p, 'v')])
            return [Binomial(n, k, p, 'p'), Binomial(n, k, p, 'e'), Binomial(n, k, p, 'v')]
        elif m == 'CDF':
            return [sum([Binomial(n, i, p, 'p') for i in range(0, k + 1)]), Binomial(n, k, p, 'e'), Binomial(n, k, p, 'v')]
        else:
            raise ValueError('Invalid choice of distribution type.')
    elif w == 'Multinomial' and mult_params is not None:
        categories, probs = mult_params
        expectation = mnom(n, categories, probs, 'e')
        variance = mnom(n, categories, probs, 'v')
        probs = np.array(probs)
        if m == 'PMF':
            probability = mnom(n, categories, probs, 'p')
            return [round(probability, 15), expectation, variance]
        elif m == 'CDF':
            total_prob = 0
            outcomes = product(*[range(c + 1) for c in categories])
            for outcome in outcomes:
                if sum(outcome) == n:
                    coeff = Multinomial_coeff(n, outcome)
                    outcome_prob = coeff * np.prod([p ** k for p, k in zip(probs, outcome)])
                    total_prob += outcome_prob
            return [round(total_prob, 15), expectation, variance]
    elif w == 'Geometric':
        if m == 'PMF':
            return [Geometric(k, p, 'p'), Geometric(k, p, 'e'), Geometric(k, p, 'v')]
        elif m == 'CDF':
            return [Geometric(k, p, 'c'), Geometric(k, p, 'e'), Geometric(k, p, 'v')]
    elif w == 'Negative Binomial':
        if m == 'PMF':
            return [negative_Binomial(k, p, n, 'p'), negative_Binomial(k, p, n, 'e'), negative_Binomial(k, p, n, 'v')]
        elif m == 'CDF':
            return [negative_Binomial(k, p, n, 'c'), negative_Binomial(k, p, n, 'e'), negative_Binomial(k, p, n, 'v')]
    elif w == 'Poisson':
        if m == 'PMF':
            return [Poisson(k, p, 'p'), Poisson(k, p, 'e'), Poisson(k, p, 'v')]
        elif m == 'CDF':
            return [Poisson(k, p, 'c'), Poisson(k, p, 'e'), Poisson(k, p, 'v')]
    elif w == 'Normal' and Normal_params is not None:
        x, m, s = Normal_params
        if m == 'PDF':
            return [Normal(x, m, s, 'p'), m, s**2]
        elif m == 'CDF':
            return [Normal(x, m, s, 'c'), m, s**2]
        elif m == 'Quantile':
            q = entry_quantile.get()
            return [Normal(x, m, s, 'c'), m, s**2, Normal_quantile(q,m,s)]
    elif w == 'Uniform Continuous' and uniform_params is not None:
        x, a, b = uniform_params
        print(f"Uniform continuous params: x={x}, a={a}, b={b}")
        if m == 'PDF':
            return [uniform_cont(x, a, b, 'p'), uniform_cont(a, a, b, 'e'), uniform_cont(a, a, b, 'v')]
        elif m == 'CDF':
            return [uniform_cont(x, a, b, 'c'), uniform_cont(a, a, b, 'e'), uniform_cont(a, a, b, 'v')]
        elif m == 'Quantile':
            q = entry_quantile.get()
            return [uniform_cont(x, a, b, 'c'), uniform_cont(a, a, b, 'e'), uniform_cont(a, a, b, 'v'), Normal_quantile(q, uniform_cont(a, a, b, 'e'), np.sqrt(uniform_cont(a, a, b, 'v')))]
    elif w == 'Uniform Discrete':
        if n is not None:
            if m == 'PMF':
                return [uniform_disc(k, n, None, 'p'), uniform_disc(k, n, None, 'e'), uniform_disc(k, n, None, 'v')]
            elif m == 'CDF':
                return [uniform_disc(k, n, None, 'c'), uniform_disc(k, n, None, 'e'), uniform_disc(k, n, None, 'v')]
        else:
            raise ValueError('n must be assigned a value.')
    elif w== 'Exponential' and exp_params is not None:
        x, rate = exp_params
        if m == 'PDF':
            return [exp_dist(x, rate, 'p'), exp_dist(rate, rate, 'e'), exp_dist(rate, rate, 'v')]
        elif m == 'CDF':
            return [exp_dist(x, rate, 'c'), exp_dist(rate, rate, 'e'), exp_dist(rate, rate, 'v')]
        elif m == 'Quantile':
            q = entry_quantile.get()
            return [exp_dist(x, rate, 'c'), exp_dist(rate, rate, 'e'), exp_dist(rate, rate, 'v'), Exponential_quantile(q,rate)]
            
    elif w=='Custom' and custom_params is not None:
        values, probs, val = custom_params
        return [Custom(values, probs, val, 'c'), Custom(values, probs, val, 'e'), Custom(values, probs, val, 'v')]
    else:
        raise ValueError("Invalid value for parameter")
    
    return None

def init_plot():
    global fig, canvas
    fig = Figure(figsize=(20, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(column=1, row=0, padx=10, pady=10, sticky='nsew')

def plot_double_distribution(mode, dist1, dist2, dist1_params=None, dist2_params=None):
    fig.clf()
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=3.0)
    print(f"plot_double_distribution called with plotting mode {mode}")

    if mode == 'CDF Comparison':
        print(f"dist1: {dist1}, dist1_params: {dist1_params}")
        print(f"dist2: {dist2}, dist2_params: {dist2_params}")

        if dist1_params is None or dist2_params is None:
            messagebox.showerror("Input Error", "Parameters for the distributions are incorrect.")
            return

        x_vals = np.linspace(0, 30, 1000)

        if dist1 == 'Normal':
            x1, m1, s1 = dist1_params
            y1 = [Normal(x, m1, s1, 'c') for x in x_vals]
            ax.plot(x_vals, y1, label='Normal Dist 1', color='blue')
        elif dist1 == 'Uniform Continuous':
            x1, a1, b1 = dist1_params
            y1 = [uniform_cont(x, a1, b1, 'c') for x in x_vals]
            ax.plot(x_vals, y1, label='Uniform Dist 1', color='green')
        elif dist1 == 'Exponential':
            x1, rate1 = dist1_params
            y1 = [exp_dist(x, rate1, 'c') for x in x_vals]
            ax.plot(x_vals, y1, label='Exponential Dist 1', color='red')
        elif dist1 == 'Custom':
            values1, probs1 = dist1_params
            y1 = np.cumsum(probs1)
            ax.plot(values1, y1, label='Custom Dist 1', color='blue')

        if dist2 == 'Normal':
            x2, m2, s2 = dist2_params
            y2 = [Normal(x, m2, s2, 'c') for x in x_vals]
            ax.plot(x_vals, y2, label='Normal Dist 2', color='cyan')
        elif dist2 == 'Uniform Continuous':
            x2, a2, b2 = dist2_params
            y2 = [uniform_cont(x, a2, b2, 'c') for x in x_vals]
            ax.plot(x_vals, y2, label='Uniform Dist 2', color='orange')
        elif dist2 == 'Exponential':
            x2, rate2 = dist2_params
            y2 = [exp_dist(x, rate2, 'c') for x in x_vals]
            ax.plot(x_vals, y2, label='Exponential Dist 2', color='magenta')
        elif dist2 == 'Custom':
            values2, probs2 = dist2_params
            y2 = np.cumsum(probs2)
            ax.plot(values2, y2, label='Custom Dist 2', color='red')

        ax.set_title("CDF Comparison")
        ax.set_xlabel("Value")
        ax.set_ylabel("CDF")
        ax.legend()

    canvas.draw()

def plot_distribution(m, n, k, p, mode, mult_params=None, Normal_params=None, uniform_params=None, exp_params=None, custom_params=None):
    fig.clear()
    ax = fig.add_subplot(111)
    print(f"plot_distribution called with mode {mode}")

    if mode == 'Binomial':
        x = np.arange(0, n + 1)
        y = [comb(n, i) * (p**i) * ((1 - p)**(n - i)) for i in x]
        ax.bar(x, y, color='skyblue')
        ax.set_title("Binomial Distribution")
        ax.set_xlabel("Number of Successes")
        ax.set_ylabel("Probability")
        
    elif mode == 'Multinomial' and mult_params is not None:
        categories, probs = mult_params
        outcomes = list(product(*[range(n + 1) for _ in categories]))
        probabilities = [
            mnom(n, outcome, probs, "p")
            for outcome in outcomes
            if sum(outcome) == n
        ]

        outcomes_prob = [(outcome, prob) for outcome, prob in zip(outcomes, probabilities) if prob > 0.01]
        outcomes_prob.sort(key=lambda x: -x[1])
        outcomes_prob = outcomes_prob[:10]
        
        x = np.arange(len(outcomes_prob))
        labels, probabilities = zip(*outcomes_prob)
        colors = ['#d9534f', '#5bc0de', '#5e5e5e', '#f0ad4e', '#5cb85c', '#428bca']
        
        ax.bar(x, probabilities, color=colors[:len(probabilities)], width=0.6)
        
        ax.set_title("Multinomial Distribution")
        ax.set_xlabel("Outcome Index")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(label) for label in labels],
            rotation=45, ha='right', fontsize=8
        )
        
        plt.tight_layout()
        
    elif mode == 'Geometric':
        x = np.arange(1, k + 1)
        if m == 'PMF':
            y = [Geometric(i, p, 'p') for i in x]
        else:
            y = [Geometric(i, p, 'c') for i in x]
        ax.bar(x, y, color='skyblue')
        ax.set_title("Geometric Distribution")
        ax.set_xlabel("Number of Trials")
        ax.set_ylabel("Probability")
    
    elif mode == 'Negative Binomial':
        x = np.arange(n, k + 1)
        if m == 'PMF':
            y = [negative_Binomial(i, p, n, 'p') for i in x]
        else:
            y = [negative_Binomial(i, p, n, 'c') for i in x]
        ax.bar(x, y, color='skyblue')
        ax.set_title("Negative Binomial Distribution")
        ax.set_xlabel("Number of Trials")
        ax.set_ylabel("Probability")

    elif mode == 'Poisson':
        x = np.arange(0, k + 1)
        if m == 'PMF':
            y = [Poisson(i, p, 'p') for i in x]
        else:
            y = [Poisson(i, p, 'c') for i in x]
        ax.bar(x, y, color='skyblue')
        ax.set_title("Poisson Distribution")
        ax.set_xlabel("Number of Events")
        ax.set_ylabel("Probability")
    
    elif mode == 'Normal' and Normal_params is not None:
        #if m == 'Quantile':
            #x,m,s,quantile = Normal_params
        #else:
        x, m, s = Normal_params
        x_vals = np.linspace(m - 4*s, m + 4*s, 100)
        if m == 'PMF':
            y_vals = Normal(x_vals, m, s, 'p')
            ax.plot(x_vals, y_vals, color='skyblue')
        else:
            y_vals = Normal(x_vals, m, s, 'c')
            ax.plot(x_vals, y_vals, color='skyblue')
        ax.set_title("Normal Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")
    
    elif mode == 'Uniform Continuous' and uniform_params is not None:
        #if m == 'Quantile':
        #    x, a, b, quantile = uniform_params
        #else:
        x, a, b = uniform_params
        x_vals = np.linspace(a, b, 100)
        if m == 'PMF':
            y_vals = [uniform_cont(xi, a, b, 'p') for xi in x_vals]
            ax.plot(x_vals, y_vals, color='skyblue')
        else:
            y_vals = [uniform_cont(xi, a, b, 'c') for xi in x_vals]
            ax.plot(x_vals, y_vals, color='skyblue')
        ax.set_title("Uniform Continuous Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")

    elif mode == 'Uniform Discrete':
        x = np.arange(1, n + 1)
        if m == 'PMF':
            y = [uniform_disc(i, n, None, 'p') for i in x]
        else:
            y = [uniform_disc(i, n, None, 'c') for i in x]
        ax.bar(x, y, color='skyblue')
        ax.set_title("Uniform Discrete Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")
        
    elif mode == 'Exponential' and exp_params is not None:
        #if m == 'Quantile':
        #    x, rate, quantile = exp_params
        #else:
        x, rate = exp_params
        x_vals = np.linspace(0, x, 100)
        if m == 'PMF':
            y_vals = [exp_dist(xi, rate, 'p') for xi in x_vals]
            ax.plot(x_vals, y_vals, color='skyblue')
        else:
            y_vals = [exp_dist(xi, rate, 'c') for xi in x_vals]
            ax.plot(x_vals, y_vals, color='skyblue')
        ax.set_title("Exponential Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")
    
    elif mode == 'Custom' and custom_params is not None:
        #if m == 'Quantile':
        #    values1, probs1, val1, quantile = custom_params
        #else:
        values1, probs1, val1 = custom_params
        # Calculate the CDF values
        y1 = np.cumsum(probs1)
        x1 = values1
        
        ax.plot(x1, y1, label='Custom Distribution', color='blue')
        ax.set_xlabel("Value")
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Custom Distribution CDF')
        ax.legend()
    
    canvas.draw()

def update_interface(event):
    mode = combo_w.get()
    
    # Clear all entries and reset labels
    for entry in [entry_n, entry_k, entry_p, entry_categories, entry_probs, entry_x, entry_m, entry_s, entry_a, entry_b, entry_rate, entry_x2, entry_m2, entry_s2, entry_a2, entry_b2, entry_rate2]:
        entry.delete(0, tk.END)
    
    covariance_label.grid_remove()
    
    # Hide all labels and entries initially
    for widget in [label_n, entry_n, label_k, entry_k, label_p, entry_p, label_categories, entry_categories,
                   label_probs, entry_probs, label_x, entry_x, label_m, entry_m, label_s, entry_s,
                   label_a, entry_a, label_b, entry_b, label_rate, entry_rate, label_x2, entry_x2, label_m2, 
                   entry_m2, label_s2, entry_s2, label_a2, entry_a2, label_b2, entry_b2, label_rate2, entry_rate2, label_val1, 
                   entry_val1, label_val2, entry_val2,label_dist1, combo_dist1, label_dist2, combo_dist2, expectation2_label, 
                   variance2_label, result2_label, label_t, combo_t_discrete, combo_t_continuous,
                   label_custom_values1, label_custom_values2, label_custom_probs1, label_custom_probs2, entry_val1, entry_val2, 
                   label_val1, label_val2, entry_custom_probs1, entry_custom_probs2, entry_custom_values1, entry_custom_values2,
                   label_quantile, entry_quantile, zvalue_label, qvalue_label, intercepts_label
                   ]:
        widget.grid_remove()
    
    # Show the necessary widgets based on the selected mode
    
    if mode == 'CDF Comparison':
        label_dist1.config(text='Distribution 1:')
        label_dist2.config(text='Distribution 2:')
        label_dist1.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        combo_dist1.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        label_dist2.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        combo_dist2.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        result_label.config(text='Probability 1')
        expectation_label.config(text='Expectation 1')
        variance_label.config(text='Variance 1')
        result2_label.grid()
        expectation2_label.grid()
        variance2_label.grid()
        intercepts_label.grid()
        
        # Update distribution-specific widgets for the first variable
        update_dist1_widgets(None)
        update_dist2_widgets(None)
        
        # Bind the update function to the first distribution combobox
        combo_dist1.bind('<<ComboboxSelected>>', update_dist1_widgets)
        combo_dist2.bind('<<ComboboxSelected>>', update_dist2_widgets)
    elif mode == 'Custom':
        label_custom_values1.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        entry_custom_values1.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        label_custom_probs1.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        entry_custom_probs1.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        label_val1.grid(row=7, column=0, padx=5, pady=5, sticky='w')
        entry_val1.grid(row=7, column=1, padx=5, pady=5, sticky='w')
    else:   
        label_t.grid()
        if mode == 'Binomial':
            label_n.config(text="Number of trials (n):")
            label_k.config(text="Number of successes (k):")
            label_p.config(text="Probability of success (p):")
            label_n.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_n.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_k.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_k.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            label_p.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            entry_p.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            combo_t_discrete.grid()
        elif mode =='Multinomial':
            label_n.config(text="Number of trials (n):")
            label_categories.config(text="Event Number (categories):")
            label_probs.config(text="Probabilities (probs):")
            label_n.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_n.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_categories.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_categories.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            label_probs.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            entry_probs.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            combo_t_discrete.grid()
        elif mode =='Geometric':
            label_k.config(text="Number of trials (k):")
            label_p.config(text="Probability of success (p):")
            label_k.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_k.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_p.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_p.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            combo_t_discrete.grid()
        elif mode =='Negative Binomial':
            label_n.config(text="Number of successes (r):")
            label_k.config(text="Number of trials (k):")
            label_p.config(text="Probability of success (p):")
            label_n.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_n.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_k.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_k.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            label_p.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            entry_p.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            combo_t_discrete.grid()
        elif mode == 'Poisson':
            label_k.config(text="Number of events (k):")
            label_p.config(text="Rate (λ):")
            label_k.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_k.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_p.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_p.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            combo_t_discrete.grid()
        elif mode == 'Normal':
            label_x.config(text="Value (x):")
            label_x.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_x.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_m.config(text="Mean (μ):")
            label_m.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_m.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            label_s.config(text="Standard Deviation (σ):")
            label_s.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            entry_s.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            combo_t_continuous.grid()
            if combo_t_continuous.get() == 'Quantile':
                label_quantile.grid(row = 8, column = 0, padx = 5, pady = 5, sticky='w')
                entry_quantile.grid(row=8, column=1, padx = 5, pady = 5, sticky='w')
                zvalue_label.grid() 
                qvalue_label.grid()
        elif mode == 'Uniform Continuous':
            label_x.config(text="Value (x):")
            label_a.config(text="Lower bound (a):")
            label_b.config(text="Upper bound (b):")
            label_x.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_x.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_a.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_a.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            label_b.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            entry_b.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            combo_t_continuous.grid()
            if combo_t_continuous.get() == 'Quantile':
                label_quantile.grid(row = 8, column = 0, padx = 5, pady = 5, sticky='w')
                entry_quantile.grid(row=8, column=1, padx = 5, pady = 5, sticky='w')
                zvalue_label.grid() 
                qvalue_label.grid()
        elif mode == 'Uniform Discrete':
            label_n.config(text="Number of values (n):")
            label_k.config(text="Value (k):")
            label_n.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_n.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_k.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_k.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            combo_t_discrete.grid()
        elif mode == 'Exponential':
            label_x.config(text="Value (x):")
            label_rate.config(text="Rate (λ):")
            label_x.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            entry_x.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            label_rate.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            entry_rate.grid(row=6, column=1, padx=5, pady=5, sticky='w')
            combo_t_continuous.grid()
            if combo_t_continuous.get() == 'Quantile':
                label_quantile.grid(row = 7, column = 0, padx = 5, pady = 5, sticky='w')
                entry_quantile.grid(row=7, column=1, padx = 5, pady = 5, sticky='w')
                zvalue_label.grid() 
                qvalue_label.grid()      

    calculate_button.grid()  
    result_label.grid()
    expectation_label.grid()
    variance_label.grid()

def update_dist1_widgets(event):
    dist1 = combo_dist1.get()
    
    # Hide all specific widgets initially
    for widget in [label_x, entry_x, label_m, entry_m, label_s, entry_s, label_a, entry_a, label_b, entry_b, label_rate, entry_rate, label_val1, entry_val1, label_custom_values1, entry_custom_values1, label_custom_probs1, entry_custom_probs1]:
        widget.grid_remove()
    
    if dist1 == 'Normal':
        label_x.config(text='Value 1:')
        label_x.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        entry_x.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        label_m.config(text='Mean 1:')
        label_m.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        entry_m.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        label_s.config(text='Standard deviation 1:')
        label_s.grid(row=7, column=0, padx=5, pady=5, sticky='w')
        entry_s.grid(row=7, column=1, padx=5, pady=5, sticky='w')
    elif dist1 == 'Uniform Continuous':
        label_x.config(text='Value 1:')
        label_x.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        entry_x.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        label_a.config(text='Lower bound 1:')
        label_a.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        entry_a.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        label_b.config(text='Upper bound 1:')
        label_b.grid(row=7, column=0, padx=5, pady=5, sticky='w')
        entry_b.grid(row=7, column=1, padx=5, pady=5, sticky='w')
    elif dist1 == 'Exponential':
        label_x.config(text='Value 1:')
        label_x.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        entry_x.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        label_rate.config(text='Rate 1:')
        label_rate.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        entry_rate.grid(row=6, column=1, padx=5, pady=5, sticky='w')
    elif dist1 == 'Custom':
        label_custom_values1.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        entry_custom_values1.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        label_custom_probs1.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        entry_custom_probs1.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        label_val1.grid(row=7, column=0, padx=5, pady=5, sticky='w')
        entry_val1.grid(row=7, column=1, padx=5, pady=5, sticky='w')

def update_dist2_widgets(event):
    dist2 = combo_dist2.get()
    
    # Hide all specific widgets initially
    for widget in [label_x2, entry_x2, label_m2, entry_m2, label_s2, entry_s2, label_a2, entry_a2, label_b2, entry_b2, label_rate2, entry_rate2, label_val2, entry_val2, label_custom_values2, entry_custom_values2, label_custom_probs2, entry_custom_probs2]:
        widget.grid_remove()

    if dist2 == 'Normal':
        label_x2.config(text='Value 2:')
        label_x2.grid(row=10, column=0, padx=5, pady=5, sticky='w')
        entry_x2.grid(row=10, column=1, padx=5, pady=5, sticky='w')
        label_m2.config(text='Mean 2:')
        label_m2.grid(row=11, column=0, padx=5, pady=5, sticky='w')
        entry_m2.grid(row=11, column=1, padx=5, pady=5, sticky='w')
        label_s2.config(text='Standard deviation 2:')
        label_s2.grid(row=12, column=0, padx=5, pady=5, sticky='w')
        entry_s2.grid(row=12, column=1, padx=5, pady=5, sticky='w')
    elif dist2 == 'Uniform Continuous':
        label_x2.config(text='Value 2:')
        label_x2.grid(row=10, column=0, padx=5, pady=5, sticky='w')
        entry_x2.grid(row=10, column=1, padx=5, pady=5, sticky='w')
        label_a2.config(text='Lower bound 2:')
        label_a2.grid(row=11, column=0, padx=5, pady=5, sticky='w')
        entry_a2.grid(row=11, column=1, padx=5, pady=5, sticky='w')
        label_b2.config(text='Upper bound 2:')
        label_b2.grid(row=12, column=0, padx=5, pady=5, sticky='w')
        entry_b2.grid(row=12, column=1, padx=5, pady=5, sticky='w')
    elif dist2 == 'Exponential':
        label_x2.config(text='Value 2:')
        label_x2.grid(row=10, column=0, padx=5, pady=5, sticky='w')
        entry_x2.grid(row=10, column=1, padx=5, pady=5, sticky='w')
        label_rate2.config(text='Rate 2:')
        label_rate2.grid(row=11, column=0, padx=5, pady=5, sticky='w')
        entry_rate2.grid(row=11, column=1, padx=5, pady=5, sticky='w')
    elif dist2 == 'Custom':
        label_custom_values2.grid(row=10, column=0, padx=5, pady=5, sticky='w')
        entry_custom_values2.grid(row=10, column=1, padx=5, pady=5, sticky='w')
        label_custom_probs2.grid(row=11, column=0, padx=5, pady=5, sticky='w')
        entry_custom_probs2.grid(row=11, column=1, padx=5, pady=5, sticky='w')
        label_val2.grid(row=12, column=0, padx=5, pady=5, sticky='w')
        entry_val2.grid(row=12, column=1, padx=5, pady=5, sticky='w')

def calculate():
    try:
        w = combo_w.get()
        covariance = []

        print(f"Selected mode: {w}")

        if w == 'CDF Comparison':
            val1 = entry_val1.get()
            val1 = float(val1) if val1 else None
            val2 = entry_val2.get()
            val2 = float(val2) if val2 else None
            dist1 = combo_dist1.get()
            dist2 = combo_dist2.get()

            dist1_params = dist2_params = None

            if dist1 == 'Normal':
                x1 = float(entry_x.get())
                m1 = float(entry_m.get())
                s1 = float(entry_s.get())
                dist1_params = (x1, m1, s1)
            elif dist1 == 'Uniform Continuous':
                x1 = float(entry_x.get())
                a1 = float(entry_a.get())
                b1 = float(entry_b.get())
                dist1_params = (x1, a1, b1)
            elif dist1 == 'Exponential':
                x1 = float(entry_x.get())
                rate1 = float(entry_rate.get())
                dist1_params = (x1, rate1)
            elif dist1 == 'Custom':
                values1 = list(map(float, entry_custom_values1.get().split(',')))
                probs1 = list(map(float, entry_custom_probs1.get().split(',')))
                dist1_params = (values1, probs1)
            if dist2 == 'Normal':
                x2 = float(entry_x2.get())
                m2 = float(entry_m2.get())
                s2 = float(entry_s2.get())
                dist2_params = (x2, m2, s2)
            elif dist2 == 'Uniform Continuous':
                x2 = float(entry_x2.get())
                a2 = float(entry_a2.get())
                b2 = float (entry_b2.get())
                dist2_params = (x2, a2, b2)
            elif dist2 == 'Exponential':
                x2 = float(entry_x2.get())
                rate2 = float(entry_rate2.get())
                dist2_params = (x2, rate2)
            elif dist2 == 'Custom':
                values2 = list(map(float, entry_custom_values2.get().split(',')))
                probs2 = list(map(float, entry_custom_probs2.get().split(',')))
                dist2_params = (values2, probs2)

            if dist1_params is None or dist2_params is None:
                raise ValueError("Parameters for the selected distributions are not properly set.")

            # Debugging output
            print(f"dist1: {dist1}, dist1_params: {dist1_params}")
            print(f"dist2: {dist2}, dist2_params: {dist2_params}")

            cdf1, expectation1, variance1, cdf2, expectation2, variance2 = comparator(dist1, dist2, dist1_params, dist2_params, val1, val2)

            print(f"CDF1: {cdf1}, Expectation1: {expectation1}, Variance1: {variance1}")
            print(f"CDF2: {cdf2}, Expectation2: {expectation2}, Variance2: {variance2}")

            print(f"Types - CDF1: {type(cdf1)}, Expectation1: {type(expectation1)}, Variance1: {type(variance1)}")
            print(f"Types - CDF2: {type(cdf2)}, Expectation2: {type(expectation2)}, Variance2: {type(variance2)}")

            result_label.config(text=f"Probability 1: {cdf1}")
            expectation_label.config(text=f"Expectation 1: {expectation1}")
            variance_label.config(text=f"Variance 1: {variance1}")

            result2_label.config(text=f"Probability 2: {cdf2}")
            expectation2_label.config(text=f"Expectation 2: {expectation2}")
            variance2_label.config(text=f"Variance 2: {variance2}")

            x_vals = np.linspace(0, 30, 10000)

            intersections = find_intersections(dist1, dist1_params, dist2, dist2_params, x_vals)
            print(f"Intercepts: {intersections}")
            intercepts_label.config(text=f'Intercepts: {intersections}')

            plot_double_distribution('CDF Comparison', dist1, dist2, dist1_params=dist1_params, dist2_params=dist2_params)
        else: 
            if w in ['Uniform Continuous', 'Exponential', 'Normal']:
               m = combo_t_continuous.get()
            else:
                m = combo_t_discrete.get()

            if w == 'Binomial':
                n = int(entry_n.get())
                k = int(entry_k.get())
                p = float(entry_p.get())
                print(f"n: {n}, k: {k}, p: {p}")
                result = distributor(m, n, k, p, w)
                plot_distribution(m, n, k, p, w)
            elif w == 'Multinomial':
                n = int(entry_n.get())
                categories = list(map(int, entry_categories.get().split(',')))
                probs = list(map(float, entry_probs.get().split(',')))
                mult_params = (categories, probs)
                print(f"n: {n}, categories: {categories}, probs: {probs}")
                result = distributor(m, n, None, None, w, mult_params)
                covariance = mnom(n, categories, probs, 'cv')
                plot_distribution(m, n, None, None, w, mult_params)
                cov_str = ', '.join(
                    [f"Cov(X{i+1}, X{j+1}) = {round(cov, 4)}" for (cov, (i, j)) in zip(covariance, combinations(range(len(probs)), 2))]
                )
                covariance_label.config(text=f"Covariance: {cov_str}")
                covariance_label.grid()
            elif w== 'Geometric':
                k = int(entry_k.get())
                p = float(entry_p.get())
                print(f"k: {k}, p: {p}")
                result = distributor(m,None, k, p, w)
                plot_distribution(m,None, k, p, w)
            elif w == 'Negative Binomial':
                r = int(entry_n.get())  # Number of successes
                k = int(entry_k.get())
                p = float(entry_p.get())
                print(f"r: {r}, k: {k}, p: {p}")
                result = distributor(m, r, k, p, w)
                plot_distribution(m, r, k, p, w)
            elif w == 'Poisson':
                k = int(entry_k.get())
                l = float(entry_p.get())
                print(f"k: {k}, l: {l}")
                result = distributor(m, None, k, l, w)
                plot_distribution(m, None, k, l, w)
            elif w == 'Normal':
                x = float(entry_x.get())
                me = float(entry_m.get())
                s = float(entry_s.get())
                Normal_params = (x, me, s)
                print(f"x: {x}, m: {me}, s: {s}")
                result = distributor(m, None, None, None, w, Normal_params=Normal_params)
                plot_distribution(m, None, None, None, w, Normal_params=Normal_params)
            elif w == 'Uniform Continuous':
                x = float(entry_x.get())
                a = float(entry_a.get())
                b = float(entry_b.get())
                uniform_params = (x, a, b)
                print(f"x: {x}, a: {a}, b: {b}")
                result = distributor(m, None, None, None, w, uniform_params=uniform_params)
                print(result)
                plot_distribution(m, None, None, None, w, uniform_params=uniform_params)
            elif w == 'Uniform Discrete':
                n = int(entry_n.get())
                k = int(entry_k.get())
                print(f"n: {n}, k: {k}")
                result = distributor(m, n, k, None, w)
                plot_distribution(m, n, k, None, w)
            elif w == 'Exponential':
                x = float(entry_x.get())
                rate = float(entry_rate.get())
                exp_params = (x, rate)
                print(f"x: {x}, rate: {rate}")
                result = distributor(m, None, None, None, w, exp_params=exp_params)
                plot_distribution(m, None, None, None, w, exp_params=exp_params)
            elif w == 'Custom':
                values1 = list(map(float, entry_custom_values1.get().split(',')))
                probs1 = list(map(float, entry_custom_probs1.get().split(',')))
                val1 = entry_val1.get()
                if val1 != '':
                    val1 = float(entry_val1.get())
                    print(f'Values: {values1}, Probabilities: {probs1}, CDF threshold: {val1}')
                else:
                    print('No CDF threshold given, assigning threshold.')
                    val1 = max(values1)
                custom_params = (values1, probs1, val1)
                result= distributor(None, None, None, None, w, custom_params=custom_params)
                plot_distribution(None, None, None, None, w, custom_params=custom_params)
            else:
                raise ValueError("Invalid mode selected")
            
            print(result)

            if m == 'Quantile':
                probability, expectation, variance, quantile = result
                zval = normalizer(quantile, expectation, np.sqrt(quantile))
                print(f"Probability: {probability}, Expectation: {expectation}, Variance: {variance}, Quantile: {quantile}")
                result_label.config(text=f"Probability: {probability}")
                expectation_label.config(text=f"Expectation: {expectation}")
                variance_label.config(text=f"Variance: {variance}")
                qvalue_label.config(text=f'Quantile Value: {quantile}')
                zvalue_label.config(text=f'Z-Value: {zval}')
            else:
                probability, expectation, variance = result
                print(f"Probability: {probability}, Expectation: {expectation}, Variance: {variance}")
                result_label.config(text=f"Probability: {probability}")
                expectation_label.config(text=f"Expectation: {expectation}")
                variance_label.config(text=f"Variance: {variance}")
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

def on_resize(event):
    new_size = min(max(int(min(root.winfo_width(), root.winfo_height()) / 60), 10), 20)
    new_font = ('Helvetica', new_size)
    for widget in widgets:
        widget.config(font=new_font)
    
    # Set the figure size based on the window size
    width, height = event.width / 100, event.height / 100
    fig.set_size_inches(width, height)
    
    # Adjust the subplots with larger margins
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    canvas.draw()

root = tk.Tk()
root.title("Distribution Calculator")

style = ttk.Style()
style.theme_use('alt')  # Choose one of the available themes

frame = ttk.Frame(root, padding="10")
frame.grid(column=0, row=0, sticky='nsw')

widgets = []

label_w = ttk.Label(frame, text="Mode:")
label_w.grid(column=0, row=0, padx=5, pady=5, sticky='w')
widgets.append(label_w)

combo_w = ttk.Combobox(frame, values=[
    'Binomial', 'Multinomial', 'Geometric', 'Negative Binomial', 'Poisson', 'Normal', 'Uniform Continuous', 'Uniform Discrete', 'Exponential', 'Custom', 'CDF Comparison'
])

combo_w.grid(column=1, row=0, padx=5, pady=5, sticky='w')
combo_w.current(0)
combo_w.bind('<<ComboboxSelected>>', update_interface)
widgets.append(combo_w)

label_t = ttk.Label(frame, text="Distribution Type:")
label_t.grid(column=0, row=1, padx=5, pady=5, sticky='w')
widgets.append(label_t)

combo_t_discrete = ttk.Combobox(frame, values= ['PMF', 'CDF'])
combo_t_discrete.grid(column=1, row=1, padx=5, pady=5, sticky='w')
combo_t_discrete.current(0)
combo_t_discrete.bind('<<ComboboxSelected>>', update_interface)
widgets.append(combo_t_discrete)

combo_t_continuous = ttk.Combobox(frame, values= ['PDF', 'CDF', 'Quantile'])
combo_t_continuous.grid(column=1, row=1, padx=5, pady=5, sticky='w')
combo_t_continuous.current(0)
combo_t_continuous.bind('<<ComboboxSelected>>', update_interface)
widgets.append(combo_t_continuous)

label_n = ttk.Label(frame, text="Number of trials:")
label_n.grid(column=0, row=2, padx=5, pady=5, sticky='w')
widgets.append(label_n)

entry_n = ttk.Entry(frame)
entry_n.grid(column=1, row=2, padx=5, pady=5, sticky='w')
widgets.append(entry_n)

label_k = ttk.Label(frame, text="Number of successes:")
label_k.grid(column=0, row=3, padx=5, pady=5, sticky='w')
widgets.append(label_k)

entry_k = ttk.Entry(frame)
entry_k.grid(column=1, row=3, padx=5, pady=5, sticky='w')
widgets.append(entry_k)

label_p = ttk.Label(frame, text="Probability of success:")
label_p.grid(column=0, row=4, padx=5, pady=5, sticky='w')
widgets.append(label_p)

entry_p = ttk.Entry(frame)
entry_p.grid(column=1, row=4, padx=5, pady=5, sticky='w')
widgets.append(entry_p)

label_categories = ttk.Label(frame, text="Event Number:")
label_categories.grid(column=0, row=5, padx=5, pady=5, sticky='w')
widgets.append(label_categories)

entry_categories = ttk.Entry(frame)
entry_categories.grid(column=1, row=5, padx=5, pady=5, sticky='w')
widgets.append(entry_categories)

label_probs = ttk.Label(frame, text="Probabilities:")
label_probs.grid(column=0, row=6, padx=5, pady=5, sticky='w')
widgets.append(label_probs)

entry_probs = ttk.Entry(frame)
entry_probs.grid(column=1, row=6, padx=5, pady=5, sticky='w')
widgets.append(entry_probs)

label_x = ttk.Label(frame, text="Value (x):")
label_x.grid(column=0, row=5, padx=5, pady=5, sticky='w')
widgets.append(label_x)

entry_x = ttk.Entry(frame)
entry_x.grid(column=1, row=5, padx=5, pady=5, sticky='w')
widgets.append(entry_x)

label_m = ttk.Label(frame, text="Mean (μ):")
label_m.grid(column=0, row=6, padx=5, pady=5, sticky='w')
widgets.append(label_m)

entry_m = ttk.Entry(frame)
entry_m.grid(column=1, row=6, padx=5, pady=5, sticky='w')
widgets.append(entry_m)

label_s = ttk.Label(frame, text="Standard Deviation:")
label_s.grid(column=0, row=7, padx=5, pady=5, sticky='w')
widgets.append(label_s)

entry_s = ttk.Entry(frame)
entry_s.grid(column=1, row=7, padx=5, pady=5, sticky='w')
widgets.append(entry_s)

label_a = ttk.Label(frame, text="Lower bound (a):")
label_a.grid(column=0, row=8, padx=5, pady=5, sticky='w')
widgets.append(label_a)

entry_a = ttk.Entry(frame)
entry_a.grid(column=1, row=8, padx=5, pady=5, sticky='w')
widgets.append(entry_a)

label_b = ttk.Label(frame, text="Upper bound (b):")
label_b.grid(column=0, row=9, padx=5, pady=5, sticky='w')
widgets.append(label_b)

entry_b = ttk.Entry(frame)
entry_b.grid(column=1, row=9, padx=5, pady=5, sticky='w')
widgets.append(entry_b)

label_rate = ttk.Label(frame, text="Rate:")
label_rate.grid(column=0, row=8, padx=5, pady=5, sticky='w')
widgets.append(label_rate)

entry_rate = ttk.Entry(frame)
entry_rate.grid(column=1, row=8, padx=5, pady=5, sticky='w')
widgets.append(entry_rate)

label_x2 = ttk.Label(frame, text="Value (x):")
widgets.append(label_x2)

entry_x2 = ttk.Entry(frame)
widgets.append(entry_x2)

label_m2 = ttk.Label(frame, text="Mean (μ):")
widgets.append(label_m2)

entry_m2 = ttk.Entry(frame)
widgets.append(entry_m2)

label_s2 = ttk.Label(frame, text="Standard Deviation:")
widgets.append(label_s2)

entry_s2 = ttk.Entry(frame)
widgets.append(entry_s2)

label_a2 = ttk.Label(frame, text="Lower bound (a):")
widgets.append(label_a2)

entry_a2 = ttk.Entry(frame)
widgets.append(entry_a2)

label_b2 = ttk.Label(frame, text="Upper bound (b):")
widgets.append(label_b2)

entry_b2 = ttk.Entry(frame)
widgets.append(entry_b2)

label_rate2 = ttk.Label(frame, text="Rate:")
widgets.append(label_rate2)

entry_rate2 = ttk.Entry(frame)
widgets.append(entry_rate2)

label_dist1 = ttk.Label(frame, text="Distribution 1:")
label_dist1.grid(column=0, row=3, padx=5, pady=5, sticky='w')
widgets.append(label_dist1)

combo_dist1 = ttk.Combobox(frame, values=[
    'Normal', 'Uniform Continuous', 'Exponential','Custom'
])

combo_dist1.grid(column=1, row=3, padx=5, pady=5, sticky='w')
combo_w.current(0)
combo_w.bind('<<ComboboxSelected>>', update_interface)
widgets.append(combo_dist1)


label_dist2 = ttk.Label(frame, text="Distribution 2:")
label_dist2.grid(column=0, row=4, padx=5, pady=5, sticky='w')
widgets.append(label_dist2)

combo_dist2 = ttk.Combobox(frame, values=[
    'Normal', 'Uniform Continuous', 'Exponential','Custom'
])

combo_dist2.grid(column=1, row=4, padx=5, pady=5, sticky='w')
combo_w.current(0)
combo_w.bind('<<ComboboxSelected>>', update_interface)
widgets.append(combo_dist2)

label_val1 = ttk.Label(frame, text = "CDF Threshold 1:")
widgets.append(label_val1)

entry_val1 = ttk.Entry(frame)
widgets.append(entry_val1)

label_val2 = ttk.Label(frame, text = "CDF Threshold 2:")
widgets.append(label_val2)

entry_val2 = ttk.Entry(frame)
widgets.append(entry_val2)

label_custom_values1 = ttk.Label(frame, text="Values 1:")
widgets.append(label_custom_values1)

label_custom_probs1 = ttk.Label(frame, text="Probabilities 1:")
widgets.append(label_custom_probs1)

entry_custom_values1 = ttk.Entry(frame)
widgets.append(entry_custom_values1)

entry_custom_probs1 = ttk.Entry(frame)
widgets.append(entry_custom_probs1)

label_custom_values2 = ttk.Label(frame, text="Values 2:")
widgets.append(label_custom_values2)

label_custom_probs2 = ttk.Label(frame, text="Probabilities 2:")
widgets.append(label_custom_probs2)

entry_custom_values2 = ttk.Entry(frame)
widgets.append(entry_custom_values2)

entry_custom_probs2 = ttk.Entry(frame)
widgets.append(entry_custom_probs2)

label_quantile = ttk.Label(frame, text='Quantile:')
widgets.append(label_quantile)

entry_quantile = ttk.Entry(frame)
widgets.append(entry_quantile)

calculate_button = tk.Button(frame, text="Calculate", command=calculate)
calculate_button.grid(column=0, row=18, columnspan=2, pady=10)  
widgets.append(calculate_button)

result_label = ttk.Label(frame, text="Probability:")
result_label.grid(column=0, row=19, columnspan=2, sticky='w') 
widgets.append(result_label)

result2_label = ttk.Label(frame, text="Probability 2:")
result2_label.grid(column=0, row=20, columnspan=2, sticky='w') 
widgets.append(result2_label)

expectation_label = ttk.Label(frame, text="Expectation:")
expectation_label.grid(column=0, row=21, columnspan=2, sticky='w')  
widgets.append(expectation_label)

expectation2_label = ttk.Label(frame, text="Expectation 2:")
expectation2_label.grid(column=0, row=22, columnspan=2, sticky='w')  
widgets.append(expectation2_label)

variance_label = ttk.Label(frame, text="Variance:")
variance_label.grid(column=0, row=23, columnspan=2, sticky='w')  
widgets.append(variance_label)

variance2_label = ttk.Label(frame, text="Variance 2:")
variance2_label.grid(column=0, row=24, columnspan=2, sticky='w')  
widgets.append(variance2_label)

covariance_label = ttk.Label(frame, text='Covariance:')
covariance_label.grid(column=0, row=25, columnspan=2, sticky='w')  

qvalue_label = ttk.Label(frame, text = 'Quantile Value:')
qvalue_label.grid(column=0, row=26, columnspan=2, sticky='w')
widgets.append(qvalue_label)

zvalue_label = ttk.Label(frame, text = 'Z-Value:')
zvalue_label.grid(column=0, row=27, columnspan=2, sticky='w')
widgets.append(zvalue_label)

intercepts_label = ttk.Label(frame, text = 'Intercepts:')
intercepts_label.grid(column=0, row=28, columnspan=2, sticky='w')
widgets.append(intercepts_label)


fig = Figure(figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(column=1, row=0, padx=10, pady=10, sticky='nsew')

root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(0, weight=1)

root.bind('<Configure>', on_resize)
# Bind the update interface function to the combobox selection event
combo_w.bind('<<ComboboxSelected>>', update_interface)
combo_dist1.bind('<<ComboboxSelected>>', update_dist1_widgets)
combo_dist2.bind('<<ComboboxSelected>>', update_dist2_widgets)

# Initialize the interface for the default selected mode
update_interface(None)

init_plot()

root.mainloop()

