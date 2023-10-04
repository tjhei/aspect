import numpy as np
import matplotlib.pyplot as plt
import pymc
import json
import seaborn as sns
import arviz as az

basic_model = pymc.Model()

true_alpha = 3e-5
true_thickness = 140e3

minerr = 1e100

def get_error(thickness, alpha):
    return np.sqrt((thickness-true_thickness)**2 + (alpha-true_alpha)**2)

with basic_model:
    # Priors for unknown model parameters
    test_thickness_values = pymc.Uniform("test_thickness_values", lower=50e3, upper=150e3)
    test_alpha_values     = pymc.Uniform("test_alpha_values", lower=3e-5, upper=5e-5)

    print(test_thickness_values.eval())

    def theta(test_thickness_values=test_thickness_values, test_alpha_values=test_alpha_values):
        global minerr

        try:
            e = get_error(test_thickness_values.eval(), test_alpha_values.eval())
            if e < minerr:
                minerr = e
                print("best fit", e, "values:", test_thickness_values, test_alpha_values)
            return e
        except ValueError:
            return 1e20  # float("inf")

    sig = 10.0
    # this is the sampling distribution of the output (residual/misfit)
    misfit = pymc.Normal("d", mu=theta(test_thickness_values, test_alpha_values), tau=1.0 / (sig * sig), observed=True)
    
    # fitting model here
    model_estimate = pymc.find_MAP(model=basic_model)
    step  = pymc.Slice() 
    trace = pymc.sample(5000, step=step, return_inferencedata=False)

with basic_model:
    az.plot_trace(trace, var_names=["test_thickness_values", "test_alpha_values"])