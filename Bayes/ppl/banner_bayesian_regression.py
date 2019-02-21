import os
from functools import partial
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

import json

from datetime import datetime
from dateutil import parser


"""
 Data:
 focusing on three features from the dataset: 

"""
def read_data():
    with open('bannerData.json') as json_file:
        data = json.load(json_file)
        df = [( d['converted'], np.float(d['time']), d['condition']=='green') for d in data]
        df_banner = pd.DataFrame(np.array(df).reshape(len(df), 3), columns = ['converted', 'time', 'condition'])

    df_banner["logtime"] = np.log(1.+df_banner["time"])
    # df_banner["logtime"] = list(map(lambda s: np.log(1+np.float(s)), df_banner["time"]))


    # catherorical: condition HOME_PAGE
    df_banner["condition"] = (df_banner['condition']).astype(int)
    df_banner["converted"] = (df_banner['converted']).astype(int)
    print(df_banner.head(10))

    df = df_banner[["logtime", "condition", "converted"]]
    return df


def plot_data(data):
    #
    # plot two data sets
    #
    group1 = data[data["condition"] == 1]
    group2 = data[data["condition"] == 0]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    ax[0].scatter(group2["logtime"], (group2["converted"]))
    ax[0].set(xlabel="logtime",
              ylabel="converted",
              title="group1")
    ax[1].scatter(group1["logtime"], (group1["converted"]))
    ax[1].set(xlabel="logtime",
              ylabel="converted",
              title="group1")
    # set axes range
    #plt.xlim(-2, 2)
    plt.ylim(0, 5)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    ax[0].hist(group1["logtime"], color='blue', edgecolor='black', bins=10)
    ax[1].hist(group2["logtime"], color='blue', edgecolor='black', bins=10)

    plt.show()


"""
 Include an extra self.factor term meant to capture the correlation 
 between x[0] and boolean x[1].
"""


# Model-1
class RegressionModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        # p = number of features
        assert num_classes == 2

        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # (input_size, num_outputs)
        # add factor
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        y = self.linear(x) + (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)
        return torch.sigmoid(y)


# Model-2
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size,  num_classes)
        # add factor
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        out = self.linear(x) + (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)
        return out


# training data
def main(x_data, y_data, num_iterations, model, is_logit=True):

    if is_logit:
        optim = torch.optim.SGD(model.parameters(), lr=0.01)  # 70%=0.01) # 50%=0.001)
        criterion = torch.nn.CrossEntropyLoss()  # computes softmax internallyΩ
    else:
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        # criterion = torch.nn.NLLLoss()  # size_average=True
        # criterion = torch.nn.MSELoss(reduction='sum')

    for j in range(num_iterations):
        # run the model forward on the data
        if is_logit:
            y_pred = model(x_data).squeeze(-1)
            # calculate the log loss, when forward computes regression
            loss = criterion(y_pred, y_data)  # (outputs, labels)
        else:
            y_pred = model(x_data)
            # loss = criterion(input=y_pred, target=y_data)
            loss = nn.functional.binary_cross_entropy(input=y_pred.squeeze(-1), target=y_data)

        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("\n [iteration %04d] loss: %.4f  " % (j + 1, loss.item() ))
            if is_logit:
                outputs = model(x_data)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y_data).sum()
                print('Accuracy of logistic model: {} %'.format(100 * correct / y_data.size(0)))
            else:
                y_pred = model(x_data).squeeze(-1)
                error_mean = ((y_pred > 0.5).float() - y_data_float).abs().mean()
                print('Accuracy of regression model: {} %'.format(100.-100.*error_mean))

    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in model.named_parameters():
        print(name, param.data.numpy())

# Data
df_data = read_data()

print(df_data.head(5))

plot_data(df_data)

print(print(df_data.describe()))

# Train torch model
x_data = torch.tensor(df_data[['logtime', 'condition']].values, dtype=torch.float)
x_data = x_data/x_data.max()
y_data = torch.tensor(df_data[['converted']].values, dtype=torch.long).squeeze(-1)
y_data_float = y_data.float()

smoke_test = ('CI' in os.environ)
num_iterations = 1000 if not smoke_test else 2

regression_model = RegressionModel(x_data.shape[1], num_classes=2)
logistic_model = LogisticRegression(x_data.shape[1], num_classes=2)

# logistic regression
main(x_data, y_data, num_iterations, model=logistic_model)
# Test the model: don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    outputs = logistic_model(x_data)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y_data).sum()
    print('Accuracy of the model : {} %'.format(100 * correct / y_data.size(0)))
# Save the model checkpoint
torch.save(logistic_model.state_dict(), 'logistic_model.ckpt')

# nonlinear regression
main(x_data, y_data_float, num_iterations, model=regression_model, is_logit=False)
# Test the model don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    y_pred = regression_model(x_data).squeeze(-1)
    error_mean = ((y_pred > 0.5).float() - y_data_float).abs().mean()
    print('Accuracy of the model: {} %'.format(100. - 100. * error_mean))
    # Save the model checkpoint
torch.save(regression_model.state_dict(), 'regression_model.ckpt')


#
#
# Pyro:
#

"""
Bayesian modeling offers a systematic framework for reasoning about model uncertainty. 
Instead of just learning point estimates, we’re going to learn a distribution over 
variables that are consistent with the observed data.
"""

"""
each parameter in the original regression model is sampled from the provided prior. 
This allows us to repurpose vanilla regression models for use in the Bayesian setting.
"""
"""
loc = torch.zeros(1, 1)
scale = torch.ones(1, 1)
# define a unit normal prior
prior = Normal(loc, scale)
# overload the parameters in the regression module with samples from the prior
lifted_module = pyro.random_module("regression_module", nn, prior)
# sample a nn from the prior
sampled_reg_model = lifted_module()
"""



def model(x_data, y_data, regression_model):
    p = x_data.shape[1]
    # weight and bias priors
    # w_prior = Normal(torch.zeros(1, 2), torch.ones(1, 2)).to_event(1)
    # b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    w_prior = Normal(torch.zeros(1, p), torch.ones(1, p)).to_event(1)
    b_prior = Normal(torch.tensor([[1.]]), torch.tensor([[10.]])).to_event(1)

    f_prior = Normal(0., 1.)

    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}

    scale = pyro.sample("sigma", Uniform(0., 10.))

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean


def train(svi, x_data, y_data, num_iterations, regression_model):
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data, regression_model)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(y_data)))


def wrapped_model(x_data, y_data, regression_model):
    pyro.sample("prediction", Delta(model(x_data, y_data, regression_model)))


def pyro_bayesian(regression_model, y_data):

    def summary(traces, sites):
        marginal = get_marginal(traces, sites)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1) \
                [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats

    # CI testing
    assert pyro.__version__.startswith('0.3.0')
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)
    pyro.enable_validation(True)

    from pyro.contrib.autoguide import AutoDiagonalNormal
    guide = AutoDiagonalNormal(model)

    optim = Adam({"lr": 0.03})
    svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

    train(svi, x_data, y_data, num_iterations, regression_model)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

    posterior = svi.run(x_data, y_data, regression_model)


    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=1000)
    post_pred = trace_pred.run(x_data, None, regression_model)
    post_summary = summary(post_pred, sites= ['prediction', 'obs'])
    mu = post_summary["prediction"]
    y = post_summary["obs"]
    predictions = pd.DataFrame({
        "x0": x_data[:, 0],
        "x1": x_data[:, 1],
        "mu_mean": mu["mean"],
        "mu_perc_5": mu["5%"],
        "mu_perc_95": mu["95%"],
        "y_mean": y["mean"],
        "y_perc_5": y["5%"],
        "y_perc_95": y["95%"],
        "true_gdp": y_data,
    })
    # print("predictions=", predictions)

    """we need to prepend `module$$$` to all parameters of nn.Modules since
    # that is how they are stored in the ParamStore
    """
    weight = get_marginal(posterior, ['module$$$linear.weight']).squeeze(1).squeeze(1)
    factor = get_marginal(posterior, ['module$$$factor'])

    # x0, x1, x2"-home_page, x1*x2-factor
    print("weight shape=", weight.shape)
    print("factor shape=", factor.shape)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharey=True)
    ax[0].hist(weight[:, 0])
    ax[1].hist(weight[:, 1])
    ax[2].hist(factor.squeeze(1))
    plt.show()


pyro_bayesian(regression_model=logistic_model, y_data=y_data_float)
pyro_bayesian(regression_model=regression_model, y_data=y_data_float)