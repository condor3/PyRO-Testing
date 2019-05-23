import torch
import pyro
import pyro.infer
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

def flip(weight=.5):
    uni = pyro.distributions.Uniform(0,1)
    x = pyro.sample("my_sample", uni)
    if x <= weight:
        return True
    else:
        return False


def location_prior():
    if flip(.55):
        return "popular_bar"
    else:
        return "unpopular_bar"


def alice():
    return myLocation = location_prior()





def main():
    a = pyro.infer.Importance(alice, num_samples=30)
    marginal = pyro.infer.EmpiricalMarginal(a.run())
    samples = marginal.sample([100])

    labels, counts = np.unique(samples, return_counts=True, axis=0)
    probs = counts/np.sum(counts)
    plt.bar(range(len(labels)), probs, align='center', tick_label=[str(w) for w in labels])


if __name__ == "__main__":
    main()
