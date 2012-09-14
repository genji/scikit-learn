"""
Restricted Bo....

Description
"""

# Author:
#

import numpy as np

from scipy import linalg
from sklearn.utils import check_random_state
from sklearn.utils.extmath import logsumexp


def sigmoid(value):
    a = np.exp(value)
    return a / (1. + a)


def bin_perm_rep(ndim, a=0, b=1):
    """bin_perm_rep(ndim) -> ndim permutations with repetitions of (a,b).

    Returns an array with all the possible permutations with repetitions of
    (0,1) in ndim dimensions.  The array is shaped as (2**ndim,ndim), and is
    ordered with the last index changing fastest.  For examble, for ndim=3:

    Examples:

    >>> bin_perm_rep(3)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]])
    """

    # Create the leftmost column as 0,0,...,1,1,...
    nperms = 2 ** ndim
    perms = np.empty((nperms, ndim), type(a))
    perms.fill(a)
    half_point = nperms / 2
    perms[half_point:, 0] = b
    # Fill the rest of the table by sampling the pervious column every 2 items
    for j in range(1, ndim):
        half_col = perms[::2, j - 1]
        perms[:half_point, j] = half_col
        perms[half_point:, j] = half_col

    return perms


def _sample_rbm(coef_, intercept_visible_, intercept_hidden_,
               state, direction):
    """

    """
    #visible->hidden
    if (direction == 'up'):
        mean = sigmoid(np.dot(coef_, state) + intercept_hidden_)

    #hidden->visible
    elif (direction == 'down'):
        mean = sigmoid(np.dot(coef_.T, state) + intercept_visible_)

    sample = np.random.binomial(n=1, p=mean).astype(np.float)

    return sample, mean


def _gibbs_sampling(coef_, intercept_visible_, intercept_hidden_,
                    v_state, n_steps):
    for n in range(n_steps):
        h_state, _ = _sample_rbm(coef_, intercept_visible_, intercept_hidden_,
                                v_state, 'up')

        v_state, _ = _sample_rbm(coef_, intercept_visible_, intercept_hidden_,
                                h_state, 'down')
    return v_state


def compute_gradient(coef_, intercept_visible_, intercept_hidden_,
                    v_state, cd_steps, step_size):
    h_state, h_mean = _sample_rbm(coef_, intercept_visible_, intercept_hidden_,
                                v_state, 'up')
    ger, = linalg.get_blas_funcs(('ger',), (h_mean, v_state))
    ger(step_size, v_state, h_mean, a=coef_.T, overwrite_a=1)

    # import ipdb; ipdb.set_trace()
    v_state_neg, _ = _sample_rbm(coef_, intercept_visible_, intercept_hidden_,
                                h_state, 'down')

    v_state_neg = _gibbs_sampling(coef_, intercept_visible_, intercept_hidden_,
                    v_state_neg, (cd_steps - 1))

    _, h_mean_neg = _sample_rbm(coef_, intercept_visible_, intercept_hidden_,
                                v_state_neg, 'up')
    ger(-step_size, v_state_neg, h_mean_neg, a=coef_.T, overwrite_a=1)


def _compute_free_energy(coef_, intercept_visible_, intercept_hidden_,
                         v_state):

    #print np.dot(v_state, intercept_visible_).shape
    #print np.dot(coef_, v_state.T).shape
    #print intercept_hidden_.shape

    if v_state.ndim == 1:
        v_state = v_state[None, :]
    fe = np.dot(v_state, intercept_visible_) + \
         np.sum(np.log(1 + np.exp(np.dot(coef_, v_state.T)
                + intercept_hidden_[:, None])), axis=0)
    if len(fe) == 1:
        fe = float(fe)
    return fe


def compute_log_partition_function(coef_, intercept_visible_,
                                   intercept_hidden_):
    n_visible = len(intercept_visible_)
    all_v_states = bin_perm_rep(n_visible)
    return logsumexp(_compute_free_energy(coef_, intercept_visible_,
                                          intercept_hidden_, all_v_states))


def compute_log_prob(coef_, intercept_visible_, intercept_hidden_, v_state):
    return _compute_free_energy(coef_, intercept_visible_, intercept_hidden_,
            v_state) - compute_log_partition_function(coef_,
            intercept_visible_, intercept_hidden_)


def estimate_log_partition_function(coef_, intercept_visible_, intercept_hidden_, n_chains=100, beta=10000):
    # Estimate the log-partition function of an RBM using Annealed Importance Sampling.

    # If beta is an int, it is the number of steps.
    # If it is a vector, it is a sequence of temperatures.
    if len(beta) == 1:
        n_temps = beta
        beta = np.concatenate(np.linspace(0., .5, floor(n_temps*.1)),\
                              np.linspace(.5, .9, floor(n_temps*.4)),\
                              np.linspace(.9, 1., floor(n_temps*.5)));

    # Initialize the vector which contains the estimate for each Markov chain.
    ais = np.zeros(n_chains)


    # Start by computing the log-partition function of the RBM with zeros weights
    np.sum(np.log(1 + np.exp(intercept_visible_)))
    AIS = sum(log(1+exp(rbm.Bv))) + rbm.Nh*log(2);



def trainBinaryRBM(dataset, n_hidden, n_iterations):
    # Get the number of visible units from the size of the dataset.
    n_data, n_visible = dataset.shape

    # Initialize the weights and the intercepts
    coef_ = rng.randn(n_hidden, n_visible)/np.sqrt(n_visible)
    intercept_visible_ = np.zeros(n_visible)
    intercept_hidden_ = np.zeros(n_hidden)

    # Train the model.
    for iteration in xrange(n_iterations):
        for datapoint in xrange(n_data):
            compute_gradient(W, i_v, i_h, V, 1, 0.01)

    return coef_, intercept_visible_, intercept_hidden_


if __name__ == '__main__':
    V = np.array([1., 1., 0., 0.])

    rng = check_random_state(0)
    W = rng.randn(6, 4)
    #W = np.zeros((6,4))

    i_v = np.zeros(4)
    i_h = np.zeros(6)

    print "Iteration 0, log-prob=", compute_log_prob(W, i_v, i_h, V)
    for i in range(10000):
        compute_gradient(W, i_v, i_h, V, 1, 0.01)
        print "Iteration ", i + 1, ", log-prob=", compute_log_prob(
            W, i_v, i_h, V)
