# StructuralTouchRandomModel

## About
This project provides a stochastic, distance-dependent model for the number of touches between pairs of neurons and functionality to fit the model to data. Notably, if we consider a pair of neurons with at least one touch to be connected, this model is consistent with a model of connection probability that decays exponentially with distance.

The model has two parts: First, a model for the distribution of touches between a pair at a given distance; and second, a model for how its parameters change with distance.

## Distribution of touches per pair
The distribution of touches per pair at a given distance is assumed to follow a custom random distribution that I call sis-distribution or (s)tepwise-(i)ncreasing-(s)urvival distribution. In this distribution, the probability that there is another touch between a pair increases with the number of touches between them that have already been confirmed. 

Its parameters are as follows:
    - i: The initial probability, i.e. the probability to find at least one touch. If we assume that a pair is connected if it has at least one touch, then this is the connection probability.
    - f: The final probability that the process converges against. That is, given that there are already many touches between a pair are confirmed, this is the probability to find another touch
    - p: Parameterizes how quickly the probability to find another touch converges from i against f.
    - M: The maximum number of touches possible. That is, the distribution will be cut off at that value. Technically needed to avoid excessive computation times for large values of f, because of the way the distribution is implemented

### Definition of the distribution
Let N be the number of touches between a randomly chosen pair. It is recursively defined as follows:
    P(N > x | N >= x) = c_x
    c_x = c_{x-1} + p * (f - c_{x-1})
    c_0 = i

### Supported parameters
    - 0 <= i < 1
    - 0 <= f < 1
    - 0 <= p <= 1
    - M is an integer >= 1

Note that for the purpose of modelling touches per pair, f is expected to be larger than i, but technically it does not have to be.

## Parameters distance-dependence
Distance-dependence is modelled as follows:
    - M is not fitted, but must be chosen sufficiently large by the user. This is, because it is merely a technical parameter required to avoid infinite loops. For touches per pair, a value of 100 is generally sufficient.
    - p is assumed to be constant at all distances.
    - i is modeled as: i(d) = A_i * exp(-d / B_i). This makes the overall model compatible with exponentially decaying connection probability models.
    - f is modeled as: f(d) = A_f * exp(-d / B_f) + C_f
That is, parameters f and i are assumed to be independently distance-dependent, described by five parameters (two for i: A_i and B_i; three for f: A_f, B_f, C_f), p is assumed constant. M is a separate meta-parameter that must be set by the user.

## Implementation
The sis-distribution is implemented as a scipy.stats rv_discrete subclass. That is, like other discrete distributions in scipy.stats, such as binom or geom. It should be usable exactly like those classes, but note that the implementation is wonky and still not fully tested. 

See the jupyter notebook contained with this repository for an example.

## Fitting the model
The file fitting.py contains a function "optimize_touch_model" that can be used to fit the model to data. 
It can be used to find a best fit for all six parameters describing the distance-dependent model (A_i, B_i, A_f, B_f, C_f, p). Additionally, any combination of the parameters can set to fixed values by the user; in that case, only the remaining paramters are subject to the fit.

As input it requires a pandas.DataFrame with one row per edge in the data. It must contain at least the following columns: "bin", an integer specifying the distance bin an edge belongs to; "count", an integer specifying the number of touches making up the edge. 
Note that the model is fit only based on the number of touches making up an edge; pairs with 0 touches between them are not taken into account. That means, that the connection probability (probability to find > 0 touches) at a distance is not explicitly fit, but an emerging property!

See the jupyter notebook contained with this repository for an example.

### Model for touches per edge (keep graph structure; randomize touch count)
The file distribution.py contains a function "cut_zeros_and_shift" that converges a distribution of touches per _pair_ into a distribution of _additional_ (i.e. beyond the fist) touches per _edge_. That is, it removes the possibility of 0 touches, and turns the result of n touches into n - 1 touches. It can be used if you want to keep the wiring graph of the data identical, but randomize the number of touches that make up an edge of the graph.
