# Bayesian linear regression with feature selection using replica exchange Monte Carlo Method

## Model: Bayesian linear regression with feature selection

Given the data $\mathcal{D}=\\{ (\boldsymbol{\mathrm{x}}_n, \mathrm{y}_n)|n=1,2,..., N\\}$, we consider a linear regression task. The linear regression model with the feature selection is represented by the input $\mathrm{y}_n \in \mathbb{R}$, the output $\boldsymbol{\mathrm{x}}_n \in \mathbb{R}^{M}$, the weight coefficients $\boldsymbol{w} \in \mathbb{R}^{M}$, the subset indicators $\boldsymbol{g} \in \\{0, 1\\}^{M}$ and the noise $\epsilon_n \in \mathbb{R}$ as follows:

$$
\mathrm{y}_n = (\boldsymbol{g} \circ \boldsymbol{w})^{\top}\boldsymbol{\mathrm{x}}_n + \epsilon_n.
$$

Here, we assume that the noise follows a Gaussian distribution $\mathcal{N}(0, \lambda^{-1})$ as follows:

$$
\epsilon_n \sim \mathcal{N}(0, \lambda^{-1}),
$$

where $\lambda \in \mathbb{R}^{+}$ denote the precision parameter of a Gaussian distribution. Therefore, the probability distribution of $\mathrm{y}_n$ is represented by the above formula as follows:

$$
p(\mathrm{y}_n|\boldsymbol{g}, \boldsymbol{w}, \lambda, \boldsymbol{\mathrm{x}}_n) = \mathcal{N}\left ((\boldsymbol{g} \circ \boldsymbol{w})^{\top}\boldsymbol{\mathrm{x}}_n, \lambda^{-1} \right).
$$

The probability distribution of $\mathcal{D}$ can be expressed as follows:

$$
p(\mathcal{D}|\boldsymbol{g}, \boldsymbol{w}, \lambda, \boldsymbol{\mathrm{x}}_n) = \prod_n{\mathcal{N}\left ((\boldsymbol{g} \circ \boldsymbol{w})^{\top}\boldsymbol{\mathrm{x}}_n, \lambda^{-1} \right)}.
$$

The conditional probability of $\boldsymbol{g}$ can be expressed using Bayes' theorem and marginalization as follows:

$$
p(\boldsymbol{g} | \mathcal{D}) = \frac{1}{Z} \int \mathrm{d} \boldsymbol{w} \mathrm{d} \lambda \ p(\mathcal{D}|\boldsymbol{g}, \boldsymbol{w}, \lambda)p(\boldsymbol{w})p(\lambda)p(\boldsymbol{g}),
$$

where the probability distribution $p(\boldsymbol{w})$, $p(\boldsymbol{g})$, $p(\lambda)$ denote the prior distribution of the stochastic variable. This is, the free energy $F(\boldsymbol{g})$ given the used indicator $\boldsymbol{g}$ is expressed as follows: $F(\boldsymbol{g}) = -\ln{p(\boldsymbol{g}|\mathcal{D})}$. This code assumed an uninformed distribution as the prior distribution $p(\boldsymbol{w})$, $p(\boldsymbol{\lambda})$. The prior distribution $p(\boldsymbol{g})$ is set to the uninformed Bernoulli distribution.

## Algorithm: Replica Exchange Monte Carlo Method
We perform posterior visualization and the maximum a posteriori (MAP) estimation through sampling from the posterior distribution. A popular sampling method is the Monte Carlo (MC) method, which may be bounded by local solutions for cases when the initial value is affected or the cost function landscape is complex.

Therefore, the replica exchange Monte Carlo (REMC) method was used to estimate the global solution. For sampling using the REMC method, a replica was prepared with the inverse temperature $\beta$ introduced as follows:

$$
    p(\boldsymbol{g}|\mathcal{D};\beta=\beta_{\tau}) = \exp{ (-\beta_{\tau} F(\boldsymbol{g}) ) }p(\boldsymbol{g}),
$$

where the inverse temperature $\beta$ is $0 = \beta_1 < \beta_2 < \cdots < \beta_{\tau} < \beta_T = 1$. For each replica, the parameters were sampled using the Monte Carlo method.

## Estimating Density of States (DoS) using the multi-histogram method
This code can estimate the density of states (DoS) from the histogram $H(F; \beta_{\tau})$ at an inverse temperature $\beta_{\tau}$, which can be obtained by the REMC method. When given the histogram $H(F; \beta_{\tau})$, we can calculate the density of states $D(F)$ by the iteration equation of the normalization constant $Z(\beta_\tau)$ and the density of states:

$$
D(F) = \frac{ \sum_{\tau}{H(F; \beta_{\tau})} }{ \sum_{\tau}{n_\tau \exp({-\beta_{\tau}F})/Z(\beta_\tau)} },
$$

## Example


$$
Z(\beta_\tau) = \sum_{F}{D(F)\exp({-\beta_{\tau}F})}
$$

where $n_\tau$ is the number of samples at $\beta_\tau$. We alternately compute the above equations to estimate $D(F)$ and $Z(\beta_\tau)$.
