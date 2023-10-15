The linear regression model with the feature selection is represented by the input $\mathrm{y}_n \in \mathbb{R}$, the output $\boldsymbol{\mathrm{x}}_n \in \mathbb{R}^{M}$, the weight coefficients $\boldsymbol{w} \in \mathbb{R}^{M}$, the subset indicator $\boldsymbol{g} \in \{0, 1\}^{M}$ and the noise $\epsilon_n \in \mathbb{R}$ as follows:

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


$$
p(\boldsymbol{g} | \boldsymbol{\mathrm{x}}_n, \mathrm{y}_n, \boldsymbol{w}, \lambda) = \frac{ p(\mathrm{y}_n|\boldsymbol{g}, \boldsymbol{w}, \lambda, \boldsymbol{\mathrm{x}}_n) p(\boldsymbol{g}) }{Z}.
$$

$$
p(\boldsymbol{g} | \boldsymbol{\mathrm{x}}_n, \mathrm{y}_n) = \frac{1}{Z} \int \mathrm{d} \boldsymbol{w} \mathrm{d} \lambda \ p(\mathrm{y}_n|\boldsymbol{g}, \boldsymbol{w}, \lambda, \boldsymbol{\mathrm{x}}_n)p(\boldsymbol{g})
$$

