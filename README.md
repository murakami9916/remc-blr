
$$
\mathrm{y}_n = (\boldsymbol{g} \circ \boldsymbol{w})^{\top}\mathrm{x}_n + \epsilon_n
$$

$$
\epsilon_n \sim \mathcal{N}(0, \lambda^{-1})
$$

$$
p(\mathrm{y}_n|\boldsymbol{g}, \boldsymbol{w}, \lambda, \mathrm{x}_n) = \mathcal{N}\left ((\boldsymbol{g} \circ \boldsymbol{w})^{\top}\mathrm{x}_n, \lambda^{-1} \right)
$$

$$
p(\boldsymbol{g} | \mathrm{x}_n, \mathrm{y}_n, \boldsymbol{w}, \lambda) = \frac{ p(\mathrm{y}_n|\boldsymbol{g}, \boldsymbol{w}, \lambda, \mathrm{x}_n) p(\boldsymbol{g}) }{Z}
$$

$$
p(\boldsymbol{g} | \mathrm{x}_n, \mathrm{y}_n) = \frac{1}{Z} \int \mathrm{d} \boldsymbol{w} \mathrm{d} \lambda \ p(\mathrm{y}_n|\boldsymbol{g}, \boldsymbol{w}, \lambda, \mathrm{x}_n)p(\boldsymbol{g})
$$

