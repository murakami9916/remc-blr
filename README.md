
$$
\mathrm{y}_n = (\bm{g} \circ \bm{w})^{\top}\mathrm{x}_n + \epsilon_n
$$

$$
\epsilon_n \sim \mathcal{N}(0, \lambda^{-1})
$$

$$
p(\mathrm{y}_n|\bm{g}, \bm{w}, \lambda, \mathrm{x}_n) = \mathcal{N}\left ((\bm{g} \circ \bm{w})^{\top}\mathrm{x}_n, \lambda^{-1} \right)
$$

$$
p(\bm{g} | \mathrm{x}_n, \mathrm{y}_n, \bm{w}, \lambda) = \frac{ p(\mathrm{y}_n|\bm{g}, \bm{w}, \lambda, \mathrm{x}_n) p(\bm{g}) }{Z}
$$

$$
p(\bm{g} | \mathrm{x}_n, \mathrm{y}_n) = \frac{1}{Z} \int \mathrm{d} \bm{w} \mathrm{d} \lambda \ p(\mathrm{y}_n|\bm{g}, \bm{w}, \lambda, \mathrm{x}_n)p(\bm{g})
$$

