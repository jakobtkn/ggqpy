# Generalized Gaussian quadratures
Given a family of functions $\phi_1,\phi_2,...,\phi_{2n}$, we are interested in determining Gaussian quadrature rules $x_1,x_2,...,x_n,w_1,w_2,...,w_n$ such that
$$\int_a^b f(x) dx \approx \sum_{k=1}^n f(x_k)w_k $$
up to some controllable error for all $f\in {\rm span} \\{\phi_i\\}_{i=1}^{2n}$. We do this using the approach introduced in [1].

# Reproduce results
Install mamba environment using
```
mamba create --name ggqpy -c conda-forge -c bioconda --file environment.yml
```
Then activate using
```
mamba activate ggqpy
```
Generate all figures using
```
snakemake --cores 'all'
```
This final step could potentially take a while.

# References
[1] Bremer, James, et al. ‘A Nonlinear Optimization Procedure for Generalized Gaussian Quadratures’. SIAM Journal on Scientific Computing, vol. 32, no. 4, 2010, pp. 1761–1788, https://doi.org10.1137/080737046.

[2] Bremer, James, and Zydrunas Gimbutas. ‘On the Numerical Evaluation of the Singular Integrals of Scattering Theory’. Journal of Computational Physics, vol. 251, 2013, pp. 327–343, https://doi.org10.1016/j.jcp.2013.05.048.
