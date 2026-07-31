# Signal definition

## Empirical measures

Let `P_k` denote the multivariate empirical return measure constructed at
frequency `k`, and let `H_k` be its effective horizon in base periods.

Under power scaling with exponent `h`, each observation is transformed as:

```text
x_scaled = x / H_k**h
```

The reference setting uses `h = 0.5`. Realized-volatility standardization is
available as a separate robustness specification; it standardizes every asset
within each frequency by its sample volatility.

`H_k` is declared by the caller. It is not inferred from the pandas resampling
rule, because an effective financial horizon is a scientific convention rather
than a property of a timestamp label.

## Barycenter and dispersion weights

MFDRO distinguishes two weight vectors:

- `beta_k`: weights used to construct the central barycenter;
- `lambda_k`: weights used to aggregate dispersion around that center.

Both are uniform by default. Keeping them separate permits a change in the
dispersion experiment without silently changing the center.

For a multivariate free-support barycenter `Q`, the target is conceptually:

```text
Q = argmin_Q sum_k beta_k * W2(P_k, Q)**2
```

The reported signal is:

```text
rho = sum_k lambda_k * d(P_k, Q)**2
```

where `d` is either the configured sliced approximation or the exact discrete
transport distance. Consequently, `rho` is a squared dispersion and
`sqrt_rho` is its square root.

## Barycenter choices

`free_support`

: A multivariate free-support Wasserstein barycenter computed by POT. The
  support is initialized with measure-balanced weighted k-means++.

`projected_quantile`

: For each projection direction, empirical quantile functions are averaged
  with the barycenter weights. This avoids a multivariate free-support solve.
  It is a projected construction, not one stored multivariate barycenter.

## Distance choices

`sliced`

: Average projected squared `W_2` discrepancies over reproducible random unit
  directions. More projections reduce Monte Carlo error but increase runtime.

`exact`

: Compute the exact discrete squared transport cost between every empirical
  frequency measure and a free-support barycenter. This mode is generally more
  expensive and is incompatible with `projected_quantile`.

## Interpretation boundary

A large `rho` indicates that the scaled empirical measures disagree more
strongly around their configured center. It does not identify which asset will
rise, estimate expected return, or specify how much robustness a portfolio
optimizer should use.

A downstream research design may map `rho` or `sqrt_rho` into an ambiguity
radius, a regime indicator, or a model-combination weight. That mapping must be
estimated point in time and audited separately.

## Random approximation

The sliced estimator uses finitely many random directions. Every result records
its effective seed, and all frequencies inside one estimate use the same
direction stream.

For controlled comparisons, competing configurations should use common random
numbers when that is part of the experimental design.
