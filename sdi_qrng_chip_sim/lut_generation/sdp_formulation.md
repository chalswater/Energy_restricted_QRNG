# SDP/LUT Backend Formulation

## Runtime Quantity

The chip firmware only consumes a two-dimensional table:

```text
h_min = LUT(W_low, omega_high)
```

where:

```text
W_low      : finite-size lower bound on the click-contrast witness
omega_high : finite-size upper bound on monitored source energy
```

## Selected MVP Protocol

```text
x in {0, 1}
rho_0: decoy weak coherent state
rho_1: signal weak coherent state
a in {0, 1}
a = 1: click
```

Witness:

```text
W = P(1|1) - P(1|0)
```

Energy constraint:

```text
Tr(H rho_x) <= omega_x
omega_high = max_x omega_x
```

## Formal Replacement Target

For a future SDP backend, solve a guessing-probability upper bound for every
grid point `(W_i, omega_j)`:

```text
maximize    p_guess
subject to  W(rho_x, M_a) >= W_i
            Tr(H rho_x) <= omega_j
            rho_x >= 0
            Tr(rho_x) = 1
            M_a >= 0
            M_0 + M_1 = I
```

Then store:

```text
h_min(W_i, omega_j) = -log2(p_guess(W_i, omega_j))
```

For an infinite-dimensional optical source, the implementation should use a
finite Fock truncation plus a truncation-error bound, or an equivalent
energy-constrained relaxation.

## Current Backends

The default backend is:

```text
energy_constrained_relaxation_v1
```

It implements a conservative energy-envelope relaxation so that:

```text
1. runtime code no longer uses a linear placeholder;
2. firmware behavior is LUT-only;
3. patent figures can demonstrate the closed-loop architecture;
4. the formal SDP backend can be swapped in later without touching the runtime.
```

The explicit SDP hook is:

```text
cvxpy_sdp_formulation
```

It fails loudly unless an SDP solver stack is installed and the final optical
relaxation is implemented. This avoids silently presenting an engineering LUT
as a completed proof.
