# SDI QRNG Chip Protocol Specification

## Final MVP Protocol

The implemented chip simulation now uses a two-intensity prepare-and-measure
protocol as the concrete MVP target.

Per round:

```text
x in {0, 1}
x = 0: decoy weak coherent state with mean photon number mu_0
x = 1: signal weak coherent state with mean photon number mu_1
mu_0 < mu_1
a in {0, 1}
a = 1: detector click
a = 0: no click
```

The chip monitors energy through a tap coupler and estimates a conservative
upper bound:

```text
omega_high = max_x upper_confidence_bound(mu_x)
```

## Witness

The implemented witness is:

```text
W = P(a = 1 | x = 1) - P(a = 1 | x = 0)
```

The runtime uses a finite-size lower bound:

```text
W_low = W_hat - Delta_W
```

This witness subtracts the decoy click response from the signal click response,
so dark counts and common detector background reduce the certification score.

## Energy Constraint

The SDI assumption used by the chip controller is:

```text
Tr(H rho_x) <= omega_x
```

For the weak coherent-state MVP, the monitored mean photon number is used as the
operational energy proxy:

```text
omega_x ~= mu_x
```

The default security envelope is:

```text
omega_high <= 0.30
```

If `omega_high` exceeds the envelope, the LUT discounts the usable certified
response and the event-triggered calibration controller reduces source energy.

## Offline LUT

The runtime entropy function is no longer a linear placeholder. Runtime code
loads:

```text
lut_generation/lut_data.npz
```

The LUT maps:

```text
(W_low, omega_high) -> h_min
```

The current LUT backend is implemented in:

```text
lut_generation/sdp_solver.py
```

It uses a conservative energy-envelope relaxation:

```text
q_cert = max(0, W_low - penalty(omega_high))
p_guess = 1 - q_cert
h_min = -log2(p_guess)
```

The backend is intentionally isolated. For a formal security proof, replace the
body of `h_min_grid` with an SDP or convex-relaxation solver for the selected
prepare-and-measure protocol. The chip-side interface remains unchanged because
the firmware only performs LUT lookup and interpolation.

## Runtime Control

For each block:

```text
1. estimate P(a|x)
2. compute W_low
3. compute omega_high from monitor samples
4. read h_min = LUT(W_low, omega_high)
5. compute output length from leftover hash lemma
6. emit certified bits only if m_k > 0
7. trigger calibration if energy, entropy, or dark-count thresholds are crossed
```

This is the protection boundary for the patent embodiment: entropy estimation
and extractor length are coupled to real-time on-chip monitoring.

