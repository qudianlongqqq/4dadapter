# v1 to Delta-Q formulation

For Bond lengths, `Delta q* = q_ref - q_source` in Angstrom. For Angles,
`Delta q* = cos(theta_ref) - cos(theta_source)` with no degree/radian target.

The model predicts `(Delta q hat, sigma, reliability)` from graph features and
the current Source primitive. The unchanged action receives
`mu_eff = q_source + Delta q hat`, hence `q_source - mu_eff = -Delta q hat`.

The belief objective is the exact v1 J1 construction applied to correction
residuals: `NLL(Delta q*, Delta q hat, sigma) * stop_gradient(sigma^2)^0.5`,
with molecule-balanced equal Bond/Angle aggregation. Sigma is a predicted
heteroscedastic correction scale, not calibrated uncertainty.
