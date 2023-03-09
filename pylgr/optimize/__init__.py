"""Functions in this module are largely taken from version 1.10 of
`scipy.optimize` to enable backwards compatibility with version 1.5. Some
modifications are made to enable extraction of KKT multipliers from SLSQP."""

from ._minimize import minimize