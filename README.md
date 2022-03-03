# PyLGR
A basic Python implementation of a Legendre-Gauss-Radau (LGR) pseudospectral (PS) method for computational optimal control.

The code has been tested with the following package dependencies:

scipy=1.4.1, numpy=1.16.5

Other software versions may or may not work. The pytest test suite has been tested with the following additional dependencies:

pytest=6.1.1, matplotlib=3.1.2

Plotting can be enabled in test_solve.py inside each individual test function.

See the following references for details on the LGR PS approach:

[Fahroo, F. and Ross, I. M. "Pseudospectral Methods for Infinite-Horizon Optimal Control Problems," Journal of Guidance, Control, and Dynamics, Vol. 31, No. 4, July–Aug. 2008, pp. 927-936. doi: 10.2514/1.33117](https://doi.org/10.2514/1.33117)

[Ross, I. M., Gong, Q., Fahroo, F., and Kang, W., "Practical Stabilization Through Real-Time Optimal Control," Proceedings of the 2006 American Control Conference, Inst. of Electrical and Electronics Engineers, Piscataway, NJ, June 2006, pp. 14–16. doi: 10.1109/ACC.2006.1655372](https://doi.org/10.1109/ACC.2006.1655372).
