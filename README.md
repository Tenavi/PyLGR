# PyLGR

#### Note: this repository is no longer maintained. The [benchmark_ocp](https://github.com/Tenavi/benchmark_ocp) repository contains a regularly updated implementation of this code's functionality.

--------------------------------------------------------------------------------

`pylgr` is a basic Python implementation of a Legendre-Gauss-Radau (LGR) pseudospectral
(PS) method for infinite horizon computational optimal control. The optimal
control problem (OCP) is collocated in time at LGR points, turning it into a
constrained nonlinear programming problem which we solve with sequential least
squares quadratic programming (SLSQP). Rough estimates for the costates are
extracted based on the covector mapping theorem.

--------------------------------------------------------------------------------

The code has been tested with the following dependencies:

    python>=3.6
    numpy>=1.19.5
    scipy>=1.5.4
    pytest>=7.0.1
    matplotlib>=3.3.4

The code can be installed using the following command:

`pip install -e .`

You can then import pylgr in any python script. The main function is
`pylgr.solve_ocp`. Documentation can be accessed by the python command
`help(pylgr.solve_ocp)`.

--------------------------------------------------------------------------------

The test suite can be run from the main directory with the command

`pytest unit_tests -s -v`

Plotting can be enabled in test_solve.py inside each individual test function.

--------------------------------------------------------------------------------

See the following references for details on the LGR PS approach:

[Fahroo, F. and Ross, I. M. "Pseudospectral Methods for Infinite-Horizon Optimal Control Problems," Journal of Guidance, Control, and Dynamics, Vol. 31, No. 4, July–Aug. 2008, pp. 927-936. doi: 10.2514/1.33117](https://doi.org/10.2514/1.33117)

[Ross, I. M., Gong, Q., Fahroo, F., and Kang, W., "Practical Stabilization Through Real-Time Optimal Control," Proceedings of the 2006 American Control Conference, Inst. of Electrical and Electronics Engineers, Piscataway, NJ, June 2006, pp. 14–16. doi: 10.1109/ACC.2006.1655372](https://doi.org/10.1109/ACC.2006.1655372).
