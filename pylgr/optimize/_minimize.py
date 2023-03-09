import numpy as np

from scipy.optimize._differentiable_functions import FD_METHODS
from scipy.optimize._minimize import MemoizeJac
from scipy.optimize._minimize import standardize_constraints, standardize_bounds
from scipy.optimize._constraints import Bounds

from ._slsqp import _minimize_slsqp


def minimize(
        fun, x0, args=(), jac=None,
        bounds=None, constraints=(), tol=None, options=None
    ):
    """Minimization of scalar function of one or more variables.

    Wrapper and modification of `scipy.optimize.optimize` implementing shortcuts
    to the `"SLSQP"` method and extracting the KKT multipliers. Based on work by
    github user andyfaff.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            fun(x, *args) -> float
        where x is an 1-D array with shape (n,) and args
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where n is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (fun, jac and hess functions).
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:
            jac(x, *args) -> array_like, shape (n,)
        where x is an array with shape (n,) and args is a tuple with
        the fixed parameters. If jac is a Boolean and is True, fun is
        assumed to return a tuple (f, g) containing the objective
        function and the gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified bounds.
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables as an instance of Bounds class.
    constraints : scipy.optimize.Constraint or List of Constraints, optional
        Constraints defined as a single object or a list of objects specifying
        constraints to the optimization problem.
        Available constraints are:
            - LinearConstraint
            - NonlinearConstraint
    tol : float, optional
        Tolerance for termination. When tol is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to tol. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options.
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
            ftol : float
                Precision goal for the value of f in the stopping criterion.
            eps: float
                Step size used for numerical approximation of the Jacobian.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a
        Boolean flag indicating if the optimizer exited successfully and
        message which describes the cause of the termination. See
        OptimizeResult for a description of other attributes.
    """
    x0 = np.atleast_1d(np.asarray(x0))
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}

    # check gradient vector
    if callable(jac) or jac in FD_METHODS:
        pass
    elif jac is True:
        # fun returns func and grad
        fun = MemoizeJac(fun)
        jac = fun.derivative
    else:
        # default if jac option is not understood
        jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        options.setdefault('ftol', tol)

    constraints = standardize_constraints(constraints, x0, 'slsqp')

    remove_vars = False
    if bounds is not None:
        # SLSQP can't take the finite-difference derivatives when a variable is
        # fixed by the bounds. To avoid this issue, remove fixed variables from
        # the problem.

        # convert to new-style bounds so we only have to consider one case
        bounds = standardize_bounds(bounds, x0, 'new')

        # determine whether any variables are fixed
        i_fixed = (bounds.lb == bounds.ub)

        # determine whether finite differences are needed for any grad/jac
        fd_needed = (not callable(jac))
        for con in constraints:
            if not callable(con.get('jac', None)):
                fd_needed = True

        # If finite differences are ever used, remove all fixed variables
        remove_vars = i_fixed.any() and fd_needed
        if remove_vars:
            x_fixed = (bounds.lb)[i_fixed]
            x0 = x0[~i_fixed]
            bounds = _remove_from_bounds(bounds, i_fixed)
            fun = _remove_from_func(fun, i_fixed, x_fixed)
            if callable(jac):
                jac = _remove_from_func(jac, i_fixed, x_fixed, remove=1)

            # make a copy of the constraints so the user's version doesn't
            # get changed. (Shallow copy is ok)
            constraints = [con.copy() for con in constraints]
            for con in constraints:  # yes, guaranteed to be a list
                con['fun'] = _remove_from_func(con['fun'], i_fixed,
                                               x_fixed, min_dim=1,
                                               remove=0)
                if callable(con.get('jac', None)):
                    con['jac'] = _remove_from_func(con['jac'], i_fixed,
                                                   x_fixed, min_dim=2,
                                                   remove=1)
    bounds = standardize_bounds(bounds, x0, 'slsqp')

    res = _minimize_slsqp(fun, x0, args, jac, bounds, constraints, **options)

    if remove_vars:
        res.x = _add_to_array(res.x, i_fixed, x_fixed)
        res.jac = _add_to_array(res.jac, i_fixed, np.nan)
        if 'hess_inv' in res:
            res.hess_inv = None

    return res


def _remove_from_bounds(bounds, i_fixed):
    """Removes fixed variables from a `Bounds` instance."""
    lb = bounds.lb[~i_fixed]
    ub = bounds.ub[~i_fixed]
    return Bounds(lb, ub)  # don't mutate original Bounds object


def _remove_from_func(fun_in, i_fixed, x_fixed, min_dim=None, remove=0):
    """Wraps a function such that fixed variables need not be passed in."""
    def fun_out(x_in, *args, **kwargs):
        x_out = np.zeros_like(i_fixed, dtype=x_in.dtype)
        x_out[i_fixed] = x_fixed
        x_out[~i_fixed] = x_in
        y_out = fun_in(x_out, *args, **kwargs)
        y_out = np.array(y_out)

        if min_dim == 1:
            y_out = np.atleast_1d(y_out)
        elif min_dim == 2:
            y_out = np.atleast_2d(y_out)

        if remove == 1:
            y_out = y_out[..., ~i_fixed]
        elif remove == 2:
            y_out = y_out[~i_fixed, ~i_fixed]

        return y_out
    return fun_out


def _add_to_array(x_in, i_fixed, x_fixed):
    """Adds fixed variables back to an array"""
    i_free = ~i_fixed
    if x_in.ndim == 2:
        i_free = i_free[:, None] @ i_free[None, :]
    x_out = np.zeros_like(i_free, dtype=x_in.dtype)
    x_out[~i_free] = x_fixed
    x_out[i_free] = x_in.ravel()
    return x_out