import numpy as np
from scipy.optimize._slsqp import slsqp
from scipy.optimize._constraints import old_bound_to_new
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative

from ._optimize import (_prepare_scalar_function, _clip_x_for_func,
                        _check_clip_x)


def _minimize_slsqp(
        fun, x0, args=(), jac=None, bounds=None, constraints=(),
        maxiter=100, ftol=1.0E-6, iprint=1, disp=False,
        eps=np.sqrt(np.finfo(float).eps), finite_diff_rel_step=None
    ):
    """
    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP). Modified from
    `scipy.optimize._slsqp_py._minimize_slsqp` to extract KKT multipliers.
    Based on work by github user andyfaff.

    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of `jac`. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    iter = maxiter - 1
    acc = ftol

    if not disp:
        iprint = 0

    # Transform x0 into an array.
    x = np.asfarray(x0).flatten()

    # SLSQP is sent 'old-style' bounds, 'new-style' bounds are required by
    # ScalarFunction
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)

    # clip the initial guess to bounds, otherwise ScalarFunction doesn't work
    x = np.clip(x, new_bounds[0], new_bounds[1])

    # Constraints are triaged per type into a dictionary of tuples
    if isinstance(constraints, dict):
        constraints = (constraints, )

    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function. The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun):
                def cjac(x, *args):
                    x = _check_clip_x(x, new_bounds)

                    if jac in ['2-point', '3-point', 'cs']:
                        return approx_derivative(fun, x, method=jac, args=args,
                                                 rel_step=finite_diff_rel_step,
                                                 bounds=new_bounds)
                    else:
                        return approx_derivative(fun, x, method='2-point',
                                                 abs_step=eps, args=args,
                                                 bounds=new_bounds)

                return cjac
            cjac = cjac_factory(con['fun'])

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # Set the parameters that SLSQP will need
    # _meq_cv: a list containing the length of values each constraint function
    _meq_cv = [len(np.atleast_1d(c['fun'](x, *c['args']))) for c in cons['eq']]
    _mieq_cv = [len(np.atleast_1d(c['fun'](x, *c['args']))) for c in cons['ineq']]
    # meq, mieq: number of equality and inequality constraints
    meq = sum(_meq_cv)
    mieq = sum(_mieq_cv)
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = np.array([1, m]).max()
    # n = The number of independent variables
    n = len(x)

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = np.zeros(len_w)
    jw = np.zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = np.array(
            [(_arr_to_scalar(l), _arr_to_scalar(u)) for (l, u) in bounds],
            dtype=float
        )
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~np.isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # ScalarFunction provides function and gradient evaluation
    sf = _prepare_scalar_function(fun, x, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    # gh11403 SLSQP sometimes exceeds bounds by 1 or 2 ULP, make sure this
    # doesn't get sent to the func/grad evaluator.
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)

    # Initialize the iteration counter and the mode value
    mode = np.array(0, int)
    acc = np.array(acc, float)
    majiter = np.array(iter, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = np.array(0, float)
    f0 = np.array(0, float)
    gs = np.array(0, float)
    h1 = np.array(0, float)
    h2 = np.array(0, float)
    h3 = np.array(0, float)
    h4 = np.array(0, float)
    t = np.array(0, float)
    t0 = np.array(0, float)
    tol = np.array(0, float)
    iexact = np.array(0, int)
    incons = np.array(0, int)
    ireset = np.array(0, int)
    itermx = np.array(0, int)
    line = np.array(0, int)
    n1 = np.array(0, int)
    n2 = np.array(0, int)
    n3 = np.array(0, int)

    # Print the header if iprint >= 2
    if iprint >= 2:
        print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))

    # mode is zero on entry, so call objective, constraints and gradients
    # there should be no func evaluations here because it's cached from
    # ScalarFunction
    fx = wrapped_fun(x)
    g = np.append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

    while 1:
        # Call SLSQP
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)

        if mode == 1:  # objective and constraint evaluation required
            fx = wrapped_fun(x)
            c = _eval_constraint(x, cons)

        if mode == -1:  # gradient evaluation required
            g = np.append(wrapped_grad(x), 0.0)
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

        if majiter > majiter_prev:
            # Print the status of the current iterate if iprint > 2
            if iprint >= 2:
                print("%5i %5i % 16.6E % 16.6E" % (majiter, sf.nfev,
                                                   fx, np.linalg.norm(g)))

        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:
            break

        majiter_prev = int(majiter)

    # Obtain KKT multipliers
    im = 1
    il = im + la
    ix = il + (n1*n)//2 + 1
    ir = ix + n - 1
    _kkt_mult = w[ir:ir + m]

    # KKT multipliers
    w_ind = 0
    kkt_multiplier = dict()

    for _t, cv in [("eq", _meq_cv), ("ineq", _mieq_cv)]:
        kkt = []

        for dim in cv:
            kkt += [_kkt_mult[w_ind:(w_ind + dim)]]
            w_ind += dim

        kkt_multiplier[_t] = kkt

    # Optimization loop complete. Print status if requested
    if iprint >= 1:
        print(f"{exit_modes[int(mode)]}    (Exit mode {mode})")
        print("            Current function value:", fx)
        print("            Iterations:", majiter)
        print("            Function evaluations:", sf.nfev)
        print("            Gradient evaluations:", sf.ngev)

    return OptimizeResult(x=x, fun=fx, jac=g[:-1],
                          nit=int(majiter),
                          nfev=sf.nfev, njev=sf.ngev, status=int(mode),
                          message=exit_modes[int(mode)],
                          success=(mode==0),
                          kkt=kkt_multiplier)


def _eval_constraint(x, cons):
    # Compute constraints
    if cons['eq']:
        c_eq = np.concatenate(
            [np.atleast_1d(con['fun'](x, *con['args'])) for con in cons['eq']]
        )
    else:
        c_eq = np.zeros(0)

    if cons['ineq']:
        c_ieq = np.concatenate(
            [np.atleast_1d(con['fun'](x, *con['args'])) for con in cons['ineq']]
        )
    else:
        c_ieq = np.zeros(0)

    # Now combine c_eq and c_ieq into a single matrix
    c = np.concatenate((c_eq, c_ieq))
    return c

def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    # Compute the normals of the constraints
    if cons['eq']:
        a_eq = np.vstack(
            [con['jac'](x, *con['args']) for con in cons['eq']]
        )
    else:  # no equality constraint
        a_eq = np.zeros((meq, n))

    if cons['ineq']:
        a_ieq = np.vstack(
            [con['jac'](x, *con['args']) for con in cons['ineq']]
        )
    else:  # no inequality constraint
        a_ieq = np.zeros((mieq, n))

    # Now combine a_eq and a_ieq into a single a matrix
    if m == 0:  # no constraints
        a = np.zeros((la, n))
    else:
        a = np.vstack((a_eq, a_ieq))
    a = np.concatenate((a, np.zeros([la, 1])), 1)

    return a


def _arr_to_scalar(x):
    # If x is a numpy array, return x.item().  This will
    # fail if the array has more than one element.
    return x.item() if isinstance(x, np.ndarray) else x
