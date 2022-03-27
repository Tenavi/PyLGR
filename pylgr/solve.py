import numpy as np
import warnings
from scipy import optimize
from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import solve_ivp

from .legendre_gauss_radau import make_LGR
from . import utilities

class LagrangeInterpolator(BarycentricInterpolator):
    def __init__(self, tau, Y, lb=None, ub=None):
        super().__init__(xi=tau, yi=Y, axis=1)
        self.lb, self.ub = lb, ub

    def __call__(self, t):
        Y = super().__call__(utilities.time_map(t))
        if self.lb is not None or self.ub is not None:
            if Y.ndim < 2:
                Y = Y.reshape(-1,1)
            Y = np.squeeze(np.clip(Y, self.lb, self.ub))
        return Y

class DirectSolution:
    def __init__(
            self, NLP_res, tau, running_cost, dynamic_constr, U_lb, U_ub,
            order, separate_vars
        ):
        self.NLP_res = NLP_res
        self._separate_vars = separate_vars
        self._running_cost = running_cost

        self.success = NLP_res.success
        self.status = NLP_res.status
        self.message = NLP_res.message

        self.t = utilities.invert_time_map(tau)

        self.X, self.U = separate_vars(NLP_res.x)
        self.sol_X = LagrangeInterpolator(tau, self.X)
        self.sol_U = LagrangeInterpolator(tau, self.U, U_lb, U_ub)

        self.V = self.sol_V(self.t)

        if hasattr(NLP_res, 'constr'):
            print(dir(NLP_res))
            self.residuals = NLP_res.constr[0]
        else:
            self.residuals = dynamic_constr.fun(NLP_res.x)
        self.residuals = self.residuals.reshape(self.X.shape, order=order)
        self.residuals = np.max(np.abs(self.residuals), axis=0)

    def _value_dynamics(self, t, J):
        X = self.sol_X(t)
        U = self.sol_U(t)
        return -self._running_cost(X, U)

    def sol_V(self, t_eval):
        t_eval = np.sort(t_eval)
        t1 = np.maximum(self.t[-1], t_eval[-1])

        V0 = np.reshape(self.NLP_res.fun, (1,))
        sol = solve_ivp(self._value_dynamics, [0., t1], V0, t_eval=t_eval)
        return sol.y

def solve_ocp(
        dynamics, cost_fun, t_guess, X_guess, U_guess, U_lb=None, U_ub=None,
        dynamics_jac='2-point', cost_grad='2-point',
        n_nodes=32, tol=1e-07, maxiter=1000, solver_options={},
        reshape_order='C', verbose=0
    ):
    '''Solve an open loop OCP by LGR pseudospectral method.

    Parameters
    ----------
    dynamics : callable
        Right-hand side of the system, dXdt = dynamics(X,U).
    cost_fun : callable
        Running cost of the OCP, L = cost_fun(X,U).
    t_guess : (n_points,) array
        Time points for initial guess. Must be a strictly increasing sequence of
        real numbers with t[0]=0 and t[-1]=t1 > 0.
    X_guess : (n_states, n_points) array
        Initial guess for the state values X(t). Assumes that the initial
        condition X0 is contained in the first column of X_guess.
    U_guess : (n_controls, n_points) array
        Initial guess for the control values U(t).
    U_lb : (n_controls,1) array, optional
        Lower bounds for the controls.
    U_ub : (n_controls,1) array, optional
        Upper bounds for the controls.
    dynamics_jac : {callable, '3-point', '2-point', 'cs'}, default='2-point'
        Jacobian of the dynamics dXdt=F(X,U) with respect to states X and
        controls U. If callable, function dynamics_jac should take two arguments
        X and U with respective shapes (n_states, n_nodes) and
        (n_controls, n_nodes), and return a tuple of Jacobian arrays
        (dF/dX, dF/dU) with respective shapes (n_states, n_states, n_nodes) and
        (n_states, n_controls, n_nodes). Other string options specify the finite
        difference methods to use if the analytical Jacobian is not available.
    cost_grad : {callable, '3-point', '2-point', 'cs', bool}, default='2-point'
        Gradients of the running cost L with respect to X and U. If callable,
        cost_grad should take two arguments X and U with respective shapes
        (n_states, n_nodes) and (n_controls, n_nodes), and return dL/dX and
        dL/dU with the same shapes. If cost_grad=True, then assume that cost_fun
        returns the gradients in addition to the running cost. String options
        specify finite difference methods.
    n_nodes : int, default=32
        Number of LGR points for collocating time.
    tol : float, default=1e-07
        Tolerance for termination.
    maxiter : int, default=1000
        Maximum number of iterations to perform.
    solver_options : dict, optional
        Solver-specific keyword arguments. See scipy.optimize.minimize.SLSQP for
        details.
    reshape_order : {'C', 'F'}, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering for the
        NLP decision variables. This setting can slightly affect performance.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    Bunch object with the following fields defined:
    sol_X : BarycentricInterpolator
        Found solution for X(t) as a barycentric polynomial.
    sol_U : BarycentricInterpolator
        Found solution for U(t) as a barycentric polynomial.
    t : (n_nodes,) array
        Time points used for collocation.
    X : (n_states, n_nodes) array
        Computed optimal state values X(t).
    U : (n_controls, n_points) array
        Computed optimal control values U(t).
    V : float
        Computed optimal cost at initial point, V(X_0).
    residuals : (n_nodes,) array
        L-infinity norm, max |dynamics(X(t), U(t)) - D * X(t)|, for each t
    status : int
        Reason for algorithm termination.
    message : string
        Verbal description of the termination reason.
    success : bool
        True if the algorithm converged to the desired accuracy (status=0).
    NLP_res : object
        Bunch object containing the full result output by the NLP solver. See
        scipy.optimize.minimize for details.
    '''
    options = {'maxiter': maxiter, **solver_options}
    options['iprint'] = options.get('iprint', verbose)
    options['disp'] = options.get('disp', verbose)

    # Initialize LGR quadrature
    tau, w_hat, D_hat = make_LGR(n_nodes)

    # Time scaling for transformation to LGR points
    r_tau = utilities.deriv_time_map(tau)
    w = w_hat * r_tau
    D = np.matmul(np.diag(1./r_tau), D_hat)

    # Map initial guess to LGR points
    X0 = X_guess[:,:1]
    X_guess, U_guess = utilities.interp_guess(t_guess, X_guess, U_guess, tau)

    n_x, n_u = X_guess.shape[0], U_guess.shape[0]

    collect_vars, separate_vars = utilities.make_reshaping_funs(
        n_x, n_u, n_nodes, order=reshape_order
    )

    # Quadrature integration of running cost
    def cost_fun_wrapper(XU):
        X, U = separate_vars(XU)
        L = cost_fun(X, U).flatten()
        return np.sum(L * w)

    # Wrap running cost gradient
    if callable(cost_grad):
        def jac(XU):
            X, U = separate_vars(XU)
            dLdX, dLdU = cost_grad(X, U)
            return collect_vars(dLdX * w, dLdU * w)
    else:
        jac = cost_grad

    dyn_constr = utilities.make_dynamic_constraint(
        dynamics, D, n_x, n_u, separate_vars, jac=dynamics_jac,
        order=reshape_order
    )
    init_cond_constr = utilities.make_initial_condition_constraint(
        X0, n_u, n_nodes, order=reshape_order
    )
    bound_constr = utilities.make_bound_constraint(
        U_lb, U_ub, n_x, n_nodes, order=reshape_order
    )

    with warnings.catch_warnings():
        # Don't print warnings about unused options
        if not verbose:
            warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)

        if verbose:
            print('\nNumber of LGR nodes: %d' % n_nodes)
            print('---------------------------------------------')

        NLP_res = optimize.minimize(
            fun=cost_fun_wrapper,
            x0=collect_vars(X_guess, U_guess),
            bounds=bound_constr,
            constraints=[dyn_constr, init_cond_constr],
            method='SLSQP',
            tol=tol,
            jac=jac,
            hess=hess,
            options=options
        )

    return DirectSolution(
        NLP_res, tau, cost_fun, dyn_constr, U_lb, U_ub,
        reshape_order, separate_vars
    )
