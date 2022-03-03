import pytest

import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

from PyLGR import solve_ocp, utilities
from PyLGR import legendre_gauss_radau as LGR

from .test_data import example_problems

pytestmark = pytest.mark.parametrize('order', 'C')
solver_tols = {'SLSQP': 1e-05, 'trust-constr': 1e-05}

def _assert_converged(PS_sol, tol):
    assert PS_sol.success
    assert PS_sol.residuals.max() < tol

def _get_LQR_guess(OCP, t1, X0, tol):
    def closed_loop_dynamics(t, X):
        U = OCP.LQR_control(X)
        return OCP.dynamics(X, U)

    LQR_sol = solve_ivp(
        closed_loop_dynamics, [0.,t1], X0.flatten(), atol=tol/1000, rtol=tol
    )

    t_LQR = LQR_sol.t
    X_LQR = LQR_sol.y
    U_LQR = OCP.LQR_control(X_LQR)

    LQR_cost = OCP.running_cost(X_LQR, U_LQR)
    LQR_cost = np.trapz(LQR_cost, x=t_LQR)

    return t_LQR, X_LQR, U_LQR, LQR_cost

def _get_BVP_sol(OCP, t_guess, X_guess, tol):
    X0 = X_guess[:,:1]
    X_aug_guess = np.vstack((X_guess, 2.*np.matmul(OCP.P, X_guess)))
    BVP_sol = solve_bvp(
        OCP.bvp_dynamics, OCP.make_bc(X0), t_guess, X_aug_guess, tol=tol
    )

    t_opt = BVP_sol.x
    X_opt = BVP_sol.y[:OCP.n_states]
    U_opt = OCP.U_star(X_opt, BVP_sol.y[OCP.n_states:])

    opt_cost = OCP.running_cost(X_opt, U_opt)
    opt_cost = np.trapz(opt_cost, x=t_opt)

    return t_opt, X_opt, U_opt, opt_cost

def _plot_results(t_ref, X_ref, U_ref, PS_sol, feas_sol, problem_name):
    plt.figure()

    plt.plot(t_ref, X_ref.T, 'k')
    plt.plot(t_ref, PS_sol.sol_X(t_ref).T, '--')

    plt.gca().set_xlim(0.,t_ref.max())
    plt.title(problem_name + ": interpolated states")

    plt.figure()

    plt.plot(PS_sol.t, PS_sol.X.T, '^k')
    plt.plot(feas_sol.t, feas_sol.y.T, '--')

    plt.gca().set_xlim(0.,t_ref.max())
    plt.gca().set_ylim(2.*PS_sol.X.min(), 2.*PS_sol.X.max())
    plt.title(problem_name + ": propagated states")

    plt.figure()

    plt.plot(t_ref, U_ref.T, 'k')
    plt.plot(PS_sol.t, PS_sol.U.T, '^k')
    plt.plot(t_ref, PS_sol.sol_U(t_ref).T, '--')

    plt.gca().set_xlim(0.,t_ref.max())
    plt.title(problem_name + ": controls")

    plt.show()

@pytest.mark.parametrize('U_max', [None,.25])
@pytest.mark.parametrize('n_nodes', [11,16])
@pytest.mark.parametrize('solver', solver_tols.keys())
def test_LQR(U_max, order, n_nodes, solver):
    '''
    Evaluate the solve_ocp method against a reference LQR solution with and
    without control saturation constraints.
    '''
    tol = solver_tols[solver]
    plot_sims = False
    verbose = 0
    n_x, n_u, n_t = 3, 2, n_nodes

    OCP = example_problems.LinearSystem(n_x, n_u, U_max, seed=1234)

    X0 = np.array([[-0.0085], [1.6218], [1.1564]])
    t1 = 30.

    start_time = time.time()

    t_LQR, X_LQR, U_LQR, LQR_cost = _get_LQR_guess(OCP, t1, X0, tol)
    t_opt, X_opt, U_opt, opt_cost = _get_BVP_sol(OCP, t_LQR, X_LQR, tol/10.)

    print(
        '\nIndirect solution time: %.4fs. Optimal cost: %.4f'
        % (time.time() - start_time, opt_cost)
    )

    start_time = time.time()

    PS_sol = solve_ocp(
        OCP.dynamics, OCP.running_cost, t_LQR, X_LQR, U_LQR,
        U_lb=OCP.U_lb, U_ub=OCP.U_ub, n_nodes=n_nodes,
        cost_jac=OCP.running_cost_gradient, tol=tol, maxiter=10000,
        reshape_order=order, solver=solver, verbose=verbose
    )

    print(
        'Direct solution time:   %.4fs. LGR PS cost:  %.4f'
        % (time.time() - start_time, PS_sol.V)
    )

    _assert_converged(PS_sol, tol)
    assert PS_sol.V < opt_cost * 1.01

    if plot_sims:
        # Propagate system with open loop PS controls to assess feasibility
        def closed_loop_dynamics(t, X):
            U = example_problems.saturate(PS_sol.sol_U(t), OCP.U_lb, OCP.U_ub)
            return OCP.dynamics(X, U)
        feas_sol = solve_ivp(
            closed_loop_dynamics, [0.,t1], X0.flatten(), atol=tol/1000, rtol=tol
        )
        problem_name = 'LQR'
        if U_max is not None:
            problem_name = 'Constrained LQR'
        _plot_results(
            t_opt, X_opt, U_opt, PS_sol, feas_sol, problem_name
        )

@pytest.mark.parametrize(
    'solver,n_nodes',
    [('trust-constr', 11), ('SLSQP', 11), ('SLSQP', 32)]
)
def test_van_der_pol(order, n_nodes, solver):
    '''
    Evaluate the solve_ocp method against a reference solution obtained with an
    indirect method.
    '''
    tol = solver_tols[solver]
    plot_sims = False
    verbose = 0

    OCP = example_problems.VanDerPol()

    X0 = np.array([[-2.3997], [3.2243]])
    t1 = 30.

    start_time = time.time()

    t_LQR, X_LQR, U_LQR, LQR_cost = _get_LQR_guess(OCP, t1, X0, tol)
    t_opt, X_opt, U_opt, opt_cost = _get_BVP_sol(OCP, t_LQR, X_LQR, tol/10.)

    print(
        '\nIndirect solution time: %.4fs. Optimal cost: %.4f'
        % (time.time() - start_time, opt_cost)
    )

    start_time = time.time()

    PS_sol = solve_ocp(
        OCP.dynamics, OCP.running_cost, t_LQR, X_LQR, U_LQR,
        U_lb=OCP.U_lb, U_ub=OCP.U_ub, n_nodes=n_nodes,
        cost_jac=OCP.running_cost_gradient, tol=tol, maxiter=10000,
        reshape_order=order, solver=solver, verbose=verbose
    )

    print(
        'Direct solution time:   %.4fs. LGR PS cost:  %.4f'
        % (time.time() - start_time, PS_sol.V)
    )

    _assert_converged(PS_sol, tol)
    assert PS_sol.V < opt_cost * 1.01

    if plot_sims:
        # Propagate system with open loop PS controls to assess feasibility
        def closed_loop_dynamics(t, X):
            U = example_problems.saturate(PS_sol.sol_U(t), OCP.U_lb, OCP.U_ub)
            return OCP.dynamics(X, U)
        feas_sol = solve_ivp(
            closed_loop_dynamics, [0.,t1], X0.flatten(), atol=tol/1000, rtol=tol
        )
        _plot_results(
            t_opt, X_opt, U_opt, PS_sol, feas_sol, 'Van der Pol'
        )

@pytest.mark.parametrize(
    'solver,n_nodes',
    [('trust-constr', 11), ('SLSQP', 11), ('SLSQP', 32), ('SLSQP', 40)]
)
def test_satellite(order, n_nodes, solver):
    '''
    Evaluate the solve_ocp method against a reference solution obtained with an
    indirect method.
    '''
    tol = solver_tols[solver]
    plot_sims = False
    verbose = 0

    OCP = example_problems.Satellite()

    X0 = np.array([
        [0.17678],
        [0.91856],
        [0.17678],
        [0.30619],
        [0.],
        [0.],
        [0.]
    ])
    t1 = 120.

    start_time = time.time()

    t_LQR, X_LQR, U_LQR, LQR_cost = _get_LQR_guess(OCP, t1, X0, tol)
    t_opt, X_opt, U_opt, opt_cost = _get_BVP_sol(OCP, t_LQR, X_LQR, tol/10.)

    print(
        '\nIndirect solution time: %.4fs. Optimal cost: %.4f'
        % (time.time() - start_time, opt_cost)
    )

    start_time = time.time()

    PS_sol = solve_ocp(
        OCP.dynamics, OCP.running_cost, t_LQR, X_LQR, U_LQR,
        U_lb=OCP.U_lb, U_ub=OCP.U_ub, n_nodes=n_nodes, tol=tol, maxiter=10000,
        cost_jac=OCP.running_cost_gradient,
        reshape_order=order, solver=solver, verbose=verbose
    )

    print(
        'Direct solution time:   %.4fs. LGR PS cost:  %.4f'
        % (time.time() - start_time, PS_sol.V)
    )

    _assert_converged(PS_sol, tol)
    if n_nodes >= 40:
        assert PS_sol.V < opt_cost * 1.01

    if plot_sims:
        # Propagate system with open loop PS controls to assess feasibility
        def closed_loop_dynamics(t, X):
            U = example_problems.saturate(PS_sol.sol_U(t), OCP.U_lb, OCP.U_ub)
            return OCP.dynamics(X, U)
        feas_sol = solve_ivp(
            closed_loop_dynamics, [0.,t1], X0.flatten(), atol=tol/1000, rtol=tol
        )
        _plot_results(
            t_opt, X_opt, U_opt, PS_sol, feas_sol, 'Satellite'
        )