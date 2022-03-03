import pytest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from PyLGR import utilities

TOL = 1e-10

def _generate_nonlinear_dynamics(n_x, n_u, poly_deg=5):
    A = np.random.randn(n_x, n_x+n_u)
    # Make random polynomials of X and U with no constant term or linear term
    X_coefs = np.hstack((np.zeros((n_x,2)), np.random.randn(n_x, poly_deg-2)))
    X_polys = [
        np.polynomial.polynomial.Polynomial(X_coefs[i], domain=[-10., 10.])
        for i in range(n_x)
    ]
    U_coefs = np.hstack((np.zeros((n_u,1)), np.random.randn(n_u, poly_deg-1)))
    U_polys = [
        np.polynomial.polynomial.Polynomial(U_coefs[i], domain=[-10., 10.])
        for i in range(n_u)
    ]

    def dynamics(X, U):
        X_poly = np.vstack(
            np.atleast_2d([X_polys[i](X[i]) for i in range(n_x)])
        )
        U_poly = np.vstack(
            np.atleast_2d([U_polys[i](U[i]) for i in range(n_u)])
        )
        return np.matmul(A, np.vstack((X_poly, U_poly)))

    return dynamics

def test_time_map():
    t_orig = np.linspace(0.,10.)
    tau = utilities.time_map(t_orig)
    t = utilities.invert_time_map(tau)
    assert np.allclose(t, t_orig)
    assert np.allclose(tau, (t - 1.)/(t + 1.))

@pytest.mark.parametrize('n', [10,15])
@pytest.mark.parametrize('d', [1,2])
def test_interp_initial_guess(n, d):
    '''
    Test that the interpolation code recovers the original points if tau = t.
    '''
    t = np.linspace(0.,10.,n)
    tau = utilities.time_map(t)

    X = np.cos(t) * t
    U = np.sin(-t)

    X = np.atleast_2d(X)
    U = np.atleast_2d(U)
    for k in range(d-1):
        X = np.vstack((X, X[0] + k))
        U = np.vstack((U, U[0] - k))

    X_interp, U_interp = utilities.interp_guess(t, X, U, tau)

    assert np.allclose(X_interp, X)
    assert np.allclose(U_interp, U)

@pytest.mark.parametrize('n_states', [1,3])
@pytest.mark.parametrize('n_controls', [1,2])
@pytest.mark.parametrize('order', ['F','C'])
def test_reshaping_funs(n_states, n_controls, order):
    n_nodes = 42
    X = np.random.randn(n_states, n_nodes)
    U = np.random.randn(n_controls, n_nodes)

    collect_vars, separate_vars = utilities.make_reshaping_funs(
        n_states, n_controls, n_nodes, order=order
    )

    XU = collect_vars(X, U)
    assert XU.ndim == 1
    assert XU.shape[0] == (n_states + n_controls) * n_nodes

    _X, _U = separate_vars(XU)
    assert np.allclose(_X, X)
    assert np.allclose(_U, U)

@pytest.mark.parametrize('order', ['F','C'])
def test_dynamics_setup(order):
    '''
    Test that the dynamics constraints are instantiated properly. To this end,
    make a random polynomial which represents the true state. Check that the
    constraint function is zero when evaluated for this state, and not zero when
    evaluated on a significantly perturbed state.
    '''
    n_x, n_u, n_t = 3, 2, 13

    collect_vars, separate_vars = utilities.make_reshaping_funs(
        n_x, n_u, n_t, order=order
    )

    tau, w, D = utilities.make_LGR(n_t)

    # Generate random polynomials of degree n-1 for the state
    coef = np.random.randn(n_x, n_t)
    Poly_X = [np.polynomial.polynomial.Polynomial(coef[d]) for d in range(n_x)]
    # control is ignored so can be anything
    XU = collect_vars(
        np.vstack([P(tau) for P in Poly_X]), np.random.randn(n_u, n_t)
    )

    # The derivative is the polynomial derivative
    def dXdt(X, U):
        return np.vstack([P.deriv()(tau) for P in Poly_X])

    constr = utilities.make_dynamic_constraint(
        dXdt, D, n_x, n_u, separate_vars, order=order
    )

    assert constr.lb == constr.ub == 0.

    # Check that evaluating the constraint function for the true state returns 0
    assert np.all(np.abs(constr.fun(XU)) < TOL)
    # Check that evaluating the constraint function for perturbed states does
    # not return 0
    with pytest.raises(AssertionError):
        XU = XU + np.random.randn(XU.shape[0])*10.
        assert np.all(np.abs(constr.fun(XU)) < TOL)

@pytest.mark.parametrize('n_nodes', [3,4,7,8])
@pytest.mark.parametrize('order', ['F','C'])
def test_dynamics_setup_Jacobian(n_nodes, order):
    '''
    Use numerical derivatives to verify the sparse dynamics constraint Jacobian.
    '''
    np.random.seed(42)

    n_x, n_u, n_t = 3, 2, n_nodes

    collect_vars, separate_vars = utilities.make_reshaping_funs(
        n_x, n_u, n_t, order=order
    )

    tau, w, D = utilities.make_LGR(n_t)

    # Generate random states and controls
    X = np.random.randn(n_x, n_t)
    U = np.random.randn(n_u, n_t)
    XU = collect_vars(X, U)

    # Generate some random dynamics
    dXdt = _generate_nonlinear_dynamics(n_x, n_u)

    constr = utilities.make_dynamic_constraint(
        dXdt, D, n_x, n_u, separate_vars, order=order,
        finite_diff_method='3-point'
    )

    constr_Jac = constr.jac(XU)
    expected_Jac = approx_derivative(constr.fun, XU)

    assert constr_Jac.shape == (n_x*n_t, (n_x + n_u)*n_t)
    assert np.allclose(constr_Jac.toarray(), expected_Jac)

@pytest.mark.parametrize('n_nodes', [3,4,7,8])
@pytest.mark.parametrize('order', ['F','C'])
def test_init_cond_setup(n_nodes, order):
    '''
    Check that the initial condition matrix multiplication returns the correct
    points.
    '''
    n_x, n_u, n_t = 3, 2, n_nodes

    collect_vars, separate_vars = utilities.make_reshaping_funs(
        n_x, n_u, n_t, order=order
    )

    # Generate random states and controls
    X = np.random.randn(n_x, n_t)
    U = np.random.randn(n_u, n_t)
    X0 = X[:,:1]
    XU = collect_vars(X, U)

    constr = utilities.make_initial_condition_constraint(
        X0, n_u, n_t, order=order
    )

    assert np.all(constr.lb == X0.flatten())
    assert np.all(constr.ub == X0.flatten())
    assert constr.A.shape == (n_x, (n_x+n_u)*n_t)
    # Check that evaluating the multiplying the linear constraint matrix
    # times the full state-control vector returns the initial condtion
    assert np.allclose(constr.A @ XU, X0.flatten())

@pytest.mark.parametrize('n_nodes', [3,4,5])
@pytest.mark.parametrize('order', ['F','C'])
@pytest.mark.parametrize(
    'U_lb', [None, -1., [-1.], [-1.,-2.], [-np.inf, -np.inf], [-np.inf,-2.]]
)
def test_bounds_setup(n_nodes, order, U_lb):
    '''
    Test that Bounds are initialized correctly for all different kinds of
    possible control bounds.
    '''
    if U_lb is None:
        U_ub = None
        n_u = 1
    elif np.isinf(U_lb).all():
        U_lb = None
        U_ub = None
        n_u = 2
    else:
        U_lb = np.reshape(U_lb, (-1,1))
        U_ub = - U_lb
        n_u = U_lb.shape[0]

    n_x, n_t = 3, n_nodes

    constr = utilities.make_bound_constraint(
        U_lb, U_ub, n_x, n_t, order=order
    )

    if U_lb is None and U_ub is None:
        assert constr is None
    else:
        assert constr.lb.shape == constr.ub.shape == ((n_x+n_u)*n_t,)

        # No state constraints
        assert np.isinf(constr.lb[:n_x*n_nodes]).all()
        assert np.isinf(constr.ub[:n_x*n_nodes]).all()

        # Verify control constraints
        collect_vars, _ = utilities.make_reshaping_funs(
            n_x, n_u, n_t, order=order
        )

        if U_lb is None:
            assert np.isinf(constr.lb[n_x*n_nodes:]).all()
        else:
            U = np.tile(U_lb, (1,n_nodes))
            XU = collect_vars(np.random.randn(n_x, n_t), U)
            assert np.allclose(constr.lb[n_x*n_nodes:], XU[n_x*n_nodes:])

        if U_ub is None:
            assert np.isinf(constr.ub[n_x*n_nodes:]).all()
        else:
            U = np.tile(U_ub, (1,n_nodes))
            XU = collect_vars(np.random.randn(n_x, n_t), U)
            assert np.allclose(constr.ub[n_x*n_nodes:], XU[n_x*n_nodes:])
