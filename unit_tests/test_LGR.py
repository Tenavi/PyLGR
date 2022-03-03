import pytest

import os
import numpy as np
import scipy.io

from pylgr import legendre_gauss_radau as LGR

TOL = 1e-10

# Reference data computed with Matlab scripts
test_data_path = os.path.join('unit_tests', 'test_data', 'LGR_diff.mat')
LGR_diff_mat = scipy.io.loadmat(test_data_path)
LGR_diff_reference = {}

# Process data to make it easier to deal with
for case_num in LGR_diff_mat:
    if case_num[0] == 'n':
        case_key = int(case_num[1:])
        LGR_diff_reference[case_key] = {}
        for key in ['tau','w','D']:
            LGR_diff_reference[case_key][key] = np.squeeze(
                LGR_diff_mat[case_num][key][0][0]
            )

@pytest.mark.parametrize('n', [-1,0,1,2])
@pytest.mark.parametrize('fun_name', ['make_LGR_nodes', 'make_LGR'])
def test_make_LGR_small_n(n, fun_name):
    fun = getattr(LGR, fun_name)
    with pytest.raises(ValueError):
        fun(n)

@pytest.mark.parametrize('n', [0,1,2])
@pytest.mark.parametrize(
    'fun_name', ['make_LGR_weights', 'make_LGR_diff_matrix']
)
def test_make_LGR_small_tau(n, fun_name):
    fun = getattr(LGR, fun_name)
    tau = np.random.rand(n)
    with pytest.raises(ValueError):
        fun(tau)

@pytest.mark.parametrize('n', LGR_diff_reference.keys())
def test_make_LGR(n):
    tau, w, D = LGR.make_LGR(n)
    assert np.allclose(tau, LGR_diff_reference[n]['tau'])
    assert np.allclose(w, LGR_diff_reference[n]['w'])
    assert np.allclose(D, LGR_diff_reference[n]['D'])

@pytest.mark.parametrize('n', np.arange(3,19,3))
def test_LGR_basic_int_diff(n):
    '''
    Tests some basic identities: w*x=0, Dx=ones, Dx**2=2x, w*(Dx)=2.
    '''
    tau, w, D = LGR.make_LGR(n)
    assert np.abs(np.dot(w, tau)) < TOL
    assert np.allclose(np.matmul(D, tau), np.ones_like(tau))
    assert np.allclose(np.matmul(D, tau**2), 2.*tau)
    assert np.isclose(np.dot(w, np.matmul(D, tau)), 2.)

@pytest.mark.parametrize('n', np.arange(3,19,3))
def test_LGR_integrate(n):
    '''
    LGR should be able to integrate a polynomial of degree 2n-2 to machine
    precision.
    '''
    # Generate a random polynomial of degree 2n-2
    degree = 2*n - 2
    coef = np.random.randn(degree+1)
    P = np.polynomial.polynomial.Polynomial(coef)
    expected_integral = P.integ(lbnd=-1.)(1.)

    tau = LGR.make_LGR_nodes(n)
    w = LGR.make_LGR_weights(tau)
    LGR_integral = np.dot(w, P(tau))

    assert np.isclose(LGR_integral, expected_integral)

@pytest.mark.parametrize('n', np.arange(3,19,3))
def test_LGR_differentiate(n):
    '''
    LGR should be able to differentiate a polynomial of degree n-1 to machine
    precision.
    '''
    # Generate a random polynomial of degree n-1
    degree = n - 1
    coef = np.random.randn(degree+1)
    P = np.polynomial.polynomial.Polynomial(coef)

    tau, w, D = LGR.make_LGR(n)
    LGR_derivative = np.matmul(D, P(tau))

    assert np.allclose(LGR_derivative, P.deriv()(tau))

@pytest.mark.parametrize('n_dims', [1,2,3])
def test_LGR_multivariate_integrate(n_dims):
    '''
    Test integration of degree 2n-2 polymomials in n_dims dimensions.
    '''
    n = 10

    # Generate random polynomial of degree 2n-2
    degree = 2*n - 2
    coef = np.random.randn(n_dims, degree+1)
    P = [np.polynomial.polynomial.Polynomial(coef[d]) for d in range(n_dims)]
    expected_integral = [P[d].integ(lbnd=-1.)(1.) for d in range(n_dims)]

    tau = LGR.make_LGR_nodes(n)
    w = LGR.make_LGR_weights(tau)
    P_mat = np.vstack([P[d](tau) for d in range(n_dims)])
    LGR_integral = np.matmul(P_mat, w)

    assert np.allclose(LGR_integral, expected_integral)

@pytest.mark.parametrize('n_dims', [1,2,3])
def test_LGR_multivariate_differentiate(n_dims):
    '''
    Test differentiation of degree n-1 polymomials in n_dims dimensions.
    '''
    n = 10

    # Generate a random polynomial of degree n-1
    degree = n - 1
    coef = np.random.randn(n_dims, degree+1)
    P = [np.polynomial.polynomial.Polynomial(coef[d]) for d in range(n_dims)]

    tau, w, D = LGR.make_LGR(n)
    P_mat = np.vstack([P[d](tau) for d in range(n_dims)])

    expected_derivative = [P[d].deriv()(tau) for d in range(n_dims)]
    LGR_derivative = np.matmul(P_mat, D.T)

    assert np.allclose(LGR_derivative, expected_derivative)
