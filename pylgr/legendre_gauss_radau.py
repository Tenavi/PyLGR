import numpy as np
from scipy import special

def _check_size_n(n_nodes):
    '''
    We only define the LGR quadrature for n_nodes >= 3. This utility function
    checks to make sure n_nodes is the right size.

    Parameters
    ----------
    n_nodes : int
        Number of collocation nodes.

    Returns
    -------
    n_nodes : int
        Number of collocation nodes, only returned if n_nodes >= 3.

    Raises
    ------
    ValueError
        If n_nodes < 3.
    '''
    if int(n_nodes) < 3:
        raise ValueError('Number of nodes must be at least n_nodes >= 3.')
    return int(n_nodes)

def legendre(x, n):
    '''
    Evaluates the nth order Legendre polynomial P_n (x) at a single point x in
    [-1,1]. Wraps the scipy.special.lpn function for convenience.

    Parameters
    ----------
    x : float
        Evaluation point in [-1,1].
    n : int
        Polynomial order.

    Returns
    -------
    P : float
        Evaluated Legendre polynomial P_n (x).
    '''
    P, _ = special.lpn(n, x)
    return P[-1]

def make_LGR_nodes(n_nodes):
    '''
    Constructs collocation points for LGR quadrature. These are the roots of
    P_n (tau) + P_{n-1} (tau), where P_n is the nth order Legendre polynomial
    and n = n_nodes. One can show (see e.g.
    https://mathworld.wolfram.com/JacobiPolynomial.html) that
    P_n (tau) + P_{n-1} (tau) = (1 + tau) P^(0,1)_n (tau), where P^(0,1)_n is
    the nth order Jacobi polynomial with alpha = 0 and beta = 1.

    Parameters
    ----------
    n_nodes : int
        Number of collocation nodes. Must be n_nodes >= 3.

    Returns
    -------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1,1).
    '''
    n = _check_size_n(n_nodes)
    tau, _ = special.roots_jacobi(n-1, alpha=0, beta=1)
    return np.concatenate(([-1.], tau))

def make_LGR_weights(tau):
    '''
    Constructs the LGR quadrature weights w. These are given by
        w[0] = 2 / n**2,
        w[i] = (1 - tau[i]) / (n**2 + P_{n-1} (tau[i])),    i = 1, ..., n-1
    where n = n_nodes is the number of collocation points and P_{n-1} is the
    (n-1)th order Legendre polynomial. See Fariba and Ross (2008).

    Parameters
    ----------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1,1).

    Returns
    -------
    w : (n_nodes,) array
        LGR quadrature weights corresponding to the collocation points tau.
    '''
    n = _check_size_n(tau.shape[0])
    w = np.empty_like(tau)
    w[0] = 2. / n**2
    for i in range(1,n):
        w[i] = (1. - tau[i])/(n * legendre(tau[i], n-1))**2
    return w

def make_LGR_diff_matrix(tau):
    '''
    Constructs the LGR differentiation_matrix D. The entries of D are given by
        D[i,j] = -(n-1)*(n+1)/4,    i = j = 0
        D[i,j] = 1 / (2 - 2*tau[i]),    1 <= i = j <= n-1
        D[i,j] =
            P_{n-1} (tau[i]) / P_{n-1} (tau[j])
            * (1 - tau[j]) / (1 - tau[i])
            * 1 / (tau[i] - tau[j]),    otherwise
    where n = n_nodes is the number of collocation points and P_{n-1} is the
    (n-1)th order Legendre polynomial. See Fariba and Ross (2008).

    Parameters
    ----------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1,1).

    Returns
    -------
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix corresponding to the collocation points tau.
    '''
    n = _check_size_n(tau.shape[0])
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = (
                    legendre(tau[i], n-1) / legendre(tau[j], n-1)
                    * (1. - tau[j]) / ((1. - tau[i]) * (tau[i] - tau[j]))
                )
            elif i == j == 0:
                D[i,j] = -(n-1)*(n+1)/4.
            else:
                D[i,j] = 1. / (2. * (1. - tau[i]))
    return D

def make_LGR(n_nodes):
    '''
    Constructs LGR collocation points, integration weights, and differentiation
    matrix. See make_LGR_nodes, make_LGR_weights, make_LGR_diff_matrix and
    Fariba and Ross (2008) for details.

    Parameters
    ----------
    n_nodes : int
        Number of collocation nodes. Must be n_nodes >= 3.

    Returns
    -------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1,1).
    w : (n_nodes,) array
        LGR quadrature weights corresponding to the collocation points tau.
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix corresponding to the collocation points tau.
    '''
    tau = make_LGR_nodes(n_nodes)
    w = make_LGR_weights(tau)
    D = make_LGR_diff_matrix(tau)
    return tau, w, D
