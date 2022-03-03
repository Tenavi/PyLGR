import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import solve_continuous_are as care

def saturate(U, U_lb, U_ub):
    if U_lb is not None or U_ub is not None:
        if U.ndim < 2:
            U = np.clip(U, U_lb.flatten(), U_ub.flatten())
        else:
            U = np.clip(U, U_lb, U_ub)

    return U

class TemplateOCP:
    '''Defines an optimal control problem (OCP).

    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.

    Parameters
    ----------
    X_bar : (n_states, 1) array
        Goal state, nominal linearization point.
    U_bar : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array or None
        State Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    B : (n_states, n_controls) array or None
        Control Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    U_lb : (n_controls, 1) array, optional
        Lower control saturation bounds.
    U_ub : (n_controls, 1) array, optional
        Upper control saturation bounds.
    '''
    def __init__(
            self, X_bar, U_bar,
            A=None, B=None, Q=None, R=None, U_lb=None, U_ub=None
        ):
        self.X_bar = np.reshape(X_bar, (-1,1))
        self.U_bar = np.reshape(U_bar, (-1,1))

        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        self.U_lb, self.U_ub = U_lb, U_ub

        if self.U_lb is not None:
            self.U_lb = np.reshape(self.U_lb, (-1,1))
        if self.U_ub is not None:
            self.U_ub = np.reshape(self.U_ub, (-1,1))

        # Approximate state matrices numerically if not given
        if A is None:
            A = approx_derivative(
                lambda X: self.dynamics(X, self.U_bar.flatten()),
                self.X_bar.flatten()
            )

        if B is None:
            B = approx_derivative(
                lambda U: self.dynamics(self.X_bar.flatten(), U),
                self.U_bar.flatten()
            )

        # Make Riccati matrix and LQR control gain
        self.P = care(A, B, Q, R)
        self.RB = - np.linalg.solve(R, np.transpose(B))
        self.K = np.matmul(self.RB, self.P)

    def LQR_control(self, X):
        '''
        Evaluates the (saturate) LQR feedback control, U(X)=KX, for each sample
        state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        U = self.U_bar + np.matmul(self.K, X_err)
        U = saturate(U, self.U_lb, self.U_ub)
        if X.ndim < 2:
            U = U.flatten()
        return U

    def U_star(self, X, dVdX):
        '''
        Evaluates the optimal control as a function of the state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        '''
        raise NotImplementedError

    def make_bc(self, X0):
        '''
        Generates a function to evaluate the boundary conditions for a given
        initial condition. Terminal cost is zero so final condition on lambda is
        zero.

        Parameters
        ----------
        X0 : (n_states, 1) array
            Initial condition.

        Returns
        -------
        bc : callable
            Function of X_aug_0 (augmented states at initial time) and X_aug_T
            (augmented states at final time), returning a function which
            evaluates to zero if the boundary conditions are satisfied.
        '''
        X0 = X0.flatten()
        def bc(X_aug_0, X_aug_T):
            return np.concatenate((
                X_aug_0[:self.n_states] - X0, X_aug_T[self.n_states:]
            ))
        return bc

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        raise NotImplementedError

    def running_cost_gradient(self, X, U):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs. Default implementation
        approximates this with central differences.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        L = self.running_cost(X, U)
        dLdX = approx_derivative(lambda X: self.running_cost(X, U), X, f0=L)
        dLdU = approx_derivative(lambda U: self.running_cost(X, U), U, f0=L)
        return dLdX, dLdU

    def Hamiltonian(self, X, U, dVdX):
        '''
        Evaluate the Pontryagin Hamiltonian,
        H(X,U,dVdX) = L(X,U) + [dVdX.T] F(X,U), where L(X,U) is the running
        cost, dVdX is the costate or value gradient, and F(X,U) is the dynamics.
        A necessary condition for optimality is that H(X,U,dVdX) ~ 0 for the
        whole trajectory.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian each each point in time.
        '''
        L = self.running_cost(X, U)
        F = self.dynamics(X, U)
        return L + np.sum(dVdX * F, axis=0, keepdims=True)

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        raise NotImplementedError

    def bvp_dynamics(self, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*) and costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), where U^* is the optimal control.
        '''
        raise NotImplementedError

class LinearSystem(TemplateOCP):
    def __init__(self, n_states, n_controls, U_max, seed=None):
        np.random.seed(seed)

        if U_max is None:
            U_lb, U_ub = None, None
        else:
            U_lb = np.full((n_controls, 1), -U_max)
            U_ub = np.full((n_controls, 1), U_max)

        # Generate random dynamic matrices and make dynamics and cost functions
        self.A = np.random.randn(n_states, n_states)
        self.B = np.random.randn(n_states, n_controls)
        self.Q = 1.
        self.R = 5.
        Q = self.Q/2. * np.identity(n_states)
        R = self.R/2. * np.identity(n_controls)

        super().__init__(
            np.zeros((n_states, 1)), np.zeros((n_controls, 1)),
            self.A, self.B, Q, R, U_lb=U_lb, U_ub=U_ub
        )

    def U_star(self, X, dVdX):
        '''
        Evaluates the optimal control as a function of the state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        '''
        U = self.U_bar + np.matmul(self.RB/2., dVdX)
        U = saturate(U, self.U_lb, self.U_ub)
        if X.ndim < 2:
            U = U.flatten()
        return U

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        return (
            np.sum(self.Q/2. * X**2, axis=0) + np.sum(self.R/2. * U**2, axis=0)
        )

    def running_cost_gradient(self, X, U):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        return self.Q * X, self.R * U

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        return np.matmul(self.A, X) + np.matmul(self.B, U)

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        X_aug : (2*n_states, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*) and costate dynamics
            dAdt = -dH/dX(X,U^*,dVdX), where U^* is the optimal control.
        '''
        X = X_aug[:self.n_states]
        dVdX = X_aug[self.n_states:]
        U = self.U_star(X, dVdX)

        dXdt = self.dynamics(X, U)
        dAdX = - (self.Q * X + np.matmul(self.A.T, dVdX))

        return np.vstack((dXdt, dAdX))

class VanDerPol(TemplateOCP):
    def __init__(self):
        # Dynamics parameters
        self.mu = 2.
        self.b = 1.5

        # Cost parameters
        self.Wx = 1/2
        self.Wy = 1.
        self.Wu = 4.

        # Control constraints
        U_max = 1.
        if U_max is None:
            U_lb, U_ub = None, None
        else:
            U_lb = np.full((1, 1), -U_max)
            U_ub = np.full((1, 1), U_max)

        # Linearization point
        X_bar = np.zeros((2,1))
        U_bar = 0.

        # Dynamics linearized around X_bar (dxdt ~= Ax + Bu)
        A = [[0., 1.], [-1., self.mu]]
        B = [[0.], [self.b]]
        self.B = np.array(([[0.], [self.b]]))

        # Cost matrices
        Q = np.diag([self.Wx / 2., self.Wy / 2.])
        R = [[self.Wu / 2.]]

        super().__init__(X_bar, U_bar, A, B, Q, R, U_lb=U_lb, U_ub=U_ub)

    def U_star(self, X, dVdX):
        '''
        Evaluates the optimal control as a function of the state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        '''
        U = - self.b / self.Wu * dVdX[1:]
        U = saturate(U, self.U_lb, self.U_ub)
        if X.ndim < 2:
            U = U.flatten()
        return U

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        x1 = X[:1]
        x2 = X[1:]
        return np.sum(
            self.Wx/2. * x1**2 + self.Wy/2. * x2**2 + self.Wu/2. * U**2,
            axis=0, keepdims=True
        )

    def running_cost_gradient(self, X, U):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        x1 = X[:1]
        x2 = X[1:]
        dLdX = np.concatenate((self.Wx * x1, self.Wy * x2))
        dLdU = self.Wu * U
        return dLdX, dLdU

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        x1 = X[:1]
        x2 = X[1:]
        dx1dt = x2
        dx2dt = self.mu * (1. - x1**2) * x2 - x1 + self.b * U
        return np.concatenate((dx1dt, dx2dt))

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        X_aug : (2*n_states, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*) and costate dynamics
            dAdt = -dH/dX(X,U^*,dVdX), where U^* is the optimal control.
        '''
        x1 = X_aug[:1]
        x2 = X_aug[1:2]

        # Costate
        A1 = X_aug[2:3]
        A2 = X_aug[3:4]

        # Optimal control as a function of the costate
        U = self.U_star(X_aug[:2], X_aug[2:])

        # State dynamics
        dx1dt = x2
        dx2dt = self.mu * (1. - x1**2) * x2 - x1 + self.b * U

        # Costate dynamics
        dA1dt = -self.Wx * x1 + A2 * (2.*self.mu*x1*x2 + 1.)
        dA2dt = -self.Wy * x2 - A1 - A2 * self.mu * (1. - x1**2)

        return np.vstack((dx1dt, dx2dt, dA1dt, dA2dt))

class Satellite(TemplateOCP):
    def __init__(self):
        n_controls = 3

        # Dynamics parameters
        self.J = np.array([
            [59.22, -1.14, -0.8],
            [-1.14, 40.56, 0.1],
            [-0.8, 0.1, 57.6]
        ])
        self.JT = self.J.T
        self.Jinv = np.linalg.inv(self.J)
        self.JinvT = self.Jinv.T

        # Cost parameters
        self.Wq = 1.
        self.Ww = 1.
        self.Wu = 1.

        U_max = 0.3
        if U_max is None:
            U_lb, U_ub = None, None
        else:
            U_lb = np.full((n_controls, 1), -U_max)
            U_ub = np.full((n_controls, 1), U_max)

        ##### Makes LQR controller #####

        # Linearization point
        X_bar = np.zeros((7,1))
        X_bar[0] = 1.
        U_bar = np.zeros((n_controls, 1))

        # Dynamics linearized around X_bar (dxdt ~= Ax + Bu)
        A = np.zeros((7,7))
        A[1:4,4:] = np.identity(3) / 2.
        B = np.vstack((np.zeros((4,3)), -self.Jinv))

        # Cost matrices (ignores scalar component of quaternion)
        Q = np.zeros((7,7))
        Q[1:4,1:4] = (self.Wq / 2.) * np.identity(3)
        Q[4:,4:] = (self.Ww / 2.) * np.identity(3)

        R = (self.Wu / 2.) * np.identity(3)

        super().__init__(X_bar, U_bar, A, B, Q, R, U_lb=U_lb, U_ub=U_ub)

    def _break_state(self, X):
        '''
        Break up the state vector, [q0, q, w], into individual pieced. If
        X.shape[0] >= 14, then splits off the remaining states (costates) too.

        Parameters
        ----------
        X : (7,), (14,), (7, n_samples), or (14, n_samples) array
            State (and costate, if X.shape[0] > 7)

        Returns
        -------
        q0 : (1,) or (1, n_samples) array
            Scalar component of quaternion
        q : (4,) or (4, n_samples) array
            Vector component of quaternion
        w : (3,) or (3, n_samples) array
            Angular momenta
        A0 : (1,) or (1, n_samples) array
            Costate of scalar component of quaternion
        Aq : (4,) or (4, n_samples) array
            Costate of vector component of quaternion
        Aw : (3,) or (3, n_samples) array
            Costate of angular momenta
        '''
        q0 = X[:1]
        q = X[1:4]
        w = X[4:7]

        if X.shape[0] > 7:
            A0 = X[7:8]
            Aq = X[8:11]
            Aw = X[11:14]

            return q0, q, w, A0, Aq, Aw

        return q0, q, w

    def U_star(self, X, dVdX):
        '''
        Evaluates the optimal control as a function of the state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        '''
        U = np.matmul(self.JinvT, dVdX[4:]) / self.Wu
        U = saturate(U, self.U_lb, self.U_ub)
        if X.ndim < 2:
            U = U.flatten()
        return U

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        _, q, w = self._break_state(X[:7])

        L = np.sum(
            self.Wq/2. * np.sum(q**2, axis=0, keepdims=True)
            + self.Ww/2. * np.sum(w**2, axis=0, keepdims=True)
            + self.Wu/2. * np.sum(U**2, axis=0, keepdims=True),
            axis=0, keepdims=True
        )

        return L

    def running_cost_gradient(self, X, U):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        q0, q, w = self._break_state(X[:7])

        dLdX = np.concatenate((np.zeros_like(q0), self.Wq * q, self.Ww * w))
        dLdU = self.Wu * U
        return dLdX, dLdU

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        flat_out = X.ndim < 2

        q0, q, w = self._break_state(X[:7].reshape(7,-1))

        Jw = np.matmul(self.J, w)

        dq0dt = - 0.5 * np.sum(w * q, axis=0, keepdims=True)
        dqdt = 0.5 * (-np.cross(w, q, axis=0) + q0 * w)

        dwdt = np.cross(w, Jw, axis=0) + U.reshape(3,-1)
        dwdt = np.matmul(-self.Jinv, dwdt)

        dXdt = np.vstack((dq0dt, dqdt, dwdt))
        if flat_out:
            dXdt = dXdt.flatten()
        return dXdt

    def bvp_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP.'''

        # Optimal control as a function of the costate
        U = self.U_star(X_aug[:7], X_aug[7:14])

        q0, q, w, A0, Aq, Aw = self._break_state(X_aug)

        Jw = np.matmul(self.J, w)
        JAw = np.matmul(self.JinvT, Aw)

        # State dynamics
        dq0dt = - 0.5 * np.sum(w * q, axis=0, keepdims=True)
        dqdt = 0.5 * (-np.cross(w, q, axis=0) + q0 * w)

        dwdt = np.cross(w, Jw, axis=0) + U.reshape(3,-1)
        dwdt = np.matmul(-self.Jinv, dwdt)

        # Costate dynamics
        dA0dt = - 0.5 * np.sum(w * Aq, axis=0, keepdims=True)

        dAqdt = (
            self.Wq * (self.X_bar[1:4] - q)
            + 0.5 * (- np.cross(w, Aq, axis=0) + A0 * w)
        )

        dAwdt = (
            self.Ww * (self.X_bar[4:] - w)
            + 0.5 * (np.cross(q, Aq, axis=0) - q0*Aq + A0*q)
            + np.matmul(-self.JT, np.cross(w, JAw, axis=0))
            + np.cross(Jw, JAw, axis=0)
        )

        return np.vstack((dq0dt, dqdt, dwdt, dA0dt, dAqdt, dAwdt))
