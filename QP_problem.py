#! Based on https://github.com/locuslab/DC3

import torch
import numpy as np
import osqp
import cyipopt as ipopt
from scipy.sparse import csc_matrix

import time

from abc import ABC, abstractmethod

torch.set_default_dtype(torch.float64)

class SimpleProblem(ABC):
    """
    minimize_y 1/2 * y^T Q y + p^Ty
    s.t.       Ay =  b
               Gy <= d
    """

    def __init__(self, Q, p, A, G, b, d, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._b = torch.tensor(b) # equality RHS
        self._d = torch.tensor(d) # inequality RHS

        self._eq_cm = self._A
        self._ineq_cm = self._G
        self._eq_rhs = self._b
        self._ineq_rhs = self._d

        self._valid_frac = valid_frac
        self._test_frac = test_frac

        self._Y = None
        self._ydim = Q.shape[0]

        ### For Pytorch
        self._device = None

        #! Implement in child!
        self._X = None
        self._num = None
        self._neq = None
        self._nineq = None
        self._xdim = None

        self._mu = None
        self._lamb = None

    ##### ABSTRACT METHODS #####

    @abstractmethod
    def eq_resid(self, X, Y):
        raise NotImplementedError
    
    @abstractmethod
    def ineq_resid(self, X, Y):
        raise NotImplementedError
    
    @abstractmethod
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        raise NotImplementedError

    @abstractmethod
    def calc_Y(self):
        raise NotImplementedError


    def __str__(self):
        return "SimpleProblem-{}-{}-{}-{}".format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    ##### REG METHODS #####

    @property
    def eq_cm(self):
        return self._eq_cm

    @property
    def ineq_cm(self):
        return self._ineq_cm
    
    @property
    def eq_rhs(self):
        return self._eq_rhs
    
    @property
    def ineq_rhs(self):
        return self._ineq_rhs

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def b(self):
        return self._b

    @property
    def d(self):
        return self._d

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def b_np(self):
        return self.b.detach().cpu().numpy()

    @property
    def d_np(self):
        return self.d.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac
    
    @property
    def train_indices(self):
        return list(range(int(self.num * self.train_frac)))
    
    @property
    def valid_indices(self):
        return list(range(int(self.num * self.train_frac), int(self.num * (self.train_frac + self.valid_frac))))
    
    @property
    def test_indices(self):
        return list(range(int(self.num * (self.train_frac + self.valid_frac)), self.num))

    @property
    def trainX(self):
        return self.X[: int(self.num * self.train_frac)]

    @property
    def validX(self):
        return self.X[
            int(self.num * self.train_frac) : int(
                self.num * (self.train_frac + self.valid_frac)
            )
        ]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)) :]

    @property
    def trainY(self):
        return self.Y[: int(self.num * self.train_frac)]

    @property
    def validY(self):
        return self.Y[
            int(self.num * self.train_frac) : int(
                self.num * (self.train_frac + self.valid_frac)
            )
        ]

    @property
    def testY(self):
        return self.Y[int(self.num * (self.train_frac + self.valid_frac)) :]
    
    @property
    def mu(self):
        return self._mu
    
    @property
    def lamb(self):
        return self._lamb

    @property
    def train_mu(self):
        return self._mu[: int(self.num * self.train_frac)]

    @property
    def valid_mu(self):
        return self._mu[
            int(self.num * self.train_frac) : int(
                self.num * (self.train_frac + self.valid_frac)
            )
        ]

    @property
    def test_mu(self):
        return self.mu[int(self.num * (self.train_frac + self.valid_frac)) :]
    
    @property
    def train_lamb(self):
        return self.lamb[: int(self.num * self.train_frac)]

    @property
    def valid_lamb(self):
        return self.lamb[
            int(self.num * self.train_frac) : int(
                self.num * (self.train_frac + self.valid_frac)
            )
        ]
    
    @property
    def test_lamb(self):
        return self.lamb[int(self.num * (self.train_frac + self.valid_frac)) :]

    @property
    def device(self):
        return self._device

    def obj_fn(self, X, Y):
        return (0.5 * (Y @ self.Q) * Y +  self.p * Y).sum(dim=1)

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2 * (Y @ self.A.T - X) @ self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2 * ineq_dist @ self.G

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (
            self._A_other_inv @ self._A_partial
        )
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = (
            2
            * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0)
            @ G_effective
        )
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = -(grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

class OriginalQPProblem(SimpleProblem):
    def __init__(self, Q, p, A, G, b, d, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G, b, d, valid_frac, test_frac)

        self._X = self._b
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        self._nineq = self._G.shape[0]
        self._xdim = self._X.shape[1]

        self.A_transpose = self._A.T
        self.G_transpose = self._G.T

    def eq_resid(self, X, Y):
        # Here, X is the RHS of the equality constraints
        return X - Y @ self.A.T

    def ineq_resid(self, X, Y):
        return Y @ self.G.T - self.d
    
    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-6):
        if solver_type == "osqp":
            print("running osqp")
            Q, p, A, G, d = self.Q_np, self.p_np, self.A_np, self.G_np, self.d_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(d.shape[0]) * np.inf])
                my_u = np.hstack([Xi, d])
                solver.setup(
                    P=csc_matrix(Q),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

                sols = np.array(Y)
                parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

    # Dual functions!
    def dual_obj_fn(self, X, mu, lamb):
        """-1/2(p + A^T\lambda + G^T\mu)^T Q^-1(p + A^T\lambda + G^T\mu) - \lambda^T X[i] - \mu^Td

        Args:
            mu (_type_): _description_
            lamb (_type_): _description_
        """
        # lamb = Y[:, :self.neq]
        # mu = Y[:, self.neq:]
        # Compute (p + A^T \lambda + G^T \mu)
        quad_term = self.p.unsqueeze(0) + lamb @ self.A + mu @ self.G  # Shape: (b, n)

        # Solve Q^-1 * quad_term for each batch
        Q_inv_quad_term = torch.linalg.solve(self.Q, quad_term.T).T  # Shape: (b, n)

        # Compute quadratic term: -1/2 * sum((quad_term * Q_inv_quad_term) per row)
        quad_value = -0.5 * torch.einsum("bi,bi->b", quad_term, Q_inv_quad_term)  # Shape: (b,)

        # Compute linear terms: -λ^T X and -μ^T d
        lin_term_lambda = -torch.einsum("bi,bi->b", lamb, X)  # Shape: (b,)
        lin_term_mu = -torch.einsum("bi,i->b", mu, self.d)  # Shape: (b,)

        # Compute final dual objective value
        return quad_value + lin_term_lambda + lin_term_mu

    def dual_eq_resid(self, mu, lamb):
        return torch.tensor(0.0)

    def dual_ineq_resid(self, mu, lamb):
        return -mu
    
    def dual_ineq_dist(self, mu, lamb):
        resids = self.dual_ineq_resid(mu, lamb)
        return torch.clamp(resids, 0)

    def dual_opt_solve(self, X, solver_type="osqp", tol=1e-6):
        if solver_type == "osqp":
            print("running osqp")
            Q, p, A, G, d = self.Q_np, self.p_np, self.A_np, self.G_np, self.d_np
            X_np = X.detach().cpu().numpy()
            lamb = []
            mu = []
            total_time = 0

            # Compute Q^-1
            Q_inv_p = np.linalg.solve(Q, p)  # Solves Q @ x = p for x = Q^-1 p
            Q_inv_A = np.linalg.solve(Q, A.T)  # Solves Q @ x = A^T
            Q_inv_G = np.linalg.solve(Q, G.T)  # Solves Q @ x = G^T

            # Compute block matrix P for OSQP
            A_Qinv_A = A @ Q_inv_A  # (k, k)
            A_Qinv_G = A @ Q_inv_G  # (k, m)
            G_Qinv_A = G @ Q_inv_A  # (m, k)
            G_Qinv_G = G @ Q_inv_G  # (m, m)

            # Construct full P matrix
            P_upper = np.concatenate([A_Qinv_A, A_Qinv_G], axis=1)  # (k, k+m)
            P_lower = np.concatenate([G_Qinv_A, G_Qinv_G], axis=1)  # (m, k+m)
            P = np.concatenate([P_upper, P_lower], axis=0)  # (k+m, k+m)

            # Convert to sparse matrix for OSQP
            P_sparse = csc_matrix(P)

            # Compute q, incorporating Xi
            q_mu = (self.d + (self.G @ Q_inv_p))  # (m,)

            # Construct A: (n, n) matrix
            my_A = np.block([
                [np.zeros((self.neq, self.neq)), np.zeros((self.neq, self.nineq))],  # Top block: No constraints on lambda
                [np.zeros((self.nineq, self.neq)), np.eye(self.nineq)]         # Bottom block: -I for mu (to enforce mu ≥ 0)
            ])  # Shape: (n, n)
            # my_A = np.vstack([A, G])
            my_l = np.hstack([-np.ones(A.shape[0]) * np.inf, np.zeros(self.nineq)])
            my_u = np.hstack([np.ones(A.shape[0]) * np.inf, np.ones(self.nineq) * np.inf])
        
            for Xi in X.detach():
                solver = osqp.OSQP()
                q_lambda = (Xi + (A @ Q_inv_p))  # (k,)
                q = np.concatenate([q_lambda, q_mu], axis=0)  # (k+m,)

                solver.setup(
                    P=P_sparse,
                    q=q,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    lamb.append(results.x[:self.neq])
                    mu.append(results.x[self.neq:])
                else:
                    lamb.append(np.ones(self.neq) * np.nan)
                    mu.append(np.ones(self.nineq) * np.nan)

                sols_lamb = np.array(lamb)
                sols_mu = np.array(mu)
                parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError

        return sols_mu, sols_lamb, total_time, parallel_time

    def dual_calc_Y(self):
        sols_mu, sols_lamb, total_time, parallel_time = self.dual_opt_solve(self.X)
        self._mu = torch.tensor(sols_mu)
        self._lamb = torch.tensor(sols_lamb)
        return sols_mu, sols_lamb
    
class NonconvexQPProblem(SimpleProblem):
    def __init__(self, Q, p, A, G, b, d, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G, b, d, valid_frac, test_frac)

        self._X = self._b
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        self._nineq = self._G.shape[0]
        self._xdim = self._X.shape[1]

        self.A_transpose = self._A.T
        self.G_transpose = self._G.T

    def obj_fn(self, X, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1)

    def eq_resid(self, X, Y):
        # Here, X is the RHS of the equality constraints
        return X - Y @ self.A.T

    def ineq_resid(self, X, Y):
        return Y @ self.G.T - self.d
    
    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, solver_type="ipopt", tol=1e-4):
        Q, p, A, G, d = self.Q_np, self.p_np, self.A_np, self.G_np, self.d_np
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        for i, Xi in enumerate(X_np):
            print(f"Problem: {i}/{X_np.shape[0]}")
            if solver_type == "ipopt":
                y0 = np.linalg.pinv(A) @ Xi  # feasible initial point

                # upper and lower bounds on variables
                lb = -np.infty * np.ones(y0.shape)
                ub = np.infty * np.ones(y0.shape)

                # upper and lower bounds on constraints
                cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
                cu = np.hstack([Xi, d])

                nlp = ipopt.problem(
                    n=len(y0),
                    m=len(cl),
                    problem_obj=nonconvex_ipopt(Q, p, A, G),
                    lb=lb,
                    ub=ub,
                    cl=cl,
                    cu=cu,
                )

                nlp.addOption("tol", tol)
                nlp.addOption("print_level", 0)  # 3)

                start_time = time.time()
                y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                total_time += end_time - start_time
            else:
                raise NotImplementedError

        return np.array(Y), total_time, total_time / len(X_np)

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y


class nonconvex_ipopt(object):
    def __init__(self, Q, p, A, G):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p @ np.sin(y)

    def gradient(self, y):
        return self.Q @ y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.A @ y, self.G @ y])

    def jacobian(self, y):
        return np.concatenate([self.A.flatten(), self.G.flatten()])

class QPProblemVaryingG(SimpleProblem):
    def __init__(self, X, Q, p, A, G, b, d, row_indices, col_indices, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G, b, d, valid_frac, test_frac)
        # X are the varying values of G
        self._X = X
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        self._nineq = self._G.shape[0]
        self._xdim = self._X.shape[1]

        self.row_indices = row_indices
        self.col_indices = col_indices

    def eq_resid(self, X, Y):
        # Here, X is part of the inequality constraint matrix. So we don't use it
        return self.b - Y @ self.A.T

    def ineq_resid(self, X, Y):
        # Here, X is part of the inequality constraint matrix. So, we need to plug X into the inequality constraint matrix.
        G = self.G.expand(X.shape[0], -1, -1).clone()
        G[:, self.row_indices, self.col_indices] = X
        return torch.bmm(G, Y.unsqueeze(-1)).squeeze(-1) - self.d
    
    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        """We change op_solve so that the varying G matrices are taken from the input X.
        """
        if solver_type == "osqp":
            print("running osqp")
            Q, p, b, d = self.Q_np, self.p_np, self.b_np, self.d_np
            G = self.G.expand(X.shape[0], -1, -1).clone()
            G[:, self.row_indices, self.col_indices] = X
            G = G
            A = self.A_np
            Y = []
            total_time = 0

            for Gi in G:
                
                solver = osqp.OSQP()
                my_A = np.vstack([A, Gi])
                my_l = np.hstack([b, -np.ones(d.shape[0]) * np.inf])
                my_u = np.hstack([b, d])
                solver.setup(
                    P=csc_matrix(Q),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time / len(G)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

class QPProblemVaryingQ(SimpleProblem):
    def __init__(self, X, Q, p, A, G, b, d, indices, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G, b, d, valid_frac, test_frac)
        # X are the varying values of G
        self._X = X
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        self._nineq = self._G.shape[0]
        self._xdim = self._X.shape[1]

        # Q is diagonal matrix, so indices are the same for both row and column.
        self.row_indices = indices
        self.col_indices = indices

    def obj_fn(self, X, Y):
        Q = self.Q.expand(Y.shape[0], -1, -1).clone()
        Q[:, self.row_indices, self.col_indices] = X
        quadratic_term = torch.einsum('bi,bij,bj->b', Y, Q, Y)
        return 0.5 * quadratic_term + (self.p * Y).sum(dim=1)

    def eq_resid(self, X, Y):
        # Here, X is the RHS of the equality constraints
        return X - Y @ self.A.T

    def ineq_resid(self, X, Y):
        return Y @ self.G.T - self.d
    
    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        """We change op_solve so that the varying G matrices are taken from the input X.
        """
        if solver_type == "osqp":
            print("running osqp")
            p, b, d = self.p_np, self.b_np, self.d_np
            Q = self.Q.expand(X.shape[0], -1, -1).clone()
            Q[:, self.row_indices, self.col_indices] = X
            G = self.G_np
            A = self.A_np
            Y = []
            total_time = 0

            for Qi in Q:
                
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([b, -np.ones(d.shape[0]) * np.inf])
                my_u = np.hstack([b, d])
                solver.setup(
                    P=csc_matrix(Qi),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time / len(Q)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

class QPProblemVaryingGbd(SimpleProblem):
    def __init__(self, Q, p, A, G_base, G_varying, b, d, n_varying_rows, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G_varying, b, d, valid_frac, test_frac)
        self.G_base = torch.tensor(G_base)
        self.n_varying_rows = n_varying_rows
        # Flatten the rows of G that are varying, to be added to the NN input.
        G_flattened = self.G[:, :n_varying_rows, :].flatten(start_dim=1)
        self._X = torch.concat([G_flattened, self.b, self.d], dim=1)
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        # G now has num_samples in first dimension, num_constraints in second dimension. Take second dimension!
        self._nineq = self._G.shape[1]
        self._xdim = self._X.shape[1]

    def eq_resid(self, X, Y):
        """B is now varying, we should extract it from X"""
        G, b, d = self.rebuild_Gbd_from_X(X)
        return b - Y @ self.A.T

    def rebuild_Gbd_from_X(self, X):
        # Reshape X to match the first self.n_varying_rows rows of G
        G_size = self.n_varying_rows*self.ydim
        b_size = self.neq
        flattened_G = X[:, :G_size]
        b = X[:, G_size:G_size+b_size]
        d = X[:, G_size+b_size:]
        custom_G = flattened_G.reshape(X.shape[0], self.n_varying_rows, self._ydim)  # Reshape for the batch size

        # Take only the first sample of G and clone it for modification
        G = self.G_base.clone()  # Shape is (M, P)

        # Repeat G for the batch size to avoid memory overlap
        G = G.unsqueeze(0).repeat(X.shape[0], 1, 1)  # Shape is (batch_size, M, P)

        # Assign custom_G to the first self.n_varying_rows rows of G
        G[:, :self.n_varying_rows, :] = custom_G  # Ensure dimensions match
        return G, b, d
    
    def ineq_resid(self, X, Y):
        """
        For the ineq resid, we need to extract the first n rows of the G matrix from it's flattened form X, and plug them into G.
        """

        G, b, d = self.rebuild_Gbd_from_X(X)

        # resid = Y @ G.transpose(1, 2) - h
        residual = torch.bmm(Y.unsqueeze(1), G.transpose(1, 2)).squeeze(1) - d

        # Compute inequality residual
        return residual
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        """We change op_solve so that the varying G matrices are taken from the input X.
        """
        if solver_type == "osqp":
            print("running osqp")
            Q, p, b, d = self.Q_np, self.p_np, self.b_np, self.d_np
            G, b, d = self.rebuild_Gbd_from_X(X)
            G_np, b_np, d_np = G.detach().cpu().numpy(), b.detach().cpu().numpy(), d.detach().cpu().numpy()
            A = self.A_np
            Y = []
            total_time = 0
            for idx, Gi in enumerate(G_np):
                solver = osqp.OSQP()
                my_A = np.vstack([A, Gi])
                my_l = np.hstack([b_np[idx], -np.ones(d_np[idx].shape[0]) * np.inf])
                my_u = np.hstack([b_np[idx], d_np[idx]])
                solver.setup(
                    P=csc_matrix(Q),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time / len(X)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y