"""
This file contains a class to build an extended kalman filter
"""
import numpy as np
from scipy import integrate


class EKF():
    def __init__(self,x0,P0,EKF_LS_diff_eq):
        self.x = x0 #initial state                                   (numpy.array)
        self.P = P0 #initial covariance matrix                       (numpy.array)
        self.n = len(x0)

        self.predict_diff_eq = EKF_LS_diff_eq

    def predict(self,t_in,t_out,Q,method):
        """predict method of the EKF
        """
        if t_in == t_out:
            return
        #initial conditions for the solver
        phi0 = np.eye(6)
        Y0 = np.append(self.x,phi0)

        solver = integrate.solve_ivp(self.predict_diff_eq, (t_in,t_out), Y0, method=method, t_eval=[t_out])
        solver.y = solver.y.flatten()

        Y_matrix = np.reshape(solver.y,(7,6))
        self.x = Y_matrix[0]
        phi = Y_matrix[1:]

        # discretization of Q
        Qd = phi @ Q @ phi.T * (t_out - t_in)
        self.P = phi @ self.P @ phi.T + Qd

    def update(self,obs,t,h,R,grad):
        """update step
        """
        y = h(self.x)
        #calculate the observation jacobian
        H = grad(self.x,t)
        #kalman gain
        invS = np.linalg.inv(H @ self.P @ H.T + R)
        v = (obs - y)
        K = self.P @ H.T @ invS
        self.P = self.P - K @ H @ self.P
        self.x = (self.x + K @ v)
        return invS,v
