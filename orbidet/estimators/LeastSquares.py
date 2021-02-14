"""
This file implements the least squares estimator to the orbit determination problem with the initial observations
The objective is to create an initial estimation to then feed the kalman
"""
import numpy as np
from scipy import integrate
import scipy

from orbidet.errors import LSnotConverged
from .utils import first_order_state_cov_differential_equation_cartesian as diff_eq

class LeastSquares():
    """
    class that has the solver of the Least Squares initial estimation
    INPUTS:
        DF_obs - dataframe with observations to use
        X0 - initial state (orbit)
        Rmatrix - noise covariance matrix
        RMS_threshold - threshold to stop iteration convergence
        jacobian_h and h - jacobian and measurement functions
        force : force model
    """
    METHOD = "RK45"

    def __init__(self,LS_obs,orbit0,Rmatrix,RMS_threshold,jacobian_h,h,force,method=METHOD,
                 tol = 1e-7):
        #initializations for the solver
        self.Rinv = np.linalg.inv(Rmatrix) #only invert once
        self.orbit0 = orbit0
        self.method=method
        self.tol = tol

        #get the time vector with all observation times
        self.t = [i[0].mjd*86400 for i in LS_obs]

        self.jacobian_h = jacobian_h
        self.h = h
        self.i = 1 #iteration
        self.frameECI = orbit0.frame

        self.N_sensors = len(LS_obs[0][1])
        self.final_epoch = LS_obs[-1][0]
        self.force = force

        RMS = RMS_threshold + 1

        #iterate LS solver until convergence
        while (self.i <= 25 and RMS >= RMS_threshold):

            #cleanup for the next iteration
            self.Lambda = np.zeros((6,6))
            self.N = np.zeros(6)
            self.residuals = []  #save residuals of current iteration to calculate RMS

            X = self.iterateLS(LS_obs,self.t)
            RMS = self.metric_RMS(len(self.t))
            print("iterating, RMS: ",RMS)
            self.i += 1
            self.orbit0 = X
        if RMS > RMS_threshold * 10:
            raise LSnotConverged()


    def iterateLS(self,DF_obs,t):
        #1 - Integrate the state function and state transition matrix
        phi0 = np.eye(6)
        Y0 = np.append(self.orbit0,phi0)

        results = integrate.solve_ivp(diff_eq, (t[0],t[-1]), Y0, method=self.method, t_eval=t,
                                      rtol = self.tol, atol = self.tol/1e3,
                                      args = (self.force,))

        #2 - Read all observations and accumulate results
        i = 0
        for date, obs in DF_obs:
            #for t_i unpack the STM phi(ti,t0) and state X(ti)
            col_i = results.y[:,i]
            Y_matrix = np.reshape(col_i,(7,6))
            x_i = Y_matrix[0]               #state X(t=ti)
            phi_i = Y_matrix[1:] #phi(ti,t0)

            #LS residuals
            nominal_obs =  self.h(x_i,date=date,frame=self.frameECI)
            residual = obs - nominal_obs

            #Jacobian of observations
            Jacobian_H = self.jacobian_h(x_i,date=date,frame=self.frameECI)

            # print(obs_i,"\n",nominal_obs,"\n",residual,"\n\n")
            self.residuals.append(residual)
            Hi = Jacobian_H @ phi_i

            #Accumulation
            self.Lambda += Hi.T @ self.Rinv @ Hi
            self.N += (Hi.T @ self.Rinv) @ residual

            i += 1
            # print(self.Lambda,"\n\n")
            if (date.mjd*86400 == t[-1]):
                self.phi_final = phi_i

        #3 - solve the LS equation
        # x = np.linalg.inv(self.Lambda) @ self.N
        x_ = scipy.linalg.solve(self.Lambda,self.N)

        # print("correction: ",x)
        return self.orbit0 + x_

    def metric_RMS(self,N_obs):
        """
        N_sensors = number of sensors used in each observation at time t_i
        N_obs = number of observation time instants
            Number of data points = N_sensors * N_obs
        """
        b = np.zeros(len(self.residuals[0]))
        for residual in self.residuals:
            b += residual
        aux = (b.T @ self.Rinv @ b)
        RMS = np.sqrt(aux /(self.N_sensors * N_obs))
        return RMS

    """
    Definition of getters:
        1)final converged solution of X0 and cov P0
        2)Propagated solution in time. final time (TF) has to be defined when initializing
                the object
    """
    def get_orbit0(self):
        return self.orbit0

    def get_P0(self):
        return np.linalg.inv(self.Lambda)

    def get_propagated_solution(self):
        # We cannot simply multiply X0 with the transition matrix (X(tf) = phi(tf,t0) * X(t0))
        # I'll solve the numerical dif. eq.
        t0 = self.t[0]
        tf = self.t[-1]

        from orbidet.propagators.utils import cartesian_osculating_state_fct as f
        results = integrate.solve_ivp(f,(t0,tf), self.get_orbit0(), method='RK45', t_eval=[tf],args = (self.force,))

        x = results.y.flatten()
        P = self.phi_final @ self.get_P0() @ self.phi_final.T
        return x,P,self.final_epoch
