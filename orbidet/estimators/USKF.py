"""
Unscented SemiAnalytical Kalman Filter
"""
import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline

from beyond.beyond.orbits import Orbit
from .UKF import UT_matrix,UT_get_mean_cov


def observation_function(t,equinoctial,frame,h,mean_to_osc):
    """
    osc equinoctial.form = "cartesian"
    y = h (cartesian)
    """
    orb = Orbit(equinoctial,t,"equinoctial_mean",frame,None)
    orb = mean_to_osc(orb.copy())
    return h(t,orb)



class USKF():
    def __init__(self,x0,P0,f,Q,getEtasFromFourierCoefs,mean_to_osc):
        self.x = x0 #initial state                                   (numpy.array)
        self.P = P0 #initial covariance matrix                       (numpy.array)
        self.n = len(x0)
        self.Q = Q

        self.f = f

        # UT setup
        n = len(x0)
        k = 0 #scaling parameter - usually set to 0 or (3 - n)
        alpha = 1 #spread of sigma points parameter - typically 1e-4 <= alpha <= 1
        beta = 2 #optimal value for Gaussian distributions [ref: UKF theory papers]
        self.calculate_weights(n,k,alpha,beta)

        self.corrections = np.zeros(self.n)
        self.getEtasFromFourierCoefs = getEtasFromFourierCoefs
        self.mean_to_osc = mean_to_osc


    def calculate_weights(self,n,k,alpha,beta):
        """
        calculate weight matrices for the matrix form Unscented transform
        """
        self.lamb = alpha**2 * (n + k) - n
        w_i = [1 / (2*(n + self.lamb)) for _ in range(2*n)]
        self.W_m = np.array([self.lamb / (n + self.lamb)] + w_i)
        self.W_c = np.array([self.lamb / (n + self.lamb) + (1 - alpha**2 + beta)] + w_i)

        wm_matrix = np.full((2*n+1,2*n+1),self.W_m).T
        self.W = (np.eye(2*n+1) - wm_matrix) @ np.diag(self.W_c) @ (np.eye(2*n+1) - wm_matrix).T

    def setup_new_integration_grid(self):
        self.corrections = np.zeros(self.n)
        self.previous_step = {"P_nominal":self.P.copy(),
                              "P_updated":self.P.copy()}

    def integration_grid(self,t_in,t_out,method,frame,getFourierCoefs):
        """predict method of the UKF
        """
        self.setup_new_integration_grid()
        self._t = t_in

        step = (t_out - t_in)/2
        _t = [t_in,t_in+step,t_out,t_out+step]
        _tflt = [t.mjd*86400 for t in _t]

        x0 = self.x
        # integration from t_0 to t_1
        f = lambda x: self.f(x,_t[0],_t[1])
        x1,Py,Pxy,sigma_x0,sigma_x1 = UT_matrix(f,x0,self.P,self.lamb,self.W_m,self.W,True)
        self.previous_step["sigma"] = sigma_x0

        # integration from t_1 to t_2
        f = lambda x: self.f(x,_t[1],_t[2])
        x2,Py,Pxy,_,sigma_x2 = UT_matrix(f,x1,Py,self.lamb,self.W_m,self.W,True)

        # integration from t_2 to t_3
        f = lambda x: self.f(x,_t[2],_t[3])
        x3,Py,Pxy,_,sigma_x3 = UT_matrix(f,x2,Py,self.lamb,self.W_m,self.W,True)

        # setup interpolators:
        # 1.interpolator for state sigmas (Hermite interpolator)
        self.sigma_interpolator = CubicSpline(_tflt,[sigma_x0,sigma_x1,sigma_x2,sigma_x3],axis=0,extrapolate = False)



    def observation_grid(self,obs,h,R,t,frame,*args):
        """update step
        """
        # interpolate sigmas and reconstruct mean and covariance nominal trajectory
        sigmaX_kbef = self.previous_step["sigma"];
        P_kbef = self.previous_step["P_nominal"]
        sigma_Xknominal = self.sigma_interpolator(t.mjd*86400)
        x_k,Px_k_nominal,Px_k_kbef = UT_get_mean_cov(sigmaX_kbef,sigma_Xknominal,self.W_m,self.W)
        A = Px_k_kbef.T @ np.linalg.inv(P_kbef)
        # P_error_lnrz = Px_k - A @ P_kbef @ A.T

        # get predicted state from statistical linearization (SL) of nominal trajectory
        # and filter corrections propagated from the step before
        P_kbef_updated = self.previous_step["P_updated"]
        self.corrections =  A @ self.corrections

        # discretization of Q
        Qd = A @ self.Q @ A.T * ((t - self._t).total_seconds())
        Px_k = A @ P_kbef_updated @ A.T + Qd #+ P_error_lnrz


        # update step: mean equinoctial -> osc. equinoctial -> osc. cartesian -> observations
        if obs is not None:
            obs_h = lambda x : observation_function(t,x,frame,h,self.mean_to_osc)
            y,Py,Pxy = UT_matrix(obs_h,x_k+self.corrections,Px_k,self.lamb,self.W_m,self.W,False)
            Py += R

            Sinv = np.linalg.inv(Py) #4x4
            v = (obs - y)

            K = Pxy @ Sinv #6x4
            # x_update = x_kpred + K @ v
            Px_k = Px_k - K @ Py @ K.T
            self.corrections += K @ v

        self.mean_state = {"t":t,"x":x_k+self.corrections,"frame":frame}
        self.osc_state = {"t":t,"x":x_k+self.corrections,"frame":frame}
        self.nominal_mean_state = {"t":t,"x":x_k,"frame":frame}
        self.x = x_k+self.corrections
        self.P = Px_k

        # atualizar dicionario previus_step
        self.previous_step["P_nominal"] = Px_k_nominal.copy()
        self.previous_step["P_updated"] = Px_k.copy()
        self.previous_step["sigma"] = sigma_Xknominal.copy()
        self._t = t


    @property
    def nominal_mean_state(self):
        return self._nominal_mean_state
    @nominal_mean_state.setter
    def nominal_mean_state(self,info):
        self._nominal_mean_state = Orbit(info["x"],info["t"],"equinoctial_mean",info["frame"],None)

    @property
    def mean_state(self):
        return self._mean_state
    @mean_state.setter
    def mean_state(self,info):
        self._mean_state = Orbit(info["x"],info["t"],"equinoctial_mean",info["frame"],None)

    @property
    def osc_state(self):
        return self._osc_state
    @osc_state.setter
    def osc_state(self,info):
        orb = Orbit(info["x"],info["t"],"equinoctial_mean",info["frame"],None)
        self._osc_state = self.mean_to_osc(orb)

    @property
    def P_mean(self):
        return self.P
