"""implement a wrapper for the EKF and UKF
"""
import numpy as np
from scipy.integrate import solve_ivp

from beyond.orbits import Orbit
from beyond.dates import Date,timedelta

from .EKF import EKF
from .UKF import UKF, CD_UKF
from .ESKF import ESKF
from .USKF import USKF
from ..utils.diff_eqs import osculating_state_fct,discrete_function,finite_differencing,equinoctial_state_jacobian_mean
from ..utils.diff_eqs import partials_cartesian_wrt_equinocital
from orbidet.cython_modules.filter_eqs import discrete_function as discrete_cy
from orbidet.propagators.short_period_transf import NumAv_MEP_transf

from orbidet.estimators.utils import CartesianCov_to_MeanEq,MeanEqCov_to_cartesian

class OsculatingFilter():
    def __init__(self,filter,orbit,t,P0,Q, observers,force,ECI_frame,solver,
                 use_cython=False,maximum_dt = 30,**kwargs):
        """
        args:
            *filter (str) - EKF, CD-UKF or DD-UKF
            *orbit (Orbit) - initial orbit
            *P0 (numpy.ndarray) - initial cov matrix
            *Q(``) - default process noise
            *observers (list of SatelliteObserver or GroundStation)
            *solver - integration solver (RK45 or DOP853)
            *force (ForceModel) force model to use in the state fct
            *maximum_dt - maximum integration step size allowed [seconds]

        """
        #state function
        state_fct = force.osculating_fct
        self.maximum_dt = maximum_dt
        self.ECI_frame = ECI_frame

        self.observers = observers
        if Q["type"] is not "continuous" and Q["type"] is not "discrete":
            raise Exception("Q type should be 'continuous' or 'discrete'")
        self.Q = Q
        self.solver = solver
        self.date = Date(t)

        filter_info = kwargs.get("info",None)
        if filter is "EKF":
            self.filter = EKF(np.array(orbit),P0,force.EKF_LS_diff_eq)
        # elif filter is "RAEKF":
        #     self.filter = EKF(np.array(orbit),P0,force.EKF_LS_diff_eq)
        elif filter is "UKF":
            f_discrete = lambda x0,t0,t1: (solve_ivp(state_fct,(t0,t1),np.array(x0),method=solver,
                                                    t_eval=[t1])).y.flatten()
            self.filter = UKF(np.array(orbit),P0,f_discrete)
        elif filter is "RAUKF":
            f_discrete = lambda x0,t0,t1: (solve_ivp(state_fct,(t0,t1),np.array(x0),method=solver,
                                                    t_eval=[t1])).y.flatten()
            self.filter = UKF(np.array(orbit),P0,f_discrete,robust_variation=True,parameters=filter_info["robust"])
        else:
            raise Exception("Unknown filter {}".format(filter))


    def filtering_cycle(self,date,ys):
        """self.date -> t0
            date -> t1
        """
        # predict step
        while(self.date < date):
            step = date - self.date if (date - self.date <= self.maximum_dt) else self.maximum_dt
            # print("iterating from ",self.date, "to ",self.date+step)
            self.filter.predict(self.date.mjd * 86400,(self.date+step).mjd*86400, self.Q,method=self.solver)
            self.date += step

        # update step
        for (observer,y) in zip(self.observers,ys):
            if y is not None:
                h = lambda x: observer.h(x,date,delete=True,frame=self.ECI_frame)
                H = lambda x,t: observer.grad_h(x,date,frame=self.ECI_frame)
                invS,v = self.filter.update(y,date,h,observer.R_default,H)
                return invS,v

        return (None,None)

    @property
    def x(self):
        return self.filter.x

    @property
    def P(self):
        return self.filter.P

def obs_jacobian_wrt_equinoctial(t,x,H,frame):
    orb = Orbit(t,x,"equinoctial",frame,None)
    Dosc_Dequict = partials_cartesian_wrt_equinocital(orb)
    orb.form = "cartesian"
    grad_h = H(orb)
    return grad_h @ Dosc_Dequict

def create_lambda_h(observer,frame):
    return lambda t,x: observer.h(Orbit(t,x,"equinoctial",frame,None).copy(
        form="cartesian"),delete=True)
def create_lambda_H(observer,frame,date):
    H = lambda x: observer.grad_h(x,date,frame=frame)
    return lambda t,x : obs_jacobian_wrt_equinoctial(t,x,H,frame)

class MeanFilter():
    def __init__(self,filter,orbit,P0,initial_state_str,date,Q, observers,force,force_simplified,ECI_frame,solver,integration_stepsize,
                 quadrature_order = 30,DFT_lmb_len = 64, DFT_sideral_len=32):
        """
        args:
            *filter (str) - ESKF, USKF
            *orbit (Orbit) - initial orbit
            *P0 (numpy.ndarray) - initial cov matrix
            *initial_state_str - "cartesian" or "equinoctial". If it is in "cartesian", then a transformation of
                    (mean,cov) to equinoctial form is performed
            *Q(``) - default process noise
            *observers (list of SatelliteObserver or GroundStation)
            *force (ForceModel) force model to use in the state fct
            ECI_frame (str) - frame to perform the integration (should be orbit.frame)
            integration_stepsize (timedelta)
            quadrature order, DFT lens  (ints)
        """
        self.observers = observers
        if Q["type"] is not "continuous" and Q["type"] is not "discrete":
            raise Exception("Q type should be 'continuous' or 'discrete'")
        self.Q = Q
        self.ECI_frame = ECI_frame
        self.solver = solver

        if force.DRAG:
            _types = ("zonals","drag")
        else:
            _types = ("zonals",)

        self.short_period_tf = NumAv_MEP_transf(force,DFT_LEN=DFT_lmb_len,SIDEREAL_LEN=DFT_sideral_len)
        short_period_tf_simplified = NumAv_MEP_transf(force_simplified,DFT_LEN=DFT_lmb_len,SIDEREAL_LEN=DFT_sideral_len)
        self.getFourierCoefs = short_period_tf_simplified.getFourierCoefs

        # state functions
        f = lambda t,x : force.equinoctial_state_fct(t,x,force,_types,quadrature_order)
        epsons = np.array([0.001,1e-7,1e-07,1e-06,1e-06,1e-06])
        graf_f = lambda t,x : finite_differencing(f,x,t,epsons)
        f_etas = lambda t,x : short_period_tf_simplified.getEtasFromFourierCoefs(
            short_period_tf_simplified.getFourierCoefs(x,False),x,False)
        epsons = np.array([1e-5,1e-10,1e-10,1e-10,1e-10,1e-10])
        self.B1 = lambda t,x : finite_differencing(f_etas,Orbit(t,x,"equinoctial",ECI_frame,None),t,epsons)
        self.hs = []
        self.Hs = []
        self.Rs = []
        for observer in self.observers:
            # h = lambda t,x: observer.h(Orbit(t,x,"equinoctial",self.ECI_frame,None).copy(
            #     form="cartesian"),delete=True)
            h = create_lambda_h(observer,self.ECI_frame)
            H = create_lambda_H(observer,self.ECI_frame,date)
            # H = lambda x: observer.grad_h(x,date,frame=self.ECI_frame)
            self.hs.append(h)
            # self.Hs.append(lambda t,x : obs_jacobian_wrt_equinoctial(t,x,H,ECI_frame))
            self.Hs.append(H)
            self.Rs.append(observer.R_default)
        f_discrete = lambda x0,t0,t1: (solve_ivp(f,(t0.mjd*86400,t1.mjd*86400),np.array(x0),method=solver,
                                                t_eval=[t1.mjd*86400])).y.flatten()
        # f_discrete = lambda x0,t0,t1: discrete_cy(x0,t0.mjd*86400,t1.mjd*86400,(t1-t0).total_seconds(),f)

        # creating initial conditions (mean and covariance transformation from osc cartesian to mean equinoctial)
        if initial_state_str is "cartesian":
            _orbit = Orbit(date, orbit,"cartesian",self.ECI_frame,None)
            orbit = short_period_tf_simplified.osc_to_mean(_orbit)
            P0 = CartesianCov_to_MeanEq(P0,partials_cartesian_wrt_equinocital(orbit.copy()))
        elif initial_state_str is not "equinoctial":
            raise Exception("Unknown initial state format")


        # creating filter instance
        if filter is "ESKF" or filter is "EKF":
            self.filter = ESKF(orbit,P0,f,graf_f,Q,self.short_period_tf.mean_to_osc,
                               short_period_tf_simplified.getEtasFromFourierCoefs)
        elif filter is "UKF" or filter is "USKF":
            self.filter = USKF(orbit,P0,f_discrete,Q,
                               short_period_tf_simplified.getEtasFromFourierCoefs,
                               short_period_tf_simplified.mean_to_osc)
        self._currentgrid_date = Date(date) #t0
        self.integration_stepsize = integration_stepsize

        # Do an integration grid step
        self.filter.integration_grid(date,date+integration_stepsize,solver,self.ECI_frame,
                                     self.getFourierCoefs)

    def filtering_cycle(self,date,ys):
        Sinv,residuals = None,None
        if (date == self._currentgrid_date+self.integration_stepsize):
            # print("last observation of grid ",date)
            self.filter.observation_grid(ys,self.hs,self.Rs,date,self.ECI_frame,self.Hs,self.B1)
            # print("Defining new grid ",date,date+self.integration_stepsize)
            self.filter.integration_grid(date,date+self.integration_stepsize,
                                         self.solver,self.ECI_frame,
                                         self.getFourierCoefs)
            self._currentgrid_date += self.integration_stepsize
        elif (date > self._currentgrid_date+self.integration_stepsize):
            # print("predicting until grid end",self._currentgrid_date+self.integration_stepsize)
            self.filter.observation_grid([None],self.hs,self.Rs,
                                         self._currentgrid_date+self.integration_stepsize,
                                         self.ECI_frame,self.Hs,self.B1)
            self.filter.integration_grid(self._currentgrid_date+self.integration_stepsize,
                                         self._currentgrid_date+2*self.integration_stepsize,
                                         self.solver,self.ECI_frame,
                                         self.getFourierCoefs)
            self.filter.observation_grid(ys,self.hs,self.Rs,date,self.ECI_frame,self.Hs,self.B1)
            self._currentgrid_date += self.integration_stepsize


        else:
            # print("Inside observation grid",date)
            self.filter.observation_grid(ys,self.hs,self.Rs,date,self.ECI_frame,self.Hs,self.B1)

        return (Sinv,residuals)

    @property
    def x_mean(self):
        return self.filter.mean_state
    @property
    def x_mean_nominal(self):
        return self.filter.nominal_mean_state
    @property
    def x_osc(self):
        return self.filter.osc_state

    @property
    def P_mean(self):
        return self.filter.P_mean
    @property
    def P_osc(self):
        P_mean = self.filter.P_mean
        x_mean = self.filter.mean_state
        return MeanEqCov_to_cartesian(P_mean,partials_cartesian_wrt_equinocital(x_mean.copy()))
