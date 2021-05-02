"""implement a wrapper for the EKF and UKF
"""
import numpy as np
from scipy.integrate import solve_ivp

from beyond.beyond.orbits import Orbit
from beyond.beyond.dates import Date,timedelta

from .EKF import EKF
from .UKF import UKF
from .ESKF import ESKF
from .USKF import USKF
from .utils import *
from orbidet.propagators.utils import cartesian_osculating_state_fct,equinoctial_mean_state_fct
from orbidet.propagators.mean_osc_maps import SemianalyticalMeanOscMap


class CowellFilter():
    def __init__(self,filter,orbit,t,P0,Q, observer,force,ECI_frame,solver,
                 maximum_dt = timedelta(seconds=30),**kwargs):
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
        self.maximum_dt = maximum_dt
        self.ECI_frame = ECI_frame

        self.observer = observer
        self.Q = Q
        self.solver = solver
        self.date = Date(t)


        if filter is "EKF":
            f = lambda t,x : first_order_state_cov_differential_equation_cartesian(t,x,force)
            self.filter = EKF(np.array(orbit),P0,f)

        elif filter is "UKF":
            f = lambda t,x : cartesian_osculating_state_fct(t,x,force)
            f_discrete = lambda x0,t0,t1: (solve_ivp(f,(t0,t1),np.array(x0),method=solver,
                                                    t_eval=[t1])).y.flatten()
            self.filter = UKF(np.array(orbit),P0,f_discrete)
        else:
            raise Exception("Unknown filter {}".format(filter))


    def filtering_cycle(self,date,y,observer):
        """self.date -> t0
            date -> t1
        """
        # predict step
        while(self.date < date):
            step = date - self.date if (date - self.date <= self.maximum_dt) else self.maximum_dt
            self.filter.predict(self.date.mjd * 86400,(self.date+step).mjd*86400, self.Q,method=self.solver)
            self.date += step

        # update step
        if y is not None:
            h = lambda x: observer.h(x,date,frame=self.ECI_frame)
            H = lambda x,t: observer.jacobian_h(x,date,frame=self.ECI_frame)
            invS,v = self.filter.update(y,date,h,observer.R_default,H)
            return invS,v

        return (None,None)

    @property
    def x_osc(self):
        return self.filter.x

    @property
    def P_osc(self):
        return self.filter.P









class SemianalyticalFilter():
    def __init__(self,filter,orbit,P0,date,Q, observer,force,ECI_frame,solver,integration_stepsize,
                 quadrature_order = 30,DFT_lmb_len = 64, DFT_sideral_len=32):
        """
        args:
            *filter (str) - ESKF, USKF
            *orbit (Orbit) - initial orbit
            *P0 (numpy.ndarray) - initial cov matrix
            *Q(``) - default process noise
            *observers (list of SatelliteObserver or GroundStation)
            *force (ForceModel) force model to use in the state fct
            ECI_frame (str) - frame to perform the integration (should be orbit.frame)
            integration_stepsize (timedelta)
            quadrature order, DFT lens  (ints)
        """
        self.observer = observer
        self.Q = Q
        self.ECI_frame = ECI_frame
        self.solver = solver
        self.short_period_tf = SemianalyticalMeanOscMap(force,DFT_LEN=DFT_lmb_len,SIDEREAL_LEN=DFT_sideral_len)

        # creating state function lambdas
        state_fct = lambda t,x : equinoctial_mean_state_fct(t,x,force,quadrature_order)
        jacobian_state_fct = lambda t,x : finite_differencing(state_fct,x,t,np.array([0.001,1e-7,1e-07,1e-06,1e-06,1e-06]))

        f_etas = lambda t,x : self.short_period_tf.getEtasFromFourierCoefs(
            self.short_period_tf.getFourierCoefs(x,False),x,False)
        self.B1 = lambda t,x : finite_differencing(f_etas,Orbit(x,t,"equinoctial_mean",ECI_frame,None),t,
                        np.array([1e-5,1e-10,1e-10,1e-10,1e-10,1e-10]))

        # observation functions from 'observer' are defined with respect to cartesian osculating state,
        # hence, a transformation to mean equinoctial is needed
        self.h = lambda t,x: observer.h(Orbit(x,t,"equinoctial_mean",ECI_frame,None).copy(form="cartesian"))
        H = lambda x: observer.jacobian_h(x,date,frame=ECI_frame)
        self.jacobian_h = lambda t,x : obs_jacobian_wrt_equinoctial(t,x,H,ECI_frame)


        # creating initial conditions (mean and covariance transformation from osc cartesian to mean equinoctial)
        if str(orbit.form) is "cartesian":
            orbit = self.short_period_tf.osc_to_mean(orbit.copy())
            P0 = CartesianCov_to_MeanEq(P0,partials_cartesian_wrt_equinocital(orbit.copy()))
        elif str(orbit.form) is not "equinoctial_mean":
            raise Exception("Unknown initial state format")


        # creating filter instance
        if filter is "ESKF":
            self.filter = ESKF(orbit,P0,state_fct,jacobian_state_fct,Q,self.short_period_tf.mean_to_osc,
                               self.short_period_tf.getEtasFromFourierCoefs)
        elif filter is "USKF":
            state_fct_discrete = lambda x0,t0,t1: (solve_ivp(state_fct,(t0.mjd*86400,t1.mjd*86400),np.array(x0),method=solver,
                                                    t_eval=[t1.mjd*86400])).y.flatten()
            self.filter = USKF(orbit,P0,state_fct_discrete,Q,
                               self.short_period_tf.getEtasFromFourierCoefs,
                               self.short_period_tf.mean_to_osc)

        self._currentgrid_date = Date(date) #t0
        self.integration_stepsize = integration_stepsize
        # Do an integration grid step
        self.filter.integration_grid(date,date+integration_stepsize,solver,self.ECI_frame,
                                     self.short_period_tf.getFourierCoefs)

    def filtering_cycle(self,date,y,observer):
        Sinv,residuals = None,None
        if (date == self._currentgrid_date+self.integration_stepsize):
            # print("last observation of grid ",date)
            self.filter.observation_grid(y,self.h,observer.R_default,date,self.ECI_frame,self.jacobian_h,self.B1)
            # print("Defining new grid ",date,date+self.integration_stepsize)
            self.filter.integration_grid(date,date+self.integration_stepsize,
                                         self.solver,self.ECI_frame,
                                         self.short_period_tf.getFourierCoefs)
            self._currentgrid_date += self.integration_stepsize
        elif (date > self._currentgrid_date+self.integration_stepsize):
            # print("predicting until grid end",self._currentgrid_date+self.integration_stepsize)
            self.filter.observation_grid(None,self.h,observer.R_default,
                                         self._currentgrid_date+self.integration_stepsize,
                                         self.ECI_frame,self.jacobian_h,self.B1)
            self.filter.integration_grid(self._currentgrid_date+self.integration_stepsize,
                                         self._currentgrid_date+2*self.integration_stepsize,
                                         self.solver,self.ECI_frame,
                                         self.short_period_tf.getFourierCoefs)
            self.filter.observation_grid(y,self.h,observer.R_default,date,self.ECI_frame,self.jacobian_h,self.B1)
            self._currentgrid_date += self.integration_stepsize


        else:
            # print("Inside observation grid",date)
            self.filter.observation_grid(y,self.h,observer.R_default,date,self.ECI_frame,self.jacobian_h,self.B1)

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
