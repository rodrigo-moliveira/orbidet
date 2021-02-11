"""utility functions for the estimators
"""
import numpy as np

def MeanEqCov_to_cartesian(P_eq,dx_deq_jacbn,B1=None):
    """convert a covariance matrix in mean equinoctial state to osculating cartesian state
    P_x = G @ P_eq @ G.T
    where:
        G = dx/d_eq_osc @ d_eq_osc/d_eq_mean = dx/d_eq_osc ( I + B1 )
        If B1 is None, then a map between osculating states is performed instead
    """
    n = len(P_eq[0])
    if B1 is None:
        B1 = np.zeros((n,n))
    G = dx_deq_jacbn @ (np.eye(n) + B1)
    Px = G @ P_eq @ G.T
    return Px

def CartesianCov_to_MeanEq(Px,dx_deq_jacbn,B1=None):
    """invertion of MeanEqCov_to_cartesian function, provided that G is invertible
    """
    n = len(Px[0])
    if B1 is None:
        B1 = np.zeros((n,n))
    G = np.linalg.inv(dx_deq_jacbn @ (np.eye(n) + B1))
    Peq = G @ Px @ G.T
    return Peq














































# class MeanFilterold():
#     def __init__(self,filter,orbit,P0,date,Q_default, observers,force,ECI_frame,
#                  quadrature_order = 30,DFT_lmb_len = 64, DFT_sideral_len=32,**kwargs):
#         """
#         args:
#             *filter (str) - EKF, CD-UKF or DD-UKF
#             *orbit (Orbit) - initial orbit
#             *P0 (numpy.ndarray) - initial cov matrix
#             *Q_default(``) - default process noise
#             *observers (list of SatelliteObserver or GroundStation)
#             *force (ForceModel) force model to use in the state fct
#
#         """
#         self.observers = observers
#         self.Q_default = Q_default
#         self.ECI_frame = ECI_frame
#
#         if filter != "DD-UKF":
#             raise Exception("Currently, the only filter available for the MeanFilter is the 'DD-UKF'")
#
#         if force.DRAG:
#             _types = ("zonals","drag")
#         else:
#             _types = ("zonals",)
#
#         self.short_period_tf = NumAv_MEP_transf(force,DFT_LEN=DFT_lmb_len,SIDEREAL_LEN=DFT_sideral_len)
#
#         f = lambda t,x : force.equinoctial_state_fct(t,x,force,_types,quadrature_order)
#         if force.CYTHON:
#             f_discrete = lambda x0,t0,t1: discrete_cy(x0,t0,t1,kwargs["dt"],f)
#         else:
#             f_discrete = lambda x0,t0,t1: discrete_function(x0,t0,t1,kwargs["dt"],f)
#
#
#         # creating initial conditions (mean)
#         _orbit = Orbit(date, orbit,"cartesian",self.ECI_frame,None)
#         mean_orb = self.short_period_tf.osc_to_mean(_orbit.copy())
#
#         # creating filter instance
#         self.filter = UKF(np.array(mean_orb),P0,f_discrete)
#
#
#     def filtering_cycle(self,orbit,date_before,ys):
#         # predict step
#         self.filter.predict(date_before.mjd * 86400,orbit.date.mjd*86400, self.Q_default,method="RK45")
#
#         # update step
#         for (observer,y) in zip(self.observers,ys):
#             if y is not None:
#                 obs_h = lambda x: observer.h(x,orbit.date,delete=True)
#                 mean_to_osc = lambda x: self.short_period_tf.mean_to_osc(
#                     Orbit(orbit.date,x,"equinoctial",self.ECI_frame,None)
#                 ).copy(form="cartesian")
#
#                 h = lambda x: obs_h(mean_to_osc(x))
#                 invS,v = self.filter.update(y,orbit.date,h,observer.R_default)
#
#                 # set states
#                 self.x_osc = orbit.date
#                 return invS,v
#
#         return (None,None)
#
#     @property
#     def x_mean(self):
#         # returns equinoctial mean element
#         return self.filter.x
#     @property
#     def P_mean(self):
#         return self.filter.P
#
#     @property
#     def x_osc(self):
#         # returns osculating cartesian elements
#         return self._x_osc
#     @x_osc.setter
#     def x_osc(self,date):
#         orb_mean = Orbit(date,self.filter.x,"equinoctial",self.ECI_frame,None)
#         self._x_osc = self.short_period_tf.mean_to_osc(orb_mean).copy(form="cartesian")
