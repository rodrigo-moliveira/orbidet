"""
Extended SemiAnalytical Kalman Filter
"""
import numpy as np
from scipy import integrate
from scipy.interpolate import CubicHermiteSpline,CubicSpline

from beyond.beyond.orbits import Orbit


# used both in EKF and LS
def build_dif_eq(state_fun,jacobian_state):
    def predic_dif_eq(t,Y):
        """
        wrapper that contains the state and phi diff eqs.
        These two functions are packed in a system to feed the ODE solver
        This system is used for the EKF and LS
        """
        #unpack the variables
        Y_matrix = np.reshape(Y,(7,6))
        y = Y_matrix[0]
        phi = Y_matrix[1:]

        #apply differential functions
        dydt = state_fun(t,y) #derivative of position and velocity
        F = jacobian_state(t,y)
        dphidt = F @ phi

        #pack the variables
        return np.append(dydt,dphidt)
    return predic_dif_eq




class ESKF():
    def __init__(self,x0,P0,f,graf_f,Q,mean_to_osc,getEtasFromFourierCoefs):
        self.x = x0 #initial state                                   (numpy.array)
        self.P = P0 #initial covariance matrix                       (numpy.array)
        self.n = len(x0)
        self.Q = Q

        self.predict_diff_eq = build_dif_eq(f,graf_f)
        self.f = f

        self.corrections = np.zeros(self.n)
        self.mean_to_osc = mean_to_osc
        self.getEtasFromFourierCoefs = getEtasFromFourierCoefs


    def setup_new_integration_grid(self):
        self.corrections = np.zeros(self.n)
        self.phi_S = np.eye(self.n)

    def integration_grid(self,t_in,t_out,method,frame,getFourierCoefs):
        """predict method of the EKF
        """
        # print(" Integration grid", t_in,t_out)
        self.setup_new_integration_grid()
        self._t = t_in

        step = (t_out - t_in)/2
        _t = [t_in,t_in+step,t_out,t_out+step]
        _tflt = [t.mjd*86400 for t in _t]

        #initial conditions for the solver
        phi0 = np.eye(6)
        x0 = np.array(self.x)
        Y0 = np.append(x0,phi0)

        solver = integrate.solve_ivp(self.predict_diff_eq, (_tflt[0],_tflt[2]),
                                     Y0, method=method, t_eval=[_tflt[1],_tflt[2]])

        # get solutions at t_half and t_1
        _data = np.reshape(solver.y[:,0],(7,6))
        x_half = _data[0]
        phi_half = _data[1:]
        phi_half_inv = np.linalg.inv(phi_half)
        _data = np.reshape(solver.y[:,1],(7,6))
        x = _data[0]
        phi = _data[1:]
        phi_inv = np.linalg.inv(phi)


        # Evaluations at the knot points
        # t0
        _data = self.predict_diff_eq(_t[0],Y0)
        _data = np.reshape(_data,(7,6))
        x0_dot = _data[0]
        phi0_dot = _data[1:]
        phi0_dot_inv = np.array(-phi0_dot)

        # t_half
        Y = np.append(x_half,phi_half)
        _data = self.predict_diff_eq(_t[1],Y)
        _data = np.reshape(_data,(7,6))
        x_half_dot = _data[0]
        phi_half_dot = _data[1:]
        phi_half_dot_inv = -phi_half_inv @ phi_half_dot @ phi_half_inv


        # t1
        Y = np.append(x,phi)
        _data = self.predict_diff_eq(_t[2],Y0)
        _data = np.reshape(_data,(7,6))
        x_dot = _data[0]
        phi_dot = _data[1:]
        phi_dot_inv = -phi_inv @ phi_dot @ phi_inv

        # setup interpolators:
        # 1.interpolator for mean state (Hermite interpolator)
        self.state_interpolator = CubicHermiteSpline(
            _tflt[0:3],[x0,x_half,x],[x0_dot,x_half_dot,x_dot],extrapolate = False)

        # 2.interpolator for phi(t,t0) (Hermite interpolator)
        self.phi_interpolator = CubicHermiteSpline(
            _tflt[0:3],[phi0,phi_half,phi],[phi0_dot,phi_half_dot,phi_dot],extrapolate = False)

        # 3.interpolator for phi_inv(t,t0)
        self.phi_inv_interpolator = CubicHermiteSpline(
            _tflt[0:3],[phi0,phi_half_inv,phi_inv],[phi0_dot_inv,phi_half_dot_inv,phi_dot_inv],extrapolate = False)

        # 4.interpolator for short period function etas
        solver = integrate.solve_ivp(self.f, (_tflt[2],_tflt[3]),
                                     x, method=method, t_eval=[_tflt[3]])
        x2 = solver.y.flatten()
        orbs = [Orbit(xi,ti,"equinoctial_mean",frame,None) for ti,xi in zip(_t,[x0,x_half,x,x2])]
        Cy0 = getFourierCoefs(orbs[0],True)
        Cy1 = getFourierCoefs(orbs[1],True)
        Cy2 = getFourierCoefs(orbs[2],True)
        Cy3 = getFourierCoefs(orbs[3],True)
        dct = {}
        for _type in Cy0.keys():
            Cf0 = [Cy0[_type][0],Cy1[_type][0],Cy2[_type][0],Cy3[_type][0]]
            Cf1 = [Cy0[_type][1],Cy1[_type][1],Cy2[_type][1],Cy3[_type][1]]
            Cf2 = [Cy0[_type][2],Cy1[_type][2],Cy2[_type][2],Cy3[_type][2]]
            Cf3 = [Cy0[_type][3],Cy1[_type][3],Cy2[_type][3],Cy3[_type][3]]
            Cf4 = [Cy0[_type][4],Cy1[_type][4],Cy2[_type][4],Cy3[_type][4]]
            Cf5 = [Cy0[_type][5],Cy1[_type][5],Cy2[_type][5],Cy3[_type][5]]
            dct[_type] = {
                "0":CubicSpline(_tflt,Cf0,axis=0),
                "1":CubicSpline(_tflt,Cf1,axis=0),
                "2":CubicSpline(_tflt,Cf2,axis=0),
                "3":CubicSpline(_tflt,Cf3,axis=0),
                "4":CubicSpline(_tflt,Cf4,axis=0),
                "5":CubicSpline(_tflt,Cf5,axis=0)}
        self.short_period_interpolator = dct


    def observation_grid(self,y,h,R,t,frame,H,B1_fct):
        """update step
        """
        Sinv = None
        residuals = None

        # interpolate state , phi and get phi_inv from phi_S
        x_t = self.state_interpolator(t.mjd*86400)
        phi_t = self.phi_interpolator(t.mjd*86400)
        phi_inv_t = self.phi_S
        # print(x_t)
        # print(phi_t)
        # print(phi_inv_t)
        # exit()

        # interpolate short period function etas
        dctFouriers = self.interpolate_SPG(t.mjd*86400,self.short_period_interpolator)
        etas = self.getEtasFromFourierCoefs(dctFouriers,Orbit(x_t,t,"equinoctial_mean",frame,None),True)

        # compute transitional matrices and predicted corrections
        phi_obs = phi_t @ phi_inv_t
        self.corrections = phi_obs @ self.corrections

        # linearization of observation equation (prediction of predicted osculating elements)
        B1 = B1_fct(t,x_t)

        # discretization of Q
        Qd = phi_obs @ self.Q @ phi_obs.T * ((t - self._t).total_seconds())
        self.P = phi_obs @ self.P @ phi_obs.T + Qd #predicted covariance


        if y is not None:
            #for each observer, recalculate osc state (with new updated corrections)
            x_osc = x_t + self.corrections + etas + B1 @ self.corrections

            # residuals of observations and observation jacobian
            residuals = y - h(t,x_osc)
            H = H(t,x_osc) @ (np.eye(len(x_t)) + B1)

            # update phase of filter
            Sinv = np.linalg.inv(H @ self.P @ H.T + R)
            K = self.P @ H.T @ Sinv
            self.corrections += K @ residuals
            self.P = (np.eye(len(x_osc)) - K @ H) @ self.P

        self.mean_state = {"t":t,"x":x_t+self.corrections,"frame":frame}
        self.osc_state = {"t":t,"x":x_t + self.corrections + etas + B1 @ self.corrections,"frame":frame}
        self.nominal_mean_state = {"t":t,"x":x_t,"frame":frame}
        self.x = x_t+self.corrections

        # interpolate phi_inv to get phi_inv(t_k,t0), where t_k is this current epoch
        self.phi_S = self.phi_inv_interpolator(t.mjd*86400)
        self._t = t

        return Sinv,residuals

    def interpolate_SPG(self,t,interpolator):
        ret = {}
        for _type in interpolator.keys():
            Cf0 = interpolator[_type]["0"](t)
            Cf1 = interpolator[_type]["1"](t)
            Cf2 = interpolator[_type]["2"](t)
            Cf3 = interpolator[_type]["3"](t)
            Cf4 = interpolator[_type]["4"](t)
            Cf5 = interpolator[_type]["5"](t)
            ret[_type] = [Cf0,Cf1,Cf2,Cf3,Cf4,Cf5]
        return ret


    @property
    def mean_state(self):
        return self._mean_state
    @mean_state.setter
    def mean_state(self,info):
        self._mean_state = Orbit(info["x"],info["t"],"equinoctial_mean",info["frame"],None)

    @property
    def nominal_mean_state(self):
        return self._nominal_mean_state
    @nominal_mean_state.setter
    def nominal_mean_state(self,info):
        self._nominal_mean_state = Orbit(info["x"],info["t"],"equinoctial_mean",info["frame"],None)

    @property
    def osc_state(self):
        return self._osc_state
    @osc_state.setter
    def osc_state(self,info):
        self._osc_state = Orbit(info["x"],info["t"],"equinoctial_mean",info["frame"],None)

    @property
    def P_mean(self):
        return self.P
