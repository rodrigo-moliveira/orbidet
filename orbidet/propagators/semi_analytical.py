import csv
import warnings
from scipy.integrate import solve_ivp
import numpy as np
from scipy.interpolate import CubicHermiteSpline,lagrange,interp1d,CubicSpline

from beyond.propagators.base import NumericalPropagator
from beyond.dates import timedelta
from beyond.orbits.ephem import Ephem
from  beyond.orbits import Orbit,Ephem
from beyond.frames.iau1980 import _sideral

from .force_model import Force
from orbidet.utils.diff_eqs import equinoctial_state_fct,_PEF_to_ITRF,_J2000_to_TOD
from .short_period_transf import NumAv_MEP_transf
from orbidet.cython_modules.semi_analytical import equinoctial_state_fct as cy_equinoctial_state_fct

from beyond.constants import Earth as EARTH
mu = EARTH.mu


class SemiAnalytical(NumericalPropagator):
    """Semianalytical propagator with configurable force model through a *Force* instance.
    orbit.state:
        *"osculating"
        *"mean"
    """

    # integration methods (for scipy.solve_ivp). See Scipy documentation
    # url: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    RK45 = 'RK45'
    DOP853 = 'DOP853'

    FRAME = "EME2000"

    def __init__(self, step, force_model : Force, *, method=RK45, frame=FRAME, tol=1e-4,quadrature_order = 30,
                 DFT_lmb_len = 64, DFT_sideral_len=32,outputs=("mean","osculating")):
        """
        Args:
            step (datetime.timedelta): Step size of the propagator
            force_model (Force):
            method (str): Integration method (see class attributes)
            frame (str): Frame to use for the propagation
            tol (float): Error tolerance for adaptive stepsize methods
            outputs (tupe): Type of orbit to output ("mean" and/or "osculating")
        """
        self.step = step
        self.force_model = force_model
        self.method = method
        self.frame = frame
        self.tol = tol
        self.state_transformer = NumAv_MEP_transf(force_model,DFT_lmb_len,DFT_sideral_len)
        self.quadrature_order = quadrature_order
        if force_model.CYTHON:
            self._fun = cy_equinoctial_state_fct
        else:
            self._fun = equinoctial_state_fct

        drag = force_model.DRAG
        if drag:
            self.zonals_drag = ("zonals","drag")
        else:
            self.zonals_drag = ("zonals",)

        if "mean" in outputs and "osculating" in outputs:
            self.outputs = "both"
        elif "mean" in outputs:
            self.outputs = "mean"
        elif "osculating" in outputs:
            self.outputs = "osculating"
        else:
            raise Exception("outputs should either be 'mean' and / or 'osculating'")

    def copy(self):
        return self.__class__(
            self.step, self.force_model, method=self.method, frame=self.frame,tol=self.tol
        )

    @property
    def orbit(self):
        return self._orbit if hasattr(self, "_orbit") else None

    @orbit.setter
    def orbit(self, orbit):
        orb = orbit.copy(form = "equinoctial")
        self.frame = orbit.frame

        if not hasattr(orbit, "state"):
            raise Exception("An additional user defined attribute *state* should be created to ",
                            "set up the semi analytical propagator. either state=\"mean\" or \"osculating\"")
        elif orbit.state == "mean":
            pass
        elif orbit.state == "osculating":
            orb = self.state_transformer.osc_to_mean(orb)
        else:
            raise Exception("'state' should either be 'mean' or 'osculating'")
        self._orbit = orb.copy(form="equinoctial", frame=self.frame)


    def _make_step(self, orb, step):
        """
        Compute the next step with scipy.solve_ivp and selected method (RK45 or DOP853)
        """
        #defining datetime objects of beginning and end
        date_in = orb.date
        date_out = orb.date + step

        # initial conditions setup
        y0 = np.array(orb)

        #run solver
        solver = solve_ivp(self._fun, (date_in.mjd*86400,date_out.mjd*86400), y0, method=self.method,
                           t_eval=[date_out.mjd*86400],rtol = self.tol, atol = self.tol/1e3,
                           args = (self.force_model,self.zonals_drag,self.quadrature_order))

        x = solver.y.flatten()
        x[5] = x[5] % (2*np.pi)

        return Orbit(date_out,x,"equinoctial",self.frame,None)


    def _iter(self, **kwargs):

        dates = kwargs.get("dates")

        if dates is not None:
            start = dates.start
            stop = dates.stop
            step = None
        else:
            start = kwargs.get("start", self.orbit.date)
            stop = kwargs.get("stop")
            step = kwargs.get("step")

        listeners = kwargs.get("listeners", [])

        orb_mean = self.orbit
        if self.outputs is "mean":
            yield orb_mean
        else:
            orb_osc = self.state_transformer.mean_to_osc(orb_mean.copy(form="equinoctial"))
            if self.outputs is "both":
                yield orb_mean,orb_osc
            else:
                yield orb_osc


        date = start
        while date < stop:
            orb_mean = self._make_step(orb_mean, step)
            date += step
            if self.outputs is "mean":
                yield orb_mean
            else:
                orb_osc = self.state_transformer.mean_to_osc(orb_mean.copy(form="equinoctial"))
                if self.outputs is "both":
                    yield orb_mean,orb_osc
                else:
                    yield orb_osc




class Num_AvMEP(NumericalPropagator):
    """Semianalytical propagator with configurable force model through a *Force* instance.
    orbit.state:
        *"osculating"
        *"mean"
    """

    # integration methods (for scipy.solve_ivp). See Scipy documentation
    # url: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    RK45 = 'RK45'
    DOP853 = 'DOP853'

    FRAME = "EME2000"

    def __init__(self, step, force_model : Force, *, method=RK45, frame=FRAME, tol=1e-4,quadrature_order = 30,
                 DFT_lmb_len = 64, DFT_sideral_len=32,outputs=("mean","osculating")):
        """
        Args:
            step (datetime.timedelta): Step size of the propagator
            force_model (Force):
            method (str): Integration method (see class attributes)
            frame (str): Frame to use for the propagation
            tol (float): Error tolerance for adaptive stepsize methods
            outputs (tupe): Type of orbit to output ("mean" and/or "osculating")
        """
        self.step = step
        self.force_model = force_model
        self.method = method
        self.frame = frame
        self.tol = tol
        self.state_transformer = NumAv_MEP_transf(force_model,DFT_lmb_len,DFT_sideral_len)
        self.quadrature_order = quadrature_order
        self._fun = force_model.equinoctial_state_fct

        drag = force_model.DRAG
        if drag:
            self.zonals_drag = ("zonals","drag")
        else:
            self.zonals_drag = ("zonals",)

        if "mean" in outputs and "osculating" in outputs:
            self.outputs = "both"
        elif "mean" in outputs:
            self.outputs = "mean"
        elif "osculating" in outputs:
            self.outputs = "osculating"
        else:
            raise Exception("outputs should either be 'mean' and / or 'osculating'")

    def copy(self):
        return self.__class__(
            self.step, self.force_model, method=self.method, frame=self.frame,tol=self.tol
        )

    @property
    def orbit(self):
        return self._orbit if hasattr(self, "_orbit") else None

    @orbit.setter
    def orbit(self, orbit):
        orb = orbit.copy(form = "equinoctial")
        self.frame = orbit.frame

        if not hasattr(orbit, "state"):
            raise Exception("An additional user defined attribute *state* should be created to ",
                            "set up the semi analytical propagator. either state=\"mean\" or \"osculating\"")
        elif orbit.state == "mean":
            pass
        elif orbit.state == "osculating":
            orb = self.state_transformer.osc_to_mean(orb)
        else:
            raise Exception("'state' should either be 'mean' or 'osculating'")
        self._orbit = orb.copy(form="equinoctial", frame=self.frame)



    def _initialize_AOG_interpolator(self,orb,start,step):
        # mean state interpolator
        t = [start,start+self.step,start+2*self.step,start+3*step]
        _t = [t.mjd*86400 for t in t]
        solver = solve_ivp(self._fun, (_t[0],_t[-1]), np.array(orb), method=self.method,
                           t_eval=_t,rtol = self.tol, atol = self.tol/1e3,
                           args = (self.force_model,self.zonals_drag,self.quadrature_order))
        _x = solver.y;y0 = _x[:,0];y1 = _x[:,1];y2 = _x[:,2];y3 = _x[:,3]

        y0_dot = self._fun(t[0],y0,self.force_model,self.zonals_drag,self.quadrature_order)
        y1_dot = self._fun(t[1],y1,self.force_model,self.zonals_drag,self.quadrature_order)
        y2_dot = self._fun(t[2],y2,self.force_model,self.zonals_drag,self.quadrature_order)
        y3_dot = self._fun(t[3],y3,self.force_model,self.zonals_drag,self.quadrature_order)
        self._interpolator_data = [[t[0],y0,y0_dot],[t[1],y1,y1_dot],[t[2],y2,y2_dot],[t[3],y3,y3_dot]]
        return CubicHermiteSpline(_t[0:3],[y0,y1,y2],[y0_dot,y1_dot,y2_dot],extrapolate = False)

    def _update_AOG_interpolator(self,step):
        """delete first point of self._interpolator data, get new one to the
        right and create new interpolator object
        """
        self._interpolator_data.pop(0)
        t = self._interpolator_data[-1][0]
        t_new = t + step
        y = self._interpolator_data[-1][1]

        solver = solve_ivp(self._fun, (t.mjd*86400,t_new.mjd*86400), y, method=self.method,
                           t_eval=[t_new.mjd*86400],rtol = self.tol, atol = self.tol/1e3,
                           args = (self.force_model,self.zonals_drag,self.quadrature_order))
        y_new = solver.y.flatten()
        y_dot = self._fun(t_new,y_new,self.force_model,self.zonals_drag,self.quadrature_order)
        self._interpolator_data.append([t_new,y_new,y_dot])
        aux = self._interpolator_data

        t = [aux[i][0].mjd*86400 for i in range(1,4)]
        ys = [aux[i][1] for i in range(1,4)]
        ys_dot = [aux[i][2] for i in range(1,4)]
        return CubicHermiteSpline(t,ys,ys_dot,extrapolate = False)


    def initialize_SPG_interpolator(self):
        aux = self._interpolator_data
        t = [x[0] for x in aux]
        _t = [ti.mjd*86400 for ti in t]
        y = [x[1] for x in aux]
        orbs = [Orbit(ti,xi,"equinoctial",self.frame,None) for ti,xi in zip(t,y)]

        Cy0 = self.state_transformer.getFourierCoefs(orbs[0],True)
        Cy1 = self.state_transformer.getFourierCoefs(orbs[1],True)
        Cy2 = self.state_transformer.getFourierCoefs(orbs[2],True)
        Cy3 = self.state_transformer.getFourierCoefs(orbs[3],True)

        dct = {}
        for _type in Cy0.keys():
            Cf0 = [Cy0[_type][0],Cy1[_type][0],Cy2[_type][0],Cy3[_type][0]]
            Cf1 = [Cy0[_type][1],Cy1[_type][1],Cy2[_type][1],Cy3[_type][1]]
            Cf2 = [Cy0[_type][2],Cy1[_type][2],Cy2[_type][2],Cy3[_type][2]]
            Cf3 = [Cy0[_type][3],Cy1[_type][3],Cy2[_type][3],Cy3[_type][3]]
            Cf4 = [Cy0[_type][4],Cy1[_type][4],Cy2[_type][4],Cy3[_type][4]]
            Cf5 = [Cy0[_type][5],Cy1[_type][5],Cy2[_type][5],Cy3[_type][5]]
            dct[_type] = {
                "0":CubicSpline(_t,Cf0,axis=0),
                "1":CubicSpline(_t,Cf1,axis=0),
                "2":CubicSpline(_t,Cf2,axis=0),
                "3":CubicSpline(_t,Cf3,axis=0),
                "4":CubicSpline(_t,Cf4,axis=0),
                "5":CubicSpline(_t,Cf5,axis=0)}

        # update Fourier Coefs. to the container of interpolation data
        for data,Fr in zip(aux,[Cy0,Cy1,Cy2,Cy3]):
            data.append(Fr)
        return dct

    def _update_SPG_interpolator(self):
        aux = self._interpolator_data
        orb = Orbit(aux[-1][0],aux[-1][1],"equinoctial",self.frame,None)
        Cy = self.state_transformer.getFourierCoefs(orb,True)
        aux[-1].append(Cy)
        _t = [ti[0].mjd*86400 for ti in aux]

        dct = {}
        for _type in Cy.keys():
            Cf0 = [aux[0][-1][_type][0],aux[1][-1][_type][0],aux[2][-1][_type][0],aux[3][-1][_type][0]]
            Cf1 = [aux[0][-1][_type][1],aux[1][-1][_type][1],aux[2][-1][_type][1],aux[3][-1][_type][1]]
            Cf2 = [aux[0][-1][_type][2],aux[1][-1][_type][2],aux[2][-1][_type][2],aux[3][-1][_type][2]]
            Cf3 = [aux[0][-1][_type][3],aux[1][-1][_type][3],aux[2][-1][_type][3],aux[3][-1][_type][3]]
            Cf4 = [aux[0][-1][_type][4],aux[1][-1][_type][4],aux[2][-1][_type][4],aux[3][-1][_type][4]]
            Cf5 = [aux[0][-1][_type][5],aux[1][-1][_type][5],aux[2][-1][_type][5],aux[3][-1][_type][5]]

            dct[_type] = {
                    "0":CubicSpline(_t,Cf0,axis=0),
                    "1":CubicSpline(_t,Cf1,axis=0),
                    "2":CubicSpline(_t,Cf2,axis=0),
                    "3":CubicSpline(_t,Cf3,axis=0),
                    "4":CubicSpline(_t,Cf4,axis=0),
                    "5":CubicSpline(_t,Cf5,axis=0)}

        return dct

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


    def _iter(self, **kwargs):

        dates = kwargs.get("dates")

        if dates is not None:
            start = dates.start
            stop = dates.stop
            step = None
        else:
            start = kwargs.get("start", self.orbit.date)
            stop = kwargs.get("stop")
            step = kwargs.get("step")

        # intializations
        orb_mean = self.orbit
        self.interpolatorAOG = self._initialize_AOG_interpolator(orb_mean,start,self.step)
        self.interpolatorSPG = self.initialize_SPG_interpolator()

        # yield initial results (for t0)
        # output results
        if self.outputs is "mean":
            yield orb_mean
        else:
            orb_osc = self.state_transformer.mean_to_osc(orb_mean.copy(form="equinoctial"))
            if self.outputs is "both":
                yield orb_mean,orb_osc
            else:
                yield orb_osc


        # make the integration
        date = start + step
        new_integration_date = start + self.step
        control = 1
        while date < stop:
            while date < new_integration_date:
                # perform mean element interpolation and SPG step to get osculating results
                x_mean = self.interpolatorAOG(date.mjd*86400)
                orb_mean = Orbit(date,x_mean,"equinoctial",self.frame,None)

                date = date + step
                # output results
                if self.outputs is "mean":
                    yield orb_mean
                else:
                    # orb_osc = self.state_transformer.mean_to_osc(orb_mean.copy(form="equinoctial"))
                    dctFouriers = self.interpolate_SPG(date.mjd*86400,self.interpolatorSPG)
                    orb_osc = self.state_transformer.mean_to_osc(orb_mean.copy(),
                                self.state_transformer.getEtasFromFourierCoefs(dctFouriers,orb_mean,True))

                    if self.outputs is "both":
                        yield orb_mean,orb_osc
                    else:
                        yield orb_osc

            new_integration_date = date + self.step


            if control < 3:
                if control == 2:
                    aux = self._interpolator_data
                    t = [aux[i][0].mjd*86400 for i in range(1,4)]
                    ys = [aux[i][1] for i in range(1,4)]
                    ys_dot = [aux[i][2] for i in range(1,4)]
                    self.interpolatorAOG = CubicHermiteSpline(t,ys,ys_dot,extrapolate = False)
                control += 1
            else:
                self.interpolatorAOG = self._update_AOG_interpolator(self.step)
                self.interpolatorSPG = self._update_SPG_interpolator()
