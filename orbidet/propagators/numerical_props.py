import csv
import warnings
from scipy.integrate import solve_ivp

from beyond.beyond.propagators.base import NumericalPropagator
from beyond.beyond.dates import timedelta
from beyond.beyond.orbits import Orbit,Ephem

from orbidet.force import Force

class ImportedProp(NumericalPropagator):
    """Wrapper of propagator imported from a previosly generated computation.
    Propagation ephemeride is provided by a .CSV file
    """

    FRAME = "EME2000"
    FORM = "cartesian"


    def __init__(self, epoch0, filename="",frame=FRAME,form=FORM,**kwargs):
        """
        Args:
            step (datetime.timedelta): Step size to output results (interpolations may be done in case
                                        the requested step is different (and not multiple) of the provided
                                        by the file)
            stop (datetime.timedelta or Beyond.Date): stopping epoch of "propagation"
            frame (str): Frame used in the propagation (default is J2000)
            filename (str): name of the .CSV file with the propagation
            start (beyond.Date or convertable to Date with Date(start) [see Beyond documentation])

        Format of file (assumed cartesian form):
            t,x,y,z,vx,vy,vz
                where:
                    t is in [s] (relative to start(initial epoch))
                    x,y,z are coordinates in frame FRAME in [km]
                    vx,vy,vz are in [km/s]
        """
        self.frame = frame
        self.form=form
        self.filename = filename

        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            coord = next(reader)
            self.orbit = Orbit(coord[1:], epoch0, self.form, frame, None)
            self.step = timedelta(seconds=float(next(reader)[0]) - float(coord[0])).total_seconds()
            # step of the file

    @property
    def orbit(self):
        return self._orbit if hasattr(self, "_orbit") else None

    @orbit.setter
    def orbit(self, orbit):
        self._orbit = orbit.copy(form=self.form, frame=self.frame)

    def copy(self):
        return self.__class__(
            self.epoch0, self.filename, frame=self.frame
        )
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

        # Not very clean !
        if step is self.step:
            step = None

        # checking whether or not interpolations are needed
        if step.total_seconds() % self.step == 0:
            interpolation = False
        else:
            interpolation = True
            warnings.warn("Warning: Interpolation will be needed in the External Propagator (from file) "
                          "Note that interpolation is EXPENSIVE (memory-wise)")
        orb = self.orbit

        if not interpolation:
            with open(self.filename) as f:
                reader = csv.reader(f)
                next(reader)
                for coord in reader:
                    if float(coord[0]) % step.total_seconds() != 0:
                        continue
                    date = self.orbit.date + timedelta(seconds=float(coord[0]))
                    if date > stop: break
                    yield Orbit(coord[1:],date, self.form,self.frame,None)
        else:
            ephem = [orb]
            date = start
            with open(self.filename) as f:
                reader = csv.reader(f)
                next(reader)
                while date < stop:
                    coord = next(reader)
                    date = self.orbit.date + timedelta(seconds=float(coord[0]))
                    ephem.append(Orbit(coord[1:],date,self.form,self.frame,None))

            ephem = Ephem(ephem)
            ephem_iter = ephem.iter(dates=dates, step=step)

            for orb in ephem_iter:
                yield orb







class Cowell(NumericalPropagator):
    """Cowell numerical propagator with configurable force model through a *Force* instance.

    integration of force model is performed with external integrators from scipy.solve_ivp
    # url: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """

    RK45 = 'RK45'
    DOP853 = 'DOP853'
    FRAME = "EME2000"

    def __init__(self, step, force, *, method=RK45, frame=FRAME, tol=1e-4):
        """
        Args:
            step (datetime.timedelta): Step size of the propagator
            force (Force):
            method (str): Integration method (see class attributes)
            frame (str): Frame to use for the propagation
            tol (float): Error tolerance for adaptive stepsize methods
        """

        self.step = step
        assert isinstance(force, Force), "Force model should be of type Orbidet.force.force.Force"
        self.force = force
        self.method = method
        self.frame = frame
        self.tol = tol

    def copy(self):
        return self.__class__(
            self.step, self.force, method=self.method, frame=self.frame,tol=self.tol
        )

    @property
    def orbit(self):
        return self._orbit if hasattr(self, "_orbit") else None

    @orbit.setter
    def orbit(self, orbit):
        self._orbit = orbit.copy(form="cartesian", frame=self.frame)


    def _make_step(self, orb, step):
        """
        Compute the next step with scipy.solve_ivp and selected method (RK45 or DOP853)
        """
        #defining datetime objects of beginning and end
        date_in = orb.date
        date_out = orb.date + step

        #run solver
        solver = solve_ivp(self._fun, (date_in.mjd*86400,date_out.mjd*86400), orb, method=self.method,
                           t_eval=[date_out.mjd*86400],rtol = self.tol, atol = self.tol/1e3,
                           args = (self.force_model,))
        x = solver.y.flatten()
        return Orbit(date_out,x,"cartesian",self.frame,None)


    def _iter(self, **kwargs):
        print("ola");exit()
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

        orb = self.orbit
        yield orb
        # In order to compute the propagation with the reference step size
        # (ie self.step), but give the result at the requested step size
        # (ie step), we use an Ephem object for interpolation
        # ephem = [orb]

        date = start
        while date < stop:
            orb = self._make_step(orb, step)
            date += step
            yield orb
            # ephem.append(orb)
            # date += real_step


        # ephem = Ephem(ephem)
        #
        # if kwargs.get("real_steps", False):
        #     ephem_iter = ephem.iter(dates=dates, listeners=listeners)
        # else:
        #     ephem_iter = ephem.iter(dates=dates, step=step, listeners=listeners)
        #
        # for orb in ephem_iter:
        #     yield orb
