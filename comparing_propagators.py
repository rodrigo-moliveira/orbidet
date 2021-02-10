"""In this file the Cowell and Semianalytical propagators are compared (in terms of outputs)
"""
import numpy as np

from beyond.beyond.dates import Date, timedelta
from beyond.beyond.orbits import Orbit

from orbidet.propagators import ImportedProp, Cowell, Semianalytical
from orbidet.force import Force,TwoBody,AtmosphericDrag,GravityAcceleration,ExponentialDragDb
from orbidet.satellite import SatelliteSpecs
from orbidet.metrics.plot_utils import *


forms = {'cartesian':['x','y','z','vx','vy','vz'],
         'keplerian':["a", "e", "i", "RAAN", "w", "TA"],
         'keplerian_mean':["a", "e", "i", "RAAN", "w", "M"],
         "equinoctial_mean":["a","h","k","p","q","lmb"]
         }



def main():
    # defining initial conditions & frames
    start = Date(2010,3,1,18,00,0)
    output_step = timedelta(seconds = 60)
    propagation_step = timedelta(hours = 1)
    stop = start + timedelta(hours = 5)
    integrationFrame = "TOD"
    gravityFrame = "PEF"
    initialOrbit = Orbit(np.array([6542.76,2381.36,-0.000102,0.3928,-1.0793,7.592]),
                        start,"cartesian",integrationFrame,None)

    # creating satellite
    sat = SatelliteSpecs("SAT1", #name
                        2,       #CD
                        50,      #mass [kg]
                        2)      #area [m²]

    # creating force model
    force = Force(integrationFrame = integrationFrame, gravityFrame = gravityFrame)
    grav = GravityAcceleration(5,0)
    DragHandler = ExponentialDragDb()
    drag = AtmosphericDrag(sat,DragHandler)
    two_body = TwoBody()
    force.addForce(grav)
    # force.addForce(drag)
    force.addForce(two_body)
    # print(force)

    # creating Semianalytical propagator & generator
    prop_semianalytical = Semianalytical(propagation_step,force,method="RK45",frame=initialOrbit.frame,
                        quadrature_order = 20, DFT_lmb_len = 32, DFT_sideral_len=32,
                        outputs=("mean","osculating"))
    orbit_cow = initialOrbit.copy()
    orbit_cow.propagator = prop_semianalytical
    orbit_cow.state = "osculating"
    gen_semianalytical = orbit_cow.iter(stop=stop,step=output_step)


    # creating Cowell propagator & generator
    prop = Cowell(output_step,force,method="DOP853",frame=initialOrbit.frame)
    orbit_semi = initialOrbit.copy()
    orbit_semi.propagator = prop
    gen_cowell = orbit_semi.iter(stop=stop,step=output_step)


    ephm_cow = []
    ephm_semi_osc = []
    ephm_semi_mean = []
    output_form = "equinoctial_mean"
    # generate orbit
    for (semi_mean,semi_osc),cow in zip(gen_semianalytical,gen_cowell):

        ephm_semi_osc.append(semi_osc.copy(form=output_form))
        ephm_semi_mean.append(semi_mean.copy(form=output_form))
        ephm_cow.append(cow.copy(form=output_form))

        print(cow.date)


    # Time array
    dt = (ephm_cow[1].date - ephm_cow[0].date).total_seconds()
    delta_t = (ephm_cow[-1].date - ephm_cow[0].date).total_seconds()
    t = [t_i for t_i in range(0,round(delta_t+dt),round(dt))]
    xlabel = 'Time [s]'


    for ephm,label in zip([ephm_semi_osc,ephm_semi_mean,ephm_cow],
                ["Semianalytical osc","Semianalytical mean", "Cowell osc"]):
        x0 = [x[0] for x in ephm]
        x1 = [x[1] for x in ephm]
        x2 = [x[2] for x in ephm]
        x3 = [x[3] for x in ephm]
        x4 = [x[4] for x in ephm]
        x5 = [x[5] for x in ephm]
        plot_graphs(x0,t,forms[output_form][0],"",xlabel,i=0,label=label,show_label=True)
        plot_graphs(x1,t,forms[output_form][1],"",xlabel,i=1,label=label,show_label=True)
        plot_graphs(x2,t,forms[output_form][2],"",xlabel,i=2,label=label,show_label=True)
        plot_graphs(x3,t,forms[output_form][3],"",xlabel,i=3,label=label,show_label=True)
        plot_graphs(x4,t,forms[output_form][4],"",xlabel,i=4,label=label,show_label=True)
        plot_graphs(x5,t,forms[output_form][5],"",xlabel,i=5,label=label,show_label=True)
    show_plots(True)














if __name__ == "__main__":
    main()
