"""File demonstrating the use of propagators (examples)
"""
import numpy as np

from beyond.beyond.dates import Date, timedelta
from beyond.beyond.orbits import Orbit

from orbidet.propagators import ImportedProp, Cowell
from orbidet.force import Force,TwoBody,AtmosphericDrag,GravityAcceleration,ExponentialDragDb
from orbidet.satellite import SatelliteSpecs

def ImportedPropExample():
    # defining initial conditions and setting propagator
    start = Date(2010,3,1,18,00,0)
    filename = "./orbidet/data/trajectories/GMAT1.csv"
    prop = ImportedProp(start, filename=filename)
    initialState = prop.orbit
    # print(repr(initialState))

    # getting generator
    step = timedelta(seconds = 5)
    stop = start + timedelta(hours = 1)
    gen = prop.iter(step=step,stop=stop,start=start)

    # generate orbit
    for orbit in gen:
        print(orbit.date, orbit)


def CowellExample():
    # defining initial conditions & frames
    start = Date(2010,3,1,18,00,0)
    step = timedelta(seconds = 5)
    stop = start + timedelta(hours = 1)
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
    grav = GravityAcceleration(5,5)
    DragHandler = ExponentialDragDb()
    drag = AtmosphericDrag(sat,DragHandler)
    two_body = TwoBody()
    force.addForce(grav)
    force.addForce(drag)
    force.addForce(two_body)
    # print(force)

    # creating propagator & generator
    prop = Cowell(step,force,method="RK45",frame=initialOrbit.frame)
    initialOrbit.propagator = prop
    gen = initialOrbit.iter(stop=stop,step=step)

    # generate orbit
    for orbit in gen:
        print(orbit.date,orbit[0],orbit[1],orbit[2])#, orbit)



def main():
    CowellExample()

if __name__ == "__main__":
    main()
