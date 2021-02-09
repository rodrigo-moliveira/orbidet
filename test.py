import numpy as np
import orbidet
from beyond.beyond.dates import Date
from beyond.beyond.orbits import Orbit
from beyond.beyond.constants import Earth
from beyond.beyond.frames.orient import PEF,TOD,EME2000
from orbidet.force.drag import AtmosphericDrag,ExponentialDragDb
from orbidet.force.gravity import TwoBody,LowZonalHarmonics,GravityAcceleration
from orbidet.satellite import SatelliteSpecs
from orbidet.force import Force

# for i in range (30):
#     start = Date(2010,3,i+1,18,00,0)
#     X = np.array([6542.76,2381.36,-0.000102,0.3928,-1.0793,7.592])
#     mean = Orbit(X,start, "cartesian","EME2000",None)
#     mean.frame = "TOD"

start = Date(2010,3,1,18,00,0)
X = np.array([6542.76,2381.36,-0.000102,0.3928,-1.0793,7.592])
orbit = Orbit(X,start, "cartesian","EME2000",None)
# print(orbit)
orbit.frame="PEF"


T = EME2000.convert_to(start,TOD)
print(T @ X - orbit)


exit()
# print(repr(mean))
grav = GravityAcceleration(10,10)
a=grav.acceleration(np.array([6542.76,2381.36,-0.000102]))
drag = AtmosphericDrag()


force_model = Force()
force_model.addForce(grav)
force_model.addForce(drag)
print(force_model)
# print(a)

# sat = SatelliteSpecs("SAT1",2,10,20)
# DragHandler = ExponentialDragDb()
# drag = AtmosphericDrag()
# drag.acceleration( DragHandler, np.array([5000,200,6000]), np.array([1,2,3]),
#     sat,np.array([0,0,0.00002]))

# a=grav.acceleration(np.array([5000,200,6000]))
# print(a)
