import numpy as np
import orbidet
from beyond.beyond.dates import Date
from beyond.beyond.orbits import Orbit
from beyond.beyond.constants import Earth
from orbidet.force.drag import AtmosphericDrag

# for i in range (30):
#     start = Date(2010,3,i+1,18,00,0)
#     X = np.array([6542.76,2381.36,-0.000102,0.3928,-1.0793,7.592])
#     mean = Orbit(X,start, "cartesian","EME2000",None)
#     mean.frame = "TOD"


# print(repr(mean))

drag = AtmosphericDrag("drag")
drag.acceleration( 2, 3, 4,5)
