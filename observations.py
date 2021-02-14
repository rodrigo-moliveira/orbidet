"""This script creates measurement records of ground station passes
"""

import numpy as np

from beyond.beyond.dates import Date, timedelta
from beyond.beyond.orbits import Orbit

from orbidet.propagators import ImportedProp
from orbidet.satellite import SatelliteSpecs
from orbidet.observers import GroundStation, get_obs



def main():
    # defining initial conditions and setting propagator
    start = Date(2010,3,1,18,00,0)
    filename = "./orbidet/data/trajectories/GMAT1.csv"
    prop = ImportedProp(start, filename=filename,frame="TOD")
    initialState = prop.orbit
    # print(repr(initialState))

    # getting generator
    step = timedelta(seconds = 60)
    stop = start + timedelta(hours = 24)
    gen = prop.iter(step=step,stop=stop,start=start)

    # setting up Ground Station (GS) observer
    sensors_GS = ["Range", "Azimuth","Elevation","Range Rate"]
    GS1_name = "Lisbon"
    GS1_latlonalt = (38,-10,0)

    # Noise figures
    std_range = (100) / 1000                 #range                              [km]
    std_angles = (0.02) * np.pi/180            #angle measurements                 [rad]
    std_rangerate = (10) / 100 / 1000       #range rate                         [km/s]
    dict_std = {"range":std_range,
                "angles":std_angles,
                "range rate":std_rangerate}

    ECEF_frame = "PEF" # the topocentric frame is defined with respect to ECEF_frame
    observer = GroundStation(GS1_name,GS1_latlonalt,sensors_GS,dict_std,ECEF_frame)
    # print(repr(observer))


    # generate orbit
    for orbit in gen:
        y = get_obs(orbit.copy(),observer,dict_std,apply_noise=True,do_LOS=True)
        if y is not None:
            print(orbit.date, y)











if __name__ == "__main__":
    main()
