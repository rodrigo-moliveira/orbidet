"""This script implements the orbit determination filters in the following sequence:
    - Initialization of filter:
        Gauss initial orbit determination
        Batch least squares
    - Kalman filter - Cowell filter (EKF or UKF), Semianalytical filter (ESKF or USKF)
"""

import numpy as np

from beyond.beyond.dates import Date, timedelta
from beyond.beyond.orbits import Orbit

from orbidet.propagators import ImportedProp
from orbidet.satellite import SatelliteSpecs
from orbidet.observers import GroundStation, get_obs
from orbidet.force import Force,TwoBody,AtmosphericDrag,GravityAcceleration,ExponentialDragDb

from orbidet.IOD.Gauss import Gauss
from orbidet.estimators import LeastSquares,OsculatingFilter
from orbidet.errors import LSnotConverged


def main():
    ################# defining initial conditions and simulation setup #################
    start = Date(2010,3,1,18,00,0)
    step = timedelta(seconds = 5)
    stop = start + timedelta(hours = 24)
    MonteCarlo = 1 #number of Monte Carlo runs to run the simulation

    # frames
    frameECI = "TOD"
    frameECEF = "PEF"

    # Control flags
    apply_noise = True              #Measurement Noise
    LOS = True                     #use LOS condition to validate measurements

    # reference trajectory
    filename = "./orbidet/data/trajectories/GMAT1.csv"
    prop = ImportedProp(start, filename=filename,frame="TOD")
    initialState = prop.orbit

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

    observer = GroundStation(GS1_name,GS1_latlonalt,sensors_GS,dict_std,frameECEF)
    # print(repr(observer))

    # creating satellite & force model
    sat = SatelliteSpecs("SAT1", #name
                        2,       #CD
                        50,      #mass [kg]
                        2)      #area [m²]
    force = Force(integrationFrame = frameECI, gravityFrame = frameECEF)
    grav = GravityAcceleration(5,5)
    DragHandler = ExponentialDragDb()
    drag = AtmosphericDrag(sat,DragHandler)
    two_body = TwoBody()
    force.addForce(grav)
    force.addForce(drag)
    force.addForce(two_body)
    # print(force)


    ################# Estimation Setup #################
    InitialOD_LEN = 10 # Number of observations to collect in initialization procedure
    filterName = "EKF" #possible filters: EKF, UKF, ESKF or USKF
    Q_cartesian = np.block([[10**-9*np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),10**-12*np.eye(3)]]) #process noise cov



    ################# simulation #################
    run = 1
    while run <= MonteCarlo:
        print('Monte Carlo run %d'%run)

        gen = prop.iter(step=step,stop=stop,start=start)
        # The algorithm should only start when there is LOS (consume observations until LOS)
        if LOS:
            y = None
            while y is None:
                orbit = next(gen)
                y = get_obs(orbit.copy(),observer,dict_std,apply_noise,LOS)


        # setup Initial Orbit Determination (10 observations)
        IOD = []
        while len(IOD) < InitialOD_LEN:
            orbit = next(gen)
            y = get_obs(orbit,observer,dict_std,apply_noise,LOS)
            if y is not None:
                IOD.append([orbit.date,y])

        # run Gauss Initial Orbit Determination algorithm
        gauss = Gauss(IOD[0:3],observer,frameECI)

        # run batch least squares
        t = IOD[3][0]
        orbit = gauss.get_propagated_solution(t,force)
        try:
            LS = LeastSquares(IOD[3:],orbit,observer.R_default,0.1,observer.jacobian_h,observer.h,force)
        except LSnotConverged:
            print("this MC did not converge. Retrying this Monte Carlo run")
            continue
        orbit,P,t = LS.get_propagated_solution()


        # create Kalman filter
        filter = OsculatingFilter(filterName,orbit,t,P,Q_cartesian, observer,force,frameECI,"RK45")








        run += 1










if __name__ == "__main__":
    main()
