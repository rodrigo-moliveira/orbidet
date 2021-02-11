import numpy as np
from numpy import cos,sin,arctan2

from beyond.constants import Earth
from beyond.orbits import Orbit

from orbidet.utils.diff_eqs import osculating_state_fct as fun
COLOCAR O PROPAGATOR EM VEZ DISTO

mu = Earth.mu

"""
In this file the gauss preliminary orbit determination Algorithm is coded.

It is assumed that the observations

GAUSS method
 This function uses the Gauss method with iterative
 improvement (Algorithms 5.5 and 5.6) to calculate the state
 vector of an orbiting body from angles-only observations at
 three closely-spaced times.

#Inputs:
t1, t2, t3 - the times of the observations (s)
R1, R2, R3 - the observation site position vectors
                at t1, t2, t3 in ECI coordinates (km)
Rho1, Rho2, Rho3 - the direction cosine vectors of the
                satellite at t1, t2, t3 in ECI coordinates

#Outputs:
(r,v) - state vector (position [km], velocity [km/s]) at the end of the alg
(r_old, v_old) - state vector estimated at the end of alg 5.5
"""

# utility functions used in GAUSS IOD
def _az_el_to_ra_dec(az,el,ECI_to_horizon_fct,date):
    """
    given the observation azimuth (Az) and elevation (el), obtain topocentric right ascension (ra) and declination (dec)
    """
    R_hor_to_ECI = ECI_to_horizon_fct(date).T
    rho_horizon = np.array( [cos(el)*sin(az) ,cos(el)*cos(az) ,sin(el)] )
    rho_ECI = R_hor_to_ECI @ rho_horizon

    ra = arctan2((rho_ECI[1]),(rho_ECI[0]))
    dec = arctan2((rho_ECI[2]),(np.sqrt((rho_ECI[0])**2+(rho_ECI[1])**2)))
    return(ra,dec)


class Gauss():

    def __init__(self,IOD, ECI_to_horizon_fct,sensors,pos_R, ECI_frame):
        epoch1 = IOD[0][0];epoch2 = IOD[1][0];epoch3 = IOD[2][0]
        t1 = 0
        t2 = (epoch2-epoch1).total_seconds()
        t3 = (epoch3-epoch1).total_seconds()
        self.date2 = epoch2

        if "Azimuth" in sensors and "Elevation" in sensors:
            az1 = IOD[0][1][sensors.index("Azimuth")]
            el1 = IOD[0][1][sensors.index("Elevation")]
            ra1,dec1 = _az_el_to_ra_dec(az1,el1,ECI_to_horizon_fct,epoch1)

            az2 = IOD[1][1][sensors.index("Azimuth")]
            el2 = IOD[1][1][sensors.index("Elevation")]
            ra2,dec2 = _az_el_to_ra_dec(az2,el2,ECI_to_horizon_fct,epoch2)

            az3 = IOD[2][1][sensors.index("Azimuth")]
            el3 = IOD[2][1][sensors.index("Elevation")]
            ra3,dec3 = _az_el_to_ra_dec(az3,el3,ECI_to_horizon_fct,epoch3)


        elif "Right Ascension" in sensors and "Declination" in sensors:
            ra1 = IOD[0][1][sensors.index('Right Ascension')]
            dec1 = IOD[0][1][sensors.index('Declination')]

            ra2 = IOD[1][1][sensors.index('Right Ascension')]
            dec2 = IOD[1][1][sensors.index('Declination')]

            ra3 = IOD[2][1][sensors.index('Right Ascension')]
            dec3 = IOD[3][1][sensors.index('Declination')]
        else:
            raise Exception("Either the pair (Azimuth,Elevation) or (Right Ascension, Declination) is needed",
                            "to perform the Gauss method")


        rho1 = np.array([cos(dec1)*cos(ra1),cos(dec1)*sin(ra1),sin(dec1)])
        rho2 = np.array([cos(dec2)*cos(ra2),cos(dec2)*sin(ra2),sin(dec2)])
        rho3 = np.array([cos(dec3)*cos(ra3),cos(dec3)*sin(ra3),sin(dec3)])

        R1 = pos_R(epoch1,ECI_frame)
        R2 = pos_R(epoch2,ECI_frame)
        R3 = pos_R(epoch3,ECI_frame)

        self.ECI_frame=ECI_frame

        self._solve(t1,t2,t3,R1,R2,R3,rho1,rho2,rho3)

    def _solve(self,t1,t2,t3,R1,R2,R3,Rho1,Rho2,Rho3):

        #step 1
        tau1 = (t1 - t2)
        tau3 = (t3 - t2)
        tau = tau3 - tau1

        #step 2 - Independent cross products among the direction cosine vectors:
        p1 = np.cross(Rho2,Rho3)
        p2 = np.cross(Rho1,Rho3)
        p3 = np.cross(Rho1,Rho2)

        #step 3
        Do = np.dot(Rho1,p1)

        #step 4 and 5
        D = np.array([[np.dot(R1,p1), np.dot(R1,p2), np.dot(R1,p3)],
                [np.dot(R2,p1), np.dot(R2,p2),np.dot(R2,p3) ],
                [np.dot(R3,p1),np.dot(R3,p2),np.dot(R3,p3)]])

        #step 6
        E = np.dot(R2,Rho2)

        #step 7
        A = 1/Do * (-D[0,1]* tau3/tau + D[1,1] + D[2,1]*tau1/tau)
        B = 1/6/Do* (D[0,1] *(tau3**2 - tau**2)*tau3/tau + D[2,1]*(tau**2 - tau1**2)*tau1/tau)


        a = -(A**2 + 2*A*E + (np.linalg.norm(R2))**2)
        b = -2 * mu * B * (A + E)
        c = -(mu*B)**2

        #step 8 and 9 - Calculate the roots of Equation x**8 + ax**6 + bx**3 + c = 0
        poly = [1, 0, a, 0, 0, b, 0, 0, c]
        roots = np.roots(poly)
        x = self.positive_root(roots, poly)

        #step 10
        f1 = 1 - 1/2*mu*tau1**2/(x**3)
        f3 = 1 - 1/2*mu*tau3**2/(x**3)
        g1 = tau1 - 1/6*mu*(tau1/x)**3
        g3 = tau3 - 1/6*mu*(tau3/x)**3

        rho2 = A + mu*B/x**3
        rho1 = 1/Do*((6*(D[2,0]*tau1/tau3 + D[1,0]*tau/tau3)*x**3 + mu*D[2,0]*(tau**2 - tau1**2)*tau1/tau3) /(6*x**3 + mu*(tau**2 - tau3**2)) - D[0,0])
        rho3 = 1/Do*((6*(D[0,2]*tau3/tau1 - D[1,2]*tau/tau1)*x**3 + mu*D[0,2]*(tau**2 - tau3**2)*tau3/tau1) /(6*x**3 + mu*(tau**2 - tau3**2)) - D[2,2])

        #step 11
        r1 = R1 + rho1*Rho1
        r2 = R2 + rho2*Rho2
        r3 = R3 + rho3*Rho3

        #step 12
        v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)

        self.orbit2 = Orbit(self.date2, list(r2)+list(v2), "cartesian",self.ECI_frame, None)

        keplerian = self.orbit2.copy(form = "keplerian")
        # print(repr(keplerian));exit()
        # print results
        print('Results of Gauss IOD [{}]:'.format(self.date2))
        print('r [km]: ',r2)
        print('v [km/s]: ',v2)
        print('Ω [deg]: ', np.rad2deg(keplerian.Ω))
        print('ω [deg]: ', np.rad2deg(keplerian.ω))
        print('inc [deg]: ', np.rad2deg(keplerian.i))
        print('a [km]: ',keplerian.a)
        print('e: ',keplerian.e)
        print('theta [deg]: ', np.rad2deg(keplerian.ν))

    def positive_root(self,roots, poly):
        """
        choose the positive roots in vector roots
        If there is more than one positive root, the user is
        prompted to select the one to use.
        """
        x = [np.real(y) for y in roots if np.isreal(y) and np.real(y) > 0]

        if len(x) == 0:
            raise Exception("There are no positive roots!!! ERROR")
            x =  None
        elif len(x) == 1:
            x = x[0]
        else:
            print("There are two or more positive valid roots")
            print("Possible solutions:")
            for (i, rt) in enumerate(x):
                print(str(i) + ": " + str(rt))
            sect = -1
            while(int(sect) < 0 or int(sect) >= len(roots)):
                sect = input("Press [number] with the selected solution to continue.")
            x = x[int(sect)]

        print("Root: ",x)
        return x

    def get_propagated_solution(self,epoch_final,force):
        """
        this function propagates the solution from the middle observation t2 to t4 (given as argument)
        """
        dyn = lambda t,x : fun(t,x,force)
        solver = integrate.solve_ivp(dyn, (self.date2.mjd*86400,epoch_final.mjd*86400), np.array(self.orbit2), method='RK45', t_eval=[epoch_final.mjd*86400])
        x = solver.y.flatten()
        return x
