import numpy as np

from .acceleration import Acceleration


class AtmosphericDrag(Acceleration):


    def acceleration(self, Rho, r, v, sat, rot_vector):
        """
        computes drag acceleration
        rho - @function to compute the density rho = rho(r)
        (r,v) satellite position and velocity vectors in ECI
        sat - Satellite instance
        rot_vector - rotational speed vector to convert between ECI and ECEF (omega vector)
        """
        rho = Rho(r)

        v_r = v - np.cross(rot_vector,r)
        v_abs = np.linalg.norm(v_r)
        a = -1/2 * rho * sat.CD * sat.a_drag / sat.m * v_abs * v_r

        return a
