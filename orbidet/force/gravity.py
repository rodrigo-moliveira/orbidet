import numpy as np
from pathlib import Path
from math import factorial,sqrt,sin,cos,asin,atan2,tan


from .acceleration import Acceleration
from orbidet.errors import GravityError
from beyond.beyond.constants import Earth

mu = Earth.mu
Re = Earth.equatorial_radius
J2 = Earth.J2


class TwoBody(Acceleration):
    """
    Simple two-body acceleration
    """
    def acceleration(self,r):
        R = np.linalg.norm(r)
        a = -mu * r / (R**3)
        return a


class LowZonalHarmonics(Acceleration):
    """Hardcoded low degree zonal harmonics
    """
    def __init__(self,degree):
        self.degree = degree

    def acceleration(self,r):
        """
        compute acceleration due to zonal harmonics.
        The harmonics coded are the low-order ones (the equations are hard-coded) and
        the general harmonic series is not applied
        R - np array with position
        r - np.linalg.norm(R)
        n - order of the zonal harmonic. from n=2 (J2) to n=6(J2,J3,J4,J5,J6)
        """
        R = np.linalg.norm(r)
        x,y,z = r
        ax=ay=az=0
        r_2 = R*R

        if self.degree >= 2:#J2
            aux = -3 * J2 * mu * Re**2 / (2 * R **5)
            aux1 = aux*(1 - 5*z**2 / r_2)*x
            ax += aux1
            ay += aux1*y/x
            az += aux*(3 - 5*z**2 / r_2)*z

        if self.degree >= 3: #J3
            aux = -5*Earth.J3*mu*Re**3/(2*R**7)
            aux1 = aux * (3*z - 7*z**3/r_2)*x
            ax += aux1
            ay += aux1*y/x
            az += aux * (6*z**2 - 7*z**4/r_2 - 3*r_2/5)

        if self.degree >= 4: #J4
            r_4 = r_2 * r_2
            aux = 15*Earth.J4*mu*Re**4/(8*R**7)
            aux1 = aux*(1-14*z**2/r_2+21*z**4/r_4)*x
            ax += aux1
            ay += aux1*y/x
            az += aux * (5 - 70*z**2/(3*r_2) + 21*z**4/r_4)*z

        if self.degree >= 5: #J5
            aux=3*Earth.J5*mu*Re**5/(8*R**9)
            aux1 = aux * x*y * (35-210*z**2/r_2 + 231*z**4/r_4)
            ax += aux1; ay += aux1
            az += aux * z**2 * (105 - 315*z**2/r_2 + 231*z**4/r_4) + 15*Earth.J5*mu*Re**5/(8*R**7)

        if self.degree >= 6: #J6
            aux = -Earth.J6*mu*Re**6/(16*R**9)
            aux1 = aux*x*(35 - 945*z**2/r_2 + 3465*z**4/r_4 - 3003*z**6/R**6)
            ax += aux1; ay += aux1*y/x
            az += aux * (245 - 2205*z**2/r_2 + 4851*z**4/r_4 - 3003*z**6/R**6)

        return [ax,ay,az]


def _normalize_P(n,m):
    """Normalization constant of Coefficients (ALFs)"""
    k = 1 if m == 0 else 2
    return sqrt(  (factorial(n-m) * k * (2*n + 1)) / factorial(n+m) )


class GravityAcceleration(Acceleration):
    DEFAULT_PATH = Path("orbidet/data/EGM96.txt")

    def __init__(self,Degree,Order,FileToPotentialModel=DEFAULT_PATH):
        if Degree < 0 or Order < 0 or Order > Degree:
            raise GravityError("Error setting the Gravity Acceleration. Degree and Order must be > 0 and "
            "Degree >= Order")
        self.degree = Degree
        self.order = Order

        self._C,self._S = self.read_Snm_Cnm(FileToPotentialModel,Degree,Order)



    def read_Snm_Cnm(self,filename,N,M):
        #initialize data
        C = [[0] * (n+1) for n in range(N+1)]
        S = [[0] * (n+1) for n in range(N+1)]

        with open(filename,'r') as f:
            n=2;m=0
            for line in f:
                words = line.split()
                C[int(words[0])][int(words[1])] = float(words[2])
                S[int(words[0])][int(words[1])] = float(words[3])

                if(int(words[0]) == N and int(words[1]) == N):
                    break
        # zeroth order coeff.
        C[0][0] = 1
        return (C,S)



    def Legendre_coeff(self,N,M,phi):
        """
        computes the Legendre normalized coefficients for the function sin(phi), phi is the lattitude
        """

        _P = [[0] * (n+1) for n in range(N+1)]
        cos_phi = cos(phi)
        sin_phi = sin(phi)
        _P[0][0] = 1
        _P[1][0] = sin_phi *  _normalize_P(1,0)
        _P[1][1] = cos_phi *  _normalize_P(1,1)

        #recursive alg
        for n in range(2,N+1):
            #diagonal:
            _P[n][n] = (2*n-1)*cos_phi*_P[n-1][n-1]*_normalize_P(n,n)/_normalize_P(n-1,n-1)

            #horizontal first step coefficients
            _P[n][0] = ((2*n-1) * sin_phi * _P[n-1][0]*_normalize_P(n,0)/_normalize_P(
            n-1,0) - (n-1)*_P[n-2][0]*_normalize_P(n,0)/_normalize_P(n-2,0)) / n

            #horizontal second step coefficients
            for m in range(1,n):
                if m > n - 2:
                    P_aux = 0
                else:
                    # print(n,m)
                    P_aux = _P[n-2][m]*_normalize_P(n,m)/_normalize_P(n-2,m)
                _P[n][m] = P_aux + (2*n-1)*cos_phi*_P[n-1][m-1]*_normalize_P(n,m)/_normalize_P(n-1,m-1)

        return _P


    def acceleration(self,R,n_start=2,m_start=0):
        """
        computes the gravitational startint at n_start and m_start
        R is the position vector in the evaluation frame (should be ECEF)
        The calculations are made in the ECEF frame and then uppon return transformed back to ECI
        n_start -> initial n value to start the summation (default is 2: excludes the central force term
                (1st order terms all vanish))
        """
        r = np.linalg.norm(R)

        # get spherical coordinates
        latgc = asin(R[2]/r)
        lon = atan2(R[1],R[0])

        # Legendre coefficients
        P = self.Legendre_coeff(self.degree+2,self.degree+2,latgc)

        #partial potential derivatives (equation 7.21 VALLADO)
        dU_dr = dU_dlat = dU_dlon = 0
        q2 = q1 = q0 = 0

        for n in range(n_start,self.degree+1):

            b0 = (-mu / r**2) * (Re / r)**n * (n+1)
            b1 = b2 = (mu / r) * (Re / r)**n

            for m in range(m_start,n+1):
                if m > self.order: break
                aux = self._C[n][m]*cos(m*lon) + self._S[n][m]*sin(m*lon)
                N0 = _normalize_P(n,m)

                try:
                    N1 = _normalize_P(n,m+1)
                    y = P[n][m+1]
                except:
                    N1=1
                    y = 0
                q0 += P[n][m] * aux
                q1 += (y*N0/N1 - (m)*tan(latgc) * P[n][m]) * aux
                q2 += m * P[n][m] * (self._S[n][m]*cos(m*lon) - self._C[n][m]*sin(m*lon))

            dU_dr += q0*b0
            dU_dlat += q1*b1
            dU_dlon += q2*b2
            q2 = q1 = q0 = 0

        #ECEF acceleration components (equation 7.23 VALLADO)
        r2xy = R[0]**2 + R[1]**2
        aux_0 = (1/r*dU_dr-R[2]/(r**2*sqrt(r2xy))*dU_dlat)
        aux_1 = (1/r2xy*dU_dlon)

        ax = aux_0*R[0] - aux_1*R[1]
        ay = aux_0*R[1] + aux_1*R[0]
        az =  1/r*dU_dr*R[2]+sqrt(r2xy)/r**2*dU_dlat
        return [ax,ay,az]
