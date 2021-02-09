import numpy as np
import scipy.fftpack as sfft
from numpy import sin,cos,arctan2,arcsin,pi,sqrt,tan

from beyond.orbits import Orbit
from beyond.constants import Earth as EARTH
from beyond.frames.iau1980 import _sideral

from orbidet.utils.diff_eqs import VOP_partials,_J2000_to_ITRF_rot,_PEF_to_ITRF,_TOD_to_PEF,_J2000_to_TOD,_TOD_to_PEF_date
from orbidet.cython_modules.semi_analytical import VOP_partials as VOP_partials_cy

mu = EARTH.mu
Re = EARTH.Re
J2 = EARTH.J2
w_earth = EARTH.rot_vector[2]

class NumAv_MEP_transf():
    def __init__(self,force,DFT_LEN=64,SIDEREAL_LEN=32): #128 10
        if force.CYTHON:
            self.VOP_sys_equinoctial = lambda x,T,_types: VOP_partials_cy(x,force,T,_types)
        else:
            self.VOP_sys_equinoctial = lambda x,T,_types: VOP_partials(x,force,T,_types)
        self.DFT_LEN=DFT_LEN
        self.SIDEREAL_LEN = SIDEREAL_LEN

        self._T = not force.no_rotations
        self._force = force

        if force.Grav_complete:
            self.tesserals = force.Grav_m >= 1
        else:
            self.tesserals = False
        drag = force.DRAG
        if drag:
            self.zonals_drag = ("zonals","drag")
        else:
            self.zonals_drag = ("zonals",)


            
    def getFourierCoeffs(self,orb):
        # initial setup
        N = self.DFT_LEN
        orb.form = 'equinoctial'
        a,h,k,p,q,lmb = orb
        n = np.sqrt(mu / a**3)

        if self.tesserals:
            M = self.SIDEREAL_LEN
            theta = np.deg2rad(_sideral(orb.date,model="apparent", eop_correction=False))
            theta_arr = np.arange(0,2*np.pi,2*np.pi/M) + [theta]*M

        # sample the GAUSS VOP perturbing EOMs
        lmb_arr = np.arange(0,2*np.pi,(2*np.pi)/N)

        # ZONALS
        orb_arr = []
        for lmb_k in lmb_arr:
            orbit = orb.copy() + np.array([0,0,0,0,0,lmb_k])
            orb_arr.append(orbit)

        if self._T:
            T = _J2000_to_ITRF_rot(orb.date) if not self._force.TOD_PEF_rot else _TOD_to_PEF_date(orb.date)
            f = lambda x: self.VOP_sys_equinoctial(x, T,self.zonals_drag)
        else:
            # no rotations
            f = lambda x: self.VOP_sys_equinoctial(x, None,self.zonals_drag)

        fs = np.array(list(map(f,orb_arr)))

        # take the DFTs and shift to center them at the fundamental freq. (k=0)
        F = sfft.fft(fs.T,n=N)
        F0 = F[0] ; F0 = sfft.fftshift(F0)
        F1 = F[1] ; F1 = sfft.fftshift(F1)
        F2 = F[2] ; F2 = sfft.fftshift(F2)
        F3 = F[3] ; F3 = sfft.fftshift(F3)
        F4 = F[4] ; F4 = sfft.fftshift(F4)
        F5 = F[5] ; F5 = sfft.fftshift(F5)

        # convert DFT to Fourier Coeffs.
        ind = -1
        for j in np.arange(-N/2,N/2):
            ind += 1
            F0[ind] = np.exp(-1j*j*lmb)*F0[ind] / N
            F1[ind] = np.exp(-1j*j*lmb)*F1[ind] / N
            F2[ind] = np.exp(-1j*j*lmb)*F2[ind] / N
            F3[ind] = np.exp(-1j*j*lmb)*F3[ind] / N
            F4[ind] = np.exp(-1j*j*lmb)*F4[ind] / N
            F5[ind] = np.exp(-1j*j*lmb)*F5[ind] / N
        dctFourier = {"zonals":[F0,F1,F2,F3,F4,F5]}

        if self.tesserals:
            # TESSERALS
            orb_arr = []
            for lmb_k in lmb_arr:
                aux = []
                for theta_k in theta_arr:
                    orbit = orb.copy() + np.array([0,0,0,0,0,lmb_k])
                    orbit.sideral = theta_k
                    aux.append(orbit)
                orb_arr.append(aux)

            if not self._force.TOD_PEF_rot:
                R_PEF_to_ITRF = _PEF_to_ITRF(orb.date)
                R_J2000_to_TOD = _J2000_to_TOD(orb.date)
                f = lambda x: self.VOP_sys_equinoctial(x, R_PEF_to_ITRF @ _TOD_to_PEF(x.sideral % (2*np.pi)) @ R_J2000_to_TOD,("tesserals",))
            else:
                f = lambda x: self.VOP_sys_equinoctial(x, _TOD_to_PEF(x.sideral % (2*np.pi)) ,("tesserals",))

            fs = np.array([list(map(f,x)) for x in orb_arr])
            F = sfft.fft2(fs.T)
            F0 = F[0].T ; F0 = sfft.fftshift(F0)
            F1 = F[1].T ; F1 = sfft.fftshift(F1)
            F2 = F[2].T ; F2 = sfft.fftshift(F2)
            F3 = F[3].T ; F3 = sfft.fftshift(F3)
            F4 = F[4].T ; F4 = sfft.fftshift(F4)
            F5 = F[5].T ; F5 = sfft.fftshift(F5)

            # convert DFT to Fourier Coeffs.
            ind_N = -1
            for j in np.arange(-N/2,N/2):
                ind_N += 1
                ind_M = -1
                for m in np.arange(-M/2,M/2):
                    ind_M += 1
                    F0[ind_N][ind_M] = np.exp(-1j*(j*lmb-m*theta)) * F0[ind_N][ind_M] / N / M
                    F1[ind_N][ind_M] = np.exp(-1j*(j*lmb-m*theta)) * F1[ind_N][ind_M] / N / M
                    F2[ind_N][ind_M] = np.exp(-1j*(j*lmb-m*theta)) * F2[ind_N][ind_M] / N / M
                    F3[ind_N][ind_M] = np.exp(-1j*(j*lmb-m*theta)) * F3[ind_N][ind_M] / N / M
                    F4[ind_N][ind_M] = np.exp(-1j*(j*lmb-m*theta)) * F4[ind_N][ind_M] / N / M
                    F5[ind_N][ind_M] = np.exp(-1j*(j*lmb-m*theta)) * F5[ind_N][ind_M] / N / M

            dctFourier["tesserals"] = [F0,F1,F2,F3,F4,F5]
        return dctFourier


    def osc_to_mean(self,orb):
        orb.form = 'equinoctial'

        a0_osc,a1_osc,a2_osc,a3_osc,a4_osc,a5_osc = orb
        a0_sum,a1_sum,a2_sum,a3_sum,a4_sum,a5_sum = self._transform(orb)

        a0_mean = a0_osc - np.real(a0_sum)
        a1_mean = a1_osc - np.real(a1_sum)
        a2_mean = a2_osc - np.real(a2_sum)
        a3_mean = a3_osc - np.real(a3_sum)
        a4_mean = a4_osc - np.real(a4_sum)
        a5_mean = a5_osc - np.real(a5_sum)

        orb =  Orbit(orb.date,[a0_mean,a1_mean,a2_mean,a3_mean,a4_mean,a5_mean],'equinoctial',orb.frame,None)
        return orb


    def mean_to_osc(self,orb):
        orb.form = 'equinoctial'
        a0_mean,a1_mean,a2_mean,a3_mean,a4_mean,a5_mean = orb
        a0_sum,a1_sum,a2_sum,a3_sum,a4_sum,a5_sum = self._transform(orb)


        a0_osc = a0_mean + np.real(a0_sum)
        a1_osc = a1_mean + np.real(a1_sum)
        a2_osc = a2_mean + np.real(a2_sum)
        a3_osc = a3_mean + np.real(a3_sum)
        a4_osc = a4_mean + np.real(a4_sum)
        a5_osc = a5_mean + np.real(a5_sum)

        orb =  Orbit(orb.date,[a0_osc,a1_osc,a2_osc,a3_osc,a4_osc,a5_osc],'equinoctial',orb.frame,None)
        return orb

    def _setup_transformation(self,orb):
        # initial setup
        N = self.DFT_LEN
        orb.form = 'equinoctial'
        a,h,k,p,q,lmb = orb
        n = np.sqrt(mu / a**3)

        if self.tesserals:
            M = self.SIDEREAL_LEN
            theta = np.deg2rad(_sideral(orb.date,model="apparent", eop_correction=False))
            theta_arr = np.arange(theta,theta+2*np.pi,2*np.pi/M)

        # sample the GAUSS VOP perturbing EOMs
        lmb_arr = np.arange(0,2*np.pi,(2*np.pi)/N)

        # zonals / drag transform
        a0,a1,a2,a3,a4,a5 = self._transform_zonals_drag(orb,lmb_arr,N,n,a)

        if self.tesserals:
            # tesserals transform
            _a0,_a1,_a2,_a3,_a4,_a5 = self._transform_tesserals_non_resonant(orb,lmb_arr,theta_arr,N,M,n,a)
            a0+=_a0 ; a1+=_a1 ; a2+=_a2 ; a3+=_a3 ; a4+=_a4 ; a5+=_a5

        return a0,a1,a2,a3,a4,a5


    def _transform_zonals_drag(self,orb,lmb_arr,N,n,a):

        # sample VOP
        orb_arr = []
        for lmb_k in lmb_arr:
            orbit = orb.copy() + np.array([0,0,0,0,0,lmb_k])
            orb_arr.append(orbit)

        if self._T:
            T = _J2000_to_ITRF_rot(orb.date) if not self._force.TOD_PEF_rot else _TOD_to_PEF_date(orb.date)
            f = lambda x: self.VOP_sys_equinoctial(x, T,self.zonals_drag)
        else:
            # no rotations
            f = lambda x: self.VOP_sys_equinoctial(x, None,self.zonals_drag)

        fs = np.array(list(map(f,orb_arr)))

        # take the DFTs and shift to center them at the fundamental freq. (k=0)
        F = sfft.fft(fs.T,n=N)

        F0 = F[0] ; F0 = sfft.fftshift(F0)
        F1 = F[1] ; F1 = sfft.fftshift(F1)
        F2 = F[2] ; F2 = sfft.fftshift(F2)
        F3 = F[3] ; F3 = sfft.fftshift(F3)
        F4 = F[4] ; F4 = sfft.fftshift(F4)
        F5 = F[5] ; F5 = sfft.fftshift(F5)

        a0_sum = a1_sum = a2_sum = a3_sum = a4_sum = a5_sum = 0
        ind = -1
        for i in np.arange(-N/2,N/2):
            ind += 1
            if i == 0:
                continue
            a0_sum += F0[ind] / (i*1j)
            a1_sum += F1[ind] / (i*1j)
            a2_sum += F2[ind] / (i*1j)
            a3_sum += F3[ind] / (i*1j)
            a4_sum += F4[ind] / (i*1j)
            a5_sum += F5[ind] / (i*1j) + 3/2/a*F0[ind]/i**2
        cte = 1/n/N
        # print(a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte);exit()
        return a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte


    def _transform_tesserals_non_resonant(self,orb,lmb_arr,theta_arr,N,M,n,a):
        # sample VOP
        orb_arr = []

        for lmb_k in lmb_arr:
            aux = []
            for theta_k in theta_arr:
                orbit = orb.copy() + np.array([0,0,0,0,0,lmb_k])
                orbit.sideral = theta_k
                aux.append(orbit)
            orb_arr.append(aux)

        if not self._force.TOD_PEF_rot:
            R_PEF_to_ITRF = _PEF_to_ITRF(orb.date)
            R_J2000_to_TOD = _J2000_to_TOD(orb.date)
            f = lambda x: self.VOP_sys_equinoctial(x, R_PEF_to_ITRF @ _TOD_to_PEF(x.sideral % (2*np.pi)) @ R_J2000_to_TOD,("tesserals",))
        else:
            f = lambda x: self.VOP_sys_equinoctial(x, _TOD_to_PEF(x.sideral % (2*np.pi)) ,("tesserals",))

        fs = np.array([list(map(f,x)) for x in orb_arr])
        F = sfft.fft2(fs.T)
        F0 = F[0].T ; F0 = sfft.fftshift(F0)
        F1 = F[1].T ; F1 = sfft.fftshift(F1)
        F2 = F[2].T ; F2 = sfft.fftshift(F2)
        F3 = F[3].T ; F3 = sfft.fftshift(F3)
        F4 = F[4].T ; F4 = sfft.fftshift(F4)
        F5 = F[5].T ; F5 = sfft.fftshift(F5)


        a0_sum = a1_sum = a2_sum = a3_sum = a4_sum = a5_sum = 0
        ind_N = -1
        for i in np.arange(-N/2,N/2):
            ind_N += 1
            ind_M = -1
            if i == 0:
                continue
            for k in np.arange(-M/2,M/2):
                ind_M += 1
                if k == 0:
                    continue
                a0_sum += F0[ind_N][ind_M] / (1j * (i*n - k*w_earth) )
                a1_sum += F1[ind_N][ind_M] / (1j * (i*n - k*w_earth) )
                a2_sum += F2[ind_N][ind_M] / (1j * (i*n - k*w_earth) )
                a3_sum += F3[ind_N][ind_M] / (1j * (i*n - k*w_earth) )
                a4_sum += F4[ind_N][ind_M] / (1j * (i*n - k*w_earth) )
                a5_sum += F5[ind_N][ind_M] / (1j * (i*n - k*w_earth) ) + 3*n/2/a*F0[ind_N][ind_M]/(i*n - k*w_earth)**2
        cte = 1/N/M
        # print(a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte);exit()
        return a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte





    def reconstructOrbfrom(self,dctFouriers,orb):
        lmb = orb.lmb# % (2*np.pi)
        a = orb.a
        n = np.sqrt(mu / a**3)
        N = self.DFT_LEN
        M = self.SIDEREAL_LEN

        if "zonals" in dctFouriers:
            aux = dctFouriers["zonals"]
            F0 = aux[0] ; F1 = aux[1] ; F2 = aux[2]
            F3 = aux[3] ; F4 = aux[4] ; F5 = aux[5]
            a0_sum = a1_sum = a2_sum = a3_sum = a4_sum = a5_sum = 0
            ind = -1
            for i in np.arange(-N/2,N/2):
                ind += 1
                if i == 0:
                    continue
                a0_sum += (F0[ind] * N * np.exp(1j*i*lmb)) / (i*1j)
                a1_sum += (F1[ind] * N * np.exp(1j*i*lmb)) / (i*1j)
                a2_sum += (F2[ind] * N * np.exp(1j*i*lmb)) / (i*1j)
                a3_sum += (F3[ind] * N * np.exp(1j*i*lmb)) / (i*1j)
                a4_sum += (F4[ind] * N * np.exp(1j*i*lmb)) / (i*1j)
                a5_sum += (F5[ind] * N * np.exp(1j*i*lmb)) / (i*1j) + 3/2/a*(F0[ind] * N * np.exp(1j*i*lmb))/i**2
            cte = 1/n/N
            orb_sum = np.apply_along_axis(np.real,0,
                                          [a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte])
            orb += orb_sum


        if "tesserals" in dctFouriers:
            aux = dctFouriers["tesserals"]
            F0 = aux[0] ; F1 = aux[1] ; F2 = aux[2]
            F3 = aux[3] ; F4 = aux[4] ; F5 = aux[5]
            theta = np.deg2rad(_sideral(orb.date,model="apparent", eop_correction=False))
            a0_sum = a1_sum = a2_sum = a3_sum = a4_sum = a5_sum = 0
            ind_N = -1

            for i in np.arange(-N/2,N/2):
                ind_N += 1
                ind_M = -1
                if i == 0:
                    continue
                for k in np.arange(-M/2,M/2):
                    ind_M += 1
                    if k == 0:
                        continue
                    a0_sum += (F0[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta))) / (1j * (i*n - k*w_earth) )
                    a1_sum += (F1[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta))) / (1j * (i*n - k*w_earth) )
                    a2_sum += (F2[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta))) / (1j * (i*n - k*w_earth) )
                    a3_sum += (F3[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta))) / (1j * (i*n - k*w_earth) )
                    a4_sum += (F4[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta))) / (1j * (i*n - k*w_earth) )
                    a5_sum += (F5[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta))) / (1j * (i*n - k*w_earth) ) + 3*n/2/a*(
                        F0[ind_N][ind_M] * N*M*np.exp(1j*(i*lmb-k*theta)))/(i*n - k*w_earth)**2
            cte = 1/N/M
            orb_sum = np.apply_along_axis(np.real,0,
                                          [a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte])
            orb += orb_sum
        return orb









class Brouwer_transf:

    def __init__(self):
        pass

    def osc2mean(self,_orb):
        """
        computes osculating orbital elements from the respective mean
        using a brouwer first order map / method
        [First-Order Mapping between Mean and Osculating Orbit Elements]
        """
        # unpacking orbit info
        orbit = _orb.copy(form="keplerian_mean")
        a = orbit.a;e = orbit.e;i = orbit.i;RAAN = orbit.Ω;w = orbit.ω;M = orbit.M
        orbit.form="keplerian"
        f = orbit.ν

        #initial checks
        if e >= 1:
            raise Exception("--ERROR--\nThe orbit is not elliptic (e >= 1)")

        # transform
        gama2 = -J2 / 2 * (Re / a)**2
        a_mean,e_mean,i_mean,RAAN_mean,w_mean,M_mean = self._transform(a,e,i,RAAN,w,M,f,gama2)

        return Orbit(orbit.date,[a_mean,e_mean,i_mean,RAAN_mean,w_mean,M_mean],
                     "keplerian_mean",orbit.frame,None)

    def mean2osc(self,orbit):
        """
        computes mean orbital elements from the respective osculating
        using a brouwer first order map / method
        [First-Order Mapping between Mean and Osculating Orbit Elements]
        """
        # unpacking orbit info
        orbit = _orb.copy(form="keplerian_mean")
        a = orbit.a;e = orbit.e;i = orbit.i;RAAN = orbit.Ω;w = orbit.ω;M = orbit.M
        orbit.form="keplerian"
        f = orbit.ν

        #initial checks
        if e >= 1:
            raise Exception("--ERROR--\nThe orbit is not elliptic (e >= 1)")

        # transform
        gama2 = J2 / 2 * (Re / a)**2
        a_osc,e_osc,i_osc,RAAN_osc,w_osc,M_osc = self._transform(a,e,i,RAAN,w,M,f,gama2)

        return Orbit(orbit.date,[a_osc,e_osc,i_osc,RAAN_osc,w_osc,M_osc],
                     "keplerian_mean",orbit.frame,None)

    def _transform(self,a,e,i,RAAN,w,M,f,gama2):

        eta = sqrt(1 - e**2)
        gama2_plica = gama2 / eta**4 #gama2'


        a_over_r = (1 + e*cos(f)) / (eta**2)

        a_mean = a + a*gama2*((3*cos(i)**2 - 1) * ((a_over_r)**3 - 1 / eta**3) +
                              3*(1 - cos(i)**2)*a_over_r**3 * cos(2*w + 2*f) )

        #intermediate delta results
        delta1 = gama2_plica/8 * e * eta**2 * (1 - 11*cos(i)**2 - 40 * cos(i)**4 / (1 - 5*cos(i)**2)) * cos(2*w)

        delta_e = delta1 + eta**2/2 * (gama2*((3*cos(i)**2 - 1)/eta**6 * (e * eta + e/(1+eta) +3*cos(f) +
                 3*e*cos(f)**2 + e**2*cos(f)**3) + 3*(1-cos(i)**2)/eta**6*
                                             (e + 3*cos(f) + 3*e*cos(f)**2 + e**2*cos(f)**3)*cos(2*w+2*f)) -
                  gama2_plica*(1-cos(i)**2)*(3*cos(2*w+f) + cos(2*w+3*f)))

        delta_i = -e*delta1 / eta**2 / tan(i) + gama2_plica/2*cos(i)*sqrt(1-cos(i)**2)*(
            3*cos(2*w+2*f) + 3*e*cos(2*w+f) + e*cos(2*w+3*f))

        M_w_RAAN_plica = M + w + RAAN + gama2_plica/8*eta**3*(1 - 11*cos(i)**2 - 40*cos(i)**4/(1-5*cos(i)**2)) - gama2_plica/16*(
            2 + e**2 - 11*(2+3*e**2)*cos(i)**2 - 40*(2+5*e**2)*cos(i)**4/(1-5*cos(i)**2) - 400*e**2*cos(i)**6/((1-5*cos(i)**2)**2)) +gama2_plica/4*(
                -6*(1-5*cos(i)**2)*(f-M+e*sin(f)) +(3-5*cos(i)**2)*(3*sin(2*w+2*f) + 3*e*sin(2*w+f) + e*sin(2*w+3*f))) - gama2_plica/8*e**2*cos(i)*(
                    11 + 80*cos(i)**2/(1-5*cos(i)**2) + 200*cos(i)**4 / ((1-5*cos(i)**2)**2)) - gama2_plica/2*cos(i)*(
                        6*(f-M+e*sin(f)) - 3*sin(2*w+2*f) - 3*e*sin(2*w+f) - e*sin(2*w+3*f))


        e_delta_M = gama2_plica/8*eta**3*e*(1 - 11*cos(i)**2 - 40*cos(i)**4/(1-5*cos(i)**2)) - gama2_plica/4*eta**3*(2*(3*cos(i)**2-1)*(
            a_over_r**2*eta**2 + a_over_r +1)*sin(f) + 3*(1-cos(i)**2)* ((-a_over_r**2*eta**2 - a_over_r + 1)*sin(2*w+f) + (
                a_over_r**2*eta**2 + a_over_r + 1/3)*sin(2*w+3*f)))

        delta_RAAN = -gama2_plica/8*e**2*cos(i)*(11 + 80*cos(i)**2/(1-5*cos(i)**2) + 200*cos(i)**4/((1-5*cos(i)**2)**2)
            ) - gama2_plica/2*cos(i)*(6*(f-M+e*sin(f)) - 3*sin(2*w+2*f) -3*e*sin(2*w+f) - e*sin(2*w+3*f))


        d1 = (e + delta_e)*sin(M) + e_delta_M*cos(M)
        d2 = (e + delta_e)*cos(M) - e_delta_M*sin(M)

        M_mean = arctan2(d1,d2) % (2*pi)
        e_mean = sqrt(d1**2 + d2**2)

        d3 = (sin(i/2) + cos(i/2)*delta_i/2) * sin(RAAN) + sin(i/2)*delta_RAAN*cos(RAAN)
        d4 = (sin(i/2) + cos(i/2)*delta_i/2) * cos(RAAN) - sin(i/2)*delta_RAAN*sin(RAAN)

        RAAN_mean = arctan2(d3,d4) % (2*pi)
        i_mean = 2*arcsin(sqrt(d3**2+d4**2))
        w_mean = (M_w_RAAN_plica - M_mean - RAAN_mean) % (2*pi)

        return a_mean,e_mean,i_mean,RAAN_mean,w_mean,M_mean



def lixo():
        # f0 = np.array([[x[0] for x in y] for y in fs])
        # f1 = np.array([[x[1] for x in y] for y in fs])
        # f2 = np.array([[x[2] for x in y] for y in fs])
        # f3 = np.array([[x[3] for x in y] for y in fs])
        # f4 = np.array([[x[4] for x in y] for y in fs])
        # f5 = np.array([[x[5] for x in y] for y in fs])
# F0 = sfft.fft2(f0) ; F0 = sfft.fftshift(F0)
        # F1 = sfft.fft2(f1) ; F1 = sfft.fftshift(F1)
        # F2 = sfft.fft2(f2) ; F2 = sfft.fftshift(F2)
        # F3 = sfft.fft2(f3) ; F3 = sfft.fftshift(F3)
        # F4 = sfft.fft2(f4) ; F4 = sfft.fftshift(F4)
        # F5 = sfft.fft2(f5) ; F5 = sfft.fftshift(F5)

    # OLD VERSION
    # def _transform_old(self,orb):
    #     N = self.DFT_LEN
    #
    #     # initial definitions
    #     orb.form = 'equinoctial'
    #     a,h,k,p,q,lmb = orb
    #     n = np.sqrt(mu / a**3)
    #     if self.tesserals:
    #         # additional setup to include tesserals
    #         orb_copy = orb.copy(form="keplerian_mean")
    #         RAAN = orb_copy.raan
    #         theta = np.deg2rad(_sideral(orb.date,model="apparent", eop_correction=False))
    #         stroboscopic_node = self.resonance_P*(lmb - RAAN) - self.resonance_Q*(theta - RAAN)
    #
    #     # sample the GAUSS VOP perturbing EOMs
    #     lmb_arr = np.arange(0,2*np.pi*self.resonance_Q,(2*np.pi*self.resonance_Q)/N)
    #     orb_arr = []
    #
    #     for lmb_k in lmb_arr:
    #         orbit = orb.copy() + np.array([0,0,0,0,0,lmb_k])
    #         if self.tesserals:
    #             orbit.sideral = (self.resonance_P * (orbit.lmb+lmb_k-RAAN) - stroboscopic_node)/self.resonance_Q + RAAN
    #         orb_arr.append(orbit)
    #
    #     if self._T:
    #         if self.tesserals:
    #             R_PEF_to_ITRF = _PEF_to_ITRF(orb.date)
    #             R_J2000_to_TOD = _J2000_to_TOD(orb.date)
    #             f = lambda x: self.VOP_sys_equinoctial(x, R_PEF_to_ITRF @ _TOD_to_PEF(x.sideral) @ R_J2000_to_TOD)
    #         else:
    #             # zonals
    #             T = _J2000_to_ITRF_rot(orb.date)
    #             f = lambda x: self.VOP_sys_equinoctial(x, T)
    #     else:
    #         f = lambda x: self.VOP_sys_equinoctial(x, None)
    #
    #     fs = np.array(list(map(f,orb_arr)))
    #     # f0 = fs[0:,0]
    #     # f1 = fs[0:,1]
    #     # f2 = fs[0:,2]
    #     # f3 = fs[0:,3]
    #     # f4 = fs[0:,4]
    #     # f5 = fs[0:,5]
    #
    #
    #     # take the DFTs and shift to center them at the fundamental freq. (k=0)
    #     # F0 = np.fft.fft(f0,n=N) ; F0 = np.fft.fftshift(F0)
    #     # F1 = np.fft.fft(f1,n=N) ; F1 = np.fft.fftshift(F1)
    #     # F2 = np.fft.fft(f2,n=N) ; F2 = np.fft.fftshift(F2)
    #     # F3 = np.fft.fft(f3,n=N) ; F3 = np.fft.fftshift(F3)
    #     # F4 = np.fft.fft(f4,n=N) ; F4 = np.fft.fftshift(F4)
    #     # F5 = np.fft.fft(f5,n=N) ; F5 = np.fft.fftshift(F5)
    #     F = np.fft.fft(fs.T,n=N) #; F = np.fft.fftshift(F)
    #     F0 = F[0] ; F0 = np.fft.fftshift(F0)
    #     F1 = F[1] ; F1 = np.fft.fftshift(F1)
    #     F2 = F[2] ; F2 = np.fft.fftshift(F2)
    #     F3 = F[3] ; F3 = np.fft.fftshift(F3)
    #     F4 = F[4] ; F4 = np.fft.fftshift(F4)
    #     F5 = F[5] ; F5 = np.fft.fftshift(F5)
    #
    #     a0_sum = a1_sum = a2_sum = a3_sum = a4_sum = a5_sum = 0
    #     ind = -1
    #     for i in np.arange(-N/2,N/2):
    #         ind += 1
    #         if i == 0:
    #             continue
    #         a0_sum += F0[ind] / (i*1j)
    #         a1_sum += F1[ind] / (i*1j)
    #         a2_sum += F2[ind] / (i*1j)
    #         a3_sum += F3[ind] / (i*1j)
    #         a4_sum += F4[ind] / (i*1j)
    #         a5_sum += F5[ind] / (i*1j) + 3*self.resonance_Q/2/a*F0[ind]/i**2
    #     cte = self.resonance_Q/n/N
    #     return a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte
    #

    # def _transform_tesserals_resonant(self,orb):
    #     N = self.DFT_LEN
    #
    #     # initial definitions
    #     orb.form = 'equinoctial'
    #     a,h,k,p,q,lmb = orb
    #     n = np.sqrt(mu / a**3)
    #
    #     orb_copy = orb.copy(form="keplerian_mean")
    #     RAAN = orb_copy.raan
    #     theta = np.deg2rad(_sideral(orb.date,model="apparent", eop_correction=False))
    #     stroboscopic_node = self.resonance_P*(lmb - RAAN) - self.resonance_Q*(theta - RAAN)
    #
    #     # sample the GAUSS VOP perturbing EOMs
    #     lmb_arr = np.arange(0,2*np.pi*self.resonance_Q,(2*np.pi*self.resonance_Q)/N)
    #     orb_arr = []
    #
    #     for lmb_k in lmb_arr:
    #         orbit = orb.copy() + np.array([0,0,0,0,0,lmb_k])
    #         orbit.sideral = (self.resonance_P * (orbit.lmb-RAAN) - stroboscopic_node)/self.resonance_Q + RAAN
    #         orb_arr.append(orbit)
    #
    #     R_PEF_to_ITRF = _PEF_to_ITRF(orb.date)
    #     R_J2000_to_TOD = _J2000_to_TOD(orb.date)
    #     f = lambda x: self.VOP_sys_equinoctial(x, R_PEF_to_ITRF @ _TOD_to_PEF(x.sideral) @ R_J2000_to_TOD)
    #
    #     fs = np.array(list(map(f,orb_arr)))
    #     # take the DFTs and shift to center them at the fundamental freq. (k=0)
    #     F = np.fft.fft(fs.T,n=N)
    #     F0 = F[0] ; F0 = np.fft.fftshift(F0)
    #     F1 = F[1] ; F1 = np.fft.fftshift(F1)
    #     F2 = F[2] ; F2 = np.fft.fftshift(F2)
    #     F3 = F[3] ; F3 = np.fft.fftshift(F3)
    #     F4 = F[4] ; F4 = np.fft.fftshift(F4)
    #     F5 = F[5] ; F5 = np.fft.fftshift(F5)
    #
    #     a0_sum = a1_sum = a2_sum = a3_sum = a4_sum = a5_sum = 0
    #     ind = -1
    #     for i in np.arange(-N/2,N/2):
    #         ind += 1
    #         if i == 0:
    #             continue
    #         a0_sum += F0[ind] / (i*1j)
    #         a1_sum += F1[ind] / (i*1j)
    #         a2_sum += F2[ind] / (i*1j)
    #         a3_sum += F3[ind] / (i*1j)
    #         a4_sum += F4[ind] / (i*1j)
    #         a5_sum += F5[ind] / (i*1j) + 3*self.resonance_Q/2/a*F0[ind]/i**2
    #     cte = self.resonance_Q/n/N
    #     return a0_sum*cte,a1_sum*cte,a2_sum*cte,a3_sum*cte,a4_sum*cte,a5_sum*cte
    pass
