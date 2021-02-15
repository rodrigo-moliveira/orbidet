"""utility functions for the estimators
"""
import numpy as np

from orbidet.propagators.utils import cartesian_osculating_state_fct

from beyond.beyond.orbits import Orbit
from beyond.beyond.constants import Earth

mu = Earth.mu
J2 = Earth.J2
Re = Earth.r


############ Jacobians of State Functions (cartesian and equinoctial) ############
def jacobian_cartesian_osculating_fct(X):
    """
    evaluates the jacobian of state function
    A simplistic approach considering only an hard-coded J2 harmonic is used for simplicity
    """
    x,y,z,vx,vy,vz = X[0:6]

    r = np.sqrt(x**2+y**2+z**2)

    #two-body dynamics hardcoded
    grad = np.array([[0,0,0,1,0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1],
                    [mu*3*x**2/r**5-mu/r**3, mu*3*x*y/r**5, mu*3*x*z/r**5,0,0,0],
                    [mu*3*x*y/r**5, mu*3*y**2/r**5-mu/r**3, mu*3*y*z/r**5,0,0,0],
                    [mu*3*x*z/r**5, mu*3*y*z/r**5, mu*3*z**2/r**5-mu/r**3,0,0,0]])


    #J2
    aux = -3*J2*mu*Re**2 / (2*r**5)
    ax = aux*(1 - 5*z**2 / r**2)*x

    dax_dx = aux * (1/r**5 - 5*x**2/r**7 - 5*z**2/r**7 + 35*z**2*x**2/r**9)
    dax_dy = aux * (-5*x*y/r**7 + 35*z**2*x*y/r**9)
    dax_dz = aux * (-5*x*z/r**7 + 35*z**3*x/r**9)

    day_dx = dax_dy
    day_dy = y/x * dax_dy + ax/x
    day_dz = y/x * dax_dz

    daz_dx = aux * (-15*x*z/r**7 + 35*z**3*x/r**9)
    daz_dy = aux * (-15*y*z/r**7 + 35*z**3*y/r**9)
    daz_dz = aux * (3/r**5 - 30*z**2/r**7 + 35*z**4/r**9)

    grad_oblateness = np.array([[0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [dax_dx,dax_dy,dax_dz,0,0,0],
                            [day_dx,day_dy,day_dz,0,0,0],
                            [daz_dx,daz_dy,daz_dz,0,0,0]])
    grad += grad_oblateness

    return grad


def first_order_state_cov_differential_equation_cartesian(t,Y,force):
    """
    numerical integrates the state and covariance over time (used in EKF and LS)
    """
    #unpack the variables
    Y_matrix = np.reshape(Y,(7,6))
    y = Y_matrix[0]
    phi = Y_matrix[1:]

    #apply differential functions
    dydt = cartesian_osculating_state_fct(t,y,force) #derivative of position and velocity
    F = jacobian_cartesian_osculating_fct(y)
    dphidt = F @ phi

    #pack the variables
    return np.append(dydt,dphidt)






############ Finite Differencing ############

def finite_differencing(f,x,t,epsons):
    """
    compute central finite differencing df/dx at x
    """
    N = len(x)
    df_dx = []
    for i in range(len(x)):
        _eps = epsons[i]
        eps = np.zeros(N)
        eps[i] = _eps
        df_dx.append((f(t,x+eps) - f(t,x-eps))/2/_eps)
    return np.array(df_dx).T





def MeanEqCov_to_cartesian(P_eq,dx_deq_jacbn,B1=None):
    """convert a covariance matrix in mean equinoctial state to osculating cartesian state
    P_x = G @ P_eq @ G.T
    where:
        G = dx/d_eq_osc @ d_eq_osc/d_eq_mean = dx/d_eq_osc ( I + B1 )
        If B1 is None, then a map between osculating states is performed instead
    """
    n = len(P_eq[0])
    if B1 is None:
        B1 = np.zeros((n,n))
    G = dx_deq_jacbn @ (np.eye(n) + B1)
    Px = G @ P_eq @ G.T
    return Px

def CartesianCov_to_MeanEq(Px,dx_deq_jacbn,B1=None):
    """invertion of MeanEqCov_to_cartesian function, provided that G is invertible
    """
    n = len(Px[0])
    if B1 is None:
        B1 = np.zeros((n,n))
    G = np.linalg.inv(dx_deq_jacbn @ (np.eye(n) + B1))
    Peq = G @ Px @ G.T
    return Peq


def obs_jacobian_wrt_equinoctial(t,x,H,frame):
    orb = Orbit(x,t,"equinoctial_mean",frame,None)
    Dosc_Dequict = partials_cartesian_wrt_equinocital(orb)
    orb.form = "cartesian"
    grad_h = H(orb)
    return grad_h @ Dosc_Dequict


def partials_cartesian_wrt_equinocital(orb):
    """
    pg 9 of  [SemiAnalytic Satellite Theory]
        (Partial Derivatives of Position and Velocity with Respect
        to the Equinoctial Elements)
    """
    orb.form = "equinoctial_mean"
    a,h,k,p,q,lmb = orb
    orb.form = "cartesian"
    r,v = np.array(orb[0:3]),np.array(orb[3:])
    _r = np.linalg.norm(r)
    n = np.sqrt(mu/a**3)

    A = np.sqrt(mu*a)
    B = np.sqrt(1-h**2-k**2)
    C = 1 + p**2 + q**2

    f = 1/C*np.array([1-p**2+q**2,2*p*q,-2*p])
    g = 1/C*np.array([2*p*q,1+p**2-q**2,2*q])
    w = 1/C*np.array([2*p,-2*q,1-p**2-q**2])

    X = np.dot(r,f) ; Y = np.dot(r,g)
    Xd = np.dot(v,f) ; Yd = np.dot(v,g)

    dX_dh = -k*Xd/(n*(1+B)) + a*Y*Yd/A/B
    dY_dh = -k*Yd/(n*(1+B)) - a*X*Yd/A/B - a
    dX_dk = h*Xd/(n*(1+B)) + a*Y*Xd/A/B - a
    dY_dk = h*Yd/(n*(1+B)) - a*X*Xd/A/B

    dXd_dh = a*Yd**2/A/B + A/_r**3*(a*k*X/(1+B)-Y**2/B)
    dYd_dh = -a*Xd*Yd/A/B + A/_r**3*(a*k*Y/(1+B) + X*Y/B)
    dXd_dk = a*Xd*Yd/A/B - A/_r**3*(a*h*X/(1+B)+X*Y/B)
    dYd_dk = -a*Xd**2/A/B - A/_r**3*(a*h*Y/(1+B)-X**2/B)

    # position derivatives
    dr_da = r / a
    dr_dh = dX_dh*f + dY_dh*g
    dr_dk = dX_dk*f + dY_dk*g
    dr_dp = 2*(q * (Y*f - X*g) - X*w)/C
    dr_dq = 2*(p * (X*g - Y*f) + Y*w)/C
    dr_dlmb = v / n


    # velocity derivatives
    dv_da = -v/2/a
    dv_dh = dXd_dh*f + dYd_dh*g
    dv_dk = dXd_dk*f + dYd_dk*g
    dv_dp = 2*(q*(Yd*f - Xd*g) - Xd*w)/C
    dv_dq = 2*(p*(Xd*g - Yd*f) + Yd*w)/C
    dv_dlmb = -n*a**3*r/_r**3

    Jacobian_r = np.array([dr_da,dr_dh,dr_dk,dr_dp,dr_dq,dr_dlmb]).T
    Jacobian_v = np.array([dv_da,dv_dh,dv_dk,dv_dp,dv_dq,dv_dlmb]).T
    Jacobian = np.block([[Jacobian_r],[Jacobian_v]])
    return Jacobian
