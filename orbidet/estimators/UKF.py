"""
This file has the classes to perform both DD-UKF and CD-UKF.
The implementations are based on the algorithms of the paper
#On Unscented KF for State Estimation of Continuous-Time Nonlinear Systems
"""
import numpy as np
from scipy import integrate
from scipy.stats import chi2
from scipy.linalg import cholesky


def _f(x):
    print(x)

def matrix_function(f,X):
    """
    Y = matrix_function(f,X), where X is a matrix of columns X = [x1, x2, ..., xn]
    Y = [f(x1), f(x2), ..., f(xn)] the output is a matrix with n columns
    """
    return np.apply_along_axis(f, 0, X)


def UT_get_mean_cov(sigma_x,sigma_y,wm,W):
    """Get the statistics of the transformed variable"""
    mean = sigma_y @ wm
    P_y = sigma_y @ W @ sigma_y.T
    P_xy = sigma_x @ W @ sigma_y.T
    return mean,P_y,P_xy

def getSigmafromState(x,P_x,lamb):
    n = len(x)
    """Calculating the sigma points"""
    A = cholesky(P_x, lower=True)

    sigma = np.full((2*n+1,n),x).T
    sigma = sigma + np.sqrt(n+lamb) * np.block([np.zeros((n,1)),A,-A])
    return sigma


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def UT_matrix(f,x_mean,P_x,lamb,wm,W,return_sigmas=False):
    """
    implementation of the matrix UT algorithm 4.1 from [the CD-UKF paper]
    INPUTS:
        #f - non-linear state function that receives a vector input (the state)
            However in the matrix form algorithm the non-linear function is applied
            to a matrix (i.e. for each column) this new matrix non-linear function is defined as matrix_function(.)
        # x_mean and P_x are the state mean and covariance of the variable to be transformed
    """
    n = len(x_mean)
    """Calculating the sigma points"""
    try:
        A = cholesky(P_x, lower=True)
    except np.linalg.LinAlgError:
        A = cholesky(nearestPD(P_x), lower=True)

    sigma_X = np.full((2*n+1,n),x_mean).T
    sigma_X = sigma_X + np.sqrt(n+lamb) * np.block([np.zeros((n,1)),A,-A])

    """Transforming the sigma points"""
    sigma_Y = matrix_function(f,sigma_X)

    """Get the statistics of the transformed variable"""
    y_mean = sigma_Y @ wm
    P_y = sigma_Y @ W @ sigma_Y.T
    P_xy = sigma_X @ W @ sigma_Y.T
    if return_sigmas:
        return (y_mean,P_y,P_xy,sigma_X,sigma_Y)
    else:
        return (y_mean,P_y,P_xy)


def predic_dif_eq_sqroot(t,X,f,wm,W,Q,n,lamb):
    """
    This function applies the square root version of the continuous predict step
    equation (35)
    This fuction does not rely on the UT_matrix() function because the implementation is a bit different
    (it would get a bit messy to try an unified solution - plus this way it's less computations)
    """
    #unpack the sigma Matrix
    sigmaX = np.reshape(X,(n,2*n+1))

    sqrt_c = np.sqrt(n+lamb)

    #recover A from sigmaX and get inv(A)
    m = sigmaX[0:,0]
    mMatrix = np.full((n,n),m).T

    A = (sigmaX[0:,1:7] - mMatrix) / sqrt_c
    A_inv = np.linalg.inv(A)

    #apply state function to sigmaX
    fun = lambda _x: f(t,_x)
    sigmaY = matrix_function(fun,sigmaX) # sigmaY = f(X(t),t)

    #get M and phi(M)
    M = A_inv @ ( sigmaX @ W @ sigmaY.T + sigmaY @ W @ sigmaX.T + Q) @ A_inv.T
    phi = np.tril(np.ones((n,n)))
    np.fill_diagonal(phi, 0.5)
    phi_M = phi * M                         #element-wise multiplication. eq (33)

    #finally, get dX_dt (column by column) and pack result
    mean_sigY = np.full((2*n+1,n),sigmaY@wm).T

    block = mean_sigY + sqrt_c*np.block([np.zeros((n,1)),A @ phi_M,-A @ phi_M])
    dX_dt = block.flatten()
    return dX_dt


#UKF classes
class UKF():
    """
    This class implements the original UKF (Discret form - DD-UKF)
    However, since this problem is continuous time, the state function here is discretized
    To implement the CD-UKF, there exists a subclass of this class that
    overrides the predict method (everything else is equal)
    """

    def __init__(self, x0, P0, f,**kwargs):
        """
        x0,P0 initial state and covariance
        f is the state dynamics
        """
        n = len(x0)
        #Parameters of Unscented Transform
        k = 0 #scaling parameter - usually set to 0 or (3 - n)
        alpha = 1 #spread of sigma points parameter - typically 1e-4 <= alpha <= 1
        beta = 2 #optimal value for Gaussian distributions [ref: UKF theory papers]

        self.n = n
        self.x = x0
        self.P = P0
        self.f = f              # state transition function x_{k+1} = f(x{k})
                                #arguments of f(x0,t0,t1)
        #calculate weights
        self.calculate_weights(n,k,alpha,beta)

        self.robust = kwargs.get("robust_variation",False)
        if self.robust:
            self.robust_info = kwargs["parameters"]
            self.Qd = None

    def calculate_weights(self,n,k,alpha,beta):
        """
        calculate weight matrices for the matrix form Unscented transform
        (algorithm 4.1)
        """
        self.lamb = alpha**2 * (n + k) - n
        w_i = [1 / (2*(n + self.lamb)) for _ in range(2*n)]
        self.W_m = np.array([self.lamb / (n + self.lamb)] + w_i)
        self.W_c = np.array([self.lamb / (n + self.lamb) + (1 - alpha**2 + beta)] + w_i)

        wm_matrix = np.full((2*n+1,2*n+1),self.W_m).T
        self.W = (np.eye(2*n+1) - wm_matrix) @ np.diag(self.W_c) @ (np.eye(2*n+1) - wm_matrix).T

    def predict(self,date_in,date_out,Q,method):
        """ Performs the predict step of the DD-UKF. On return,
        self.xp and self.Pp contain the predicted state (xp)
        and covariance (Pp). 'p' stands for prediction.
        """
        if date_in == date_out:
            return

        f = lambda x: self.f(x,date_in,date_out)
        self.x,P,Px_k_kbef = UT_matrix(f,self.x,self.P,self.lamb,self.W_m,self.W)

        # discretization of Q
        if not isinstance(Q,dict):
            Qd = Q
        elif Q["type"] is "discrete":
            Qd = Q["value"]
        else: #Q["type"] is "continuous":
            phi = Px_k_kbef.T @ np.linalg.inv(self.P)
            Qd = phi @ Q["value"] @ phi.T * (date_out - date_in)
        self.P = P + Qd

        if self.robust:
            self.x_pred = self.x.copy()

    def update(self,y,t,h,R,*args):
        """
        Input - observation y measurement at time t_{k+1}
        """
        #Implementacao 1 (criar uma nova UT)
        y_mean,Py,Pxy = UT_matrix(h,self.x,self.P,self.lamb,self.W_m,self.W)
        Py = Py + R

        Sinv = np.linalg.inv(Py) #4x4
        v = (y - y_mean)

        K = Pxy @ Sinv #6x4
        self.x = self.x + K @ v
        self.P = self.P - K @ Py @ K.T

        if self.robust:
            # robust adaptive extra iteration
            mu = (y - h(self.x_pred)).reshape((len(y),1))
            phi = mu.T @ Sinv @ mu
            threshold = chi2.isf(q=1-self.robust_info["prob"], df=len(mu))
            if phi > threshold:
                # perform correction
                lmb = max(self.robust_info["lambda_0"], (phi - self.robust_info["a"]* threshold) / phi)
                self.Qd = (1 - lmb)*self.Qd + lmb*(K @ mu @ mu.T @ K.T)
                print(self.Qd)

                # 1.
                y_mean,Py,Pxy = UT_matrix(h,self.x,self.P,self.lamb,self.W_m,self.W)
                self.P += self.Qd

                # 2.
                Py = Py + R
                K = Pxy @ np.linalg.inv(Py)
                self.x = self.x + K @ (y - y_mean)
                self.P = self.P - K @ Py @ K.T

        return Sinv,v

class CD_UKF(UKF):

    def __init__(self, x0, P0, f):
        print("CD-UKF has been discontinued... Sorry" )
        exit()
        super(CD_UKF, self).__init__(x0, P0,f)

        #extra variable needed to the root version
        self.sigmaX = None

        self._n = len(x0)
        self.cython = use_cython
        self.osc_cython = osc_cython

    def predict(self,date_in,date_out,Q,method):
        if date_in == date_out:
            return
        if not self.cython:
            dynfun = lambda t,y: predic_dif_eq_sqroot(t,y,self.f, self.W_m,self.W,Q,self._n,self.lamb)
        else:
            dynfun = lambda t,y: predict_cython(t,y,self.osc_cython, self.W_m,self.W,Q,self._n,self.lamb)

        #initialize sigma X matrix for the integration
        A = cholesky(self.P, lower=True)

        n = self._n
        sigmaX = np.full((2*n+1,n),self.x).T
        self.sigmaX = sigmaX + np.sqrt(n+self.lamb)*np.block([np.zeros((n,1)),A,-A])

        #integrate function
        Y0 = self.sigmaX.flatten()
        solver = integrate.solve_ivp(dynfun, (date_in,date_out),
                                     Y0, method=method, t_eval=[date_out])


        solver.y = solver.y.flatten()

        #unpack the sigma Matrix and get the state
        sigmaX = np.reshape(solver.y,(n,2*n+1))
        x = sigmaX[0:,0]
        mMatrix = np.full((n,n),x).T
        A = (sigmaX[0:,1:n+1] - mMatrix) / np.sqrt(n+self.lamb)
        self.x = x.copy()
        self.P = A @ A.T
