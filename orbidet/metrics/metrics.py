import numpy as np
import pandas as pd

from .plot_utils import *
from scipy.stats import chi2

def _count(val,upper,lower=0):
    if val > upper or val < lower:
        return 1
    else:
        return 0


class Metrics():
    """
    this class aplies the following procedures / metrics:
        -RMS Error at each instant and the averaged error along the full trajectory
        -Absolute error (State_est - State_true) along with the error estimation (sigma covariances)
    Note that these plots can be represented in ECI or RSW frames

        -Consistency tests:
            -NEES (Normalized estimation error squared) test
            -NIS
    """
    counter = 0
    _t = 0
    dict_axis = {'ECI':['x','y','z','velocity x','velocity y','velocity z'],
            'RSW':['Radial','AlongTrack','CrossTrack','Velocity Radial','Velocity AlongTrack','Velocity CrossTrack']}

    def __init__(self,M,RMSE_errors = True, consistency_tests = False, abs_erros=False,frames = ("ECI",)):
        """
            M - number of Monte Carlo Runs
            consistency_tests - flag to calculate NEES and NIS
            abs_errors - flag to calculate absolute errors
            RMSE_errors - flag to calculate RMS errors
            frames - to calculate abs and RMSE errors (ECI or RSW)
        """

        #data variables for absolute error and RMS error accumulation
        if "ECI" in frames:
            if abs_erros:
                self.data_ECI = {}      #data = [ [t_1,error_x,...,e_vz,sigma_x,...sigma_vz], ... [t_i,e_x,...,e_vz,sigma_x,...sigma_vz], ...  ]
            if RMSE_errors:
                self.data_mse_ECI = {}  #[ [t_1,mse_x,...,mse_vz,], ... [t_i,mse_x,...,mse_vz,], ...  ]
        if "RSW" in frames:
            if abs_erros:
                self.data_RSW = {}  #the same but in RSW coordinates
            if RMSE_errors:
                self.data_mse_RSW = {}
        if "ECI" not in frames and "RSW" not in frames:
            raise Exception("either ECI or RSW frames should be selected in *frames* argument")

        if consistency_tests:
            #data variables for the CONSISTENCY TESTS:
            self.data_NEES = {} # [[t_i,NEES_i],...[],] #normalized estimation error squared
            self.data_NIS = {} # [[t_i,NIS_i],...[],] #normalized innovation squared

        self.M = M

        #control variables
        self.frames = frames
        self.RMSE_errors = RMSE_errors
        self.consistency_tests = consistency_tests
        self.abs_erros = abs_erros



    def append_estimation(self,t,x_true,x_est,P=None,R_ECI_to_RSW=None,Sinv=None,obs_err=None):
        """
        the basic implementation of this class is simply to append the position and velocity RMSE.
        Absolte errors and consistency tests are only implemented if the objects are not None
        arguments:
            t - time instant of this state estimation
            x_true,x_est - true and estimate states
            covariance P (if P exists, then append absolute error est.) ABSOLUTE ERRORS
            rotations matrix R_ECI_to_RSW PRESENT RESULTS IN RSW
            Sinv and obs_err are used for NIS and NEES   CONSISTENCY TESTS
        """
        #These calculations are always used (in the ECI frame)
        error_pos = 1000*(x_true[0:3] - x_est[0:3]) #[m]
        error_vel = 1000*(x_true[3:6] - x_est[3:6]) #[m/s]

        for frame in self.frames:
            #auxilary variables for each frame
            if frame == 'ECI':
                R = np.eye(3)
                if self.abs_erros:
                    data = self.data_ECI
                if self.RMSE_errors:
                    data_ms = self.data_mse_ECI

            elif frame == 'RSW':
                if not isinstance(R_ECI_to_RSW,type(None)):
                    R = R_ECI_to_RSW
                    if self.abs_erros:
                        data = self.data_RSW
                    if self.RMSE_errors:
                        data_ms = self.data_mse_RSW
                else:
                    continue

            #get absolute and mean square error for this instant t
            error_pos_ = R @ error_pos
            error_vel_ = R @ error_vel
            mse_pos_ = error_pos_**2
            mse_vel_ = error_vel_**2

            #if cov P is provided, get sigmas in the right frame
            if not isinstance(P,type(None)):
                P_pos = R @ P[0:3,0:3] @ R.T
                P_vel = R @ P[3:,3:] @ R.T
                sigmas = self.get_sigmas(np.block([[P_pos,np.zeros((3,3))],[np.zeros((3,3)),P_vel]]))
                self.sigmas = True
            else:
                sigmas = None
                self.sigmas = False

            #append result
            if self.abs_erros:
                entry = [error_pos_[0]/self.M,error_pos_[1]/self.M,error_pos_[2]/self.M,error_vel_[0]/self.M,error_vel_[1]/self.M,error_vel_[2]/self.M]
                if sigmas:
                    entry += [sigmas[0]/self.M,sigmas[1]/self.M,sigmas[2]/self.M,sigmas[3]/self.M,sigmas[4]/self.M,sigmas[5]/self.M]
            if self.RMSE_errors:
                entry_mse = [mse_pos_[0],mse_pos_[1],mse_pos_[2],mse_vel_[0],mse_vel_[1],mse_vel_[2]]


            if self.abs_erros:
                if t in data:
                    aux = data[t]
                    for i in range(len(aux)):
                        aux[i] += entry[i]
                    data[t] = aux
                else:
                    data[t] = entry
            if self.RMSE_errors:
                if t in data_ms:
                    aux = data_ms[t]
                    for i in range(len(aux)):
                        aux[i] += entry_mse[i]
                    data_ms[t] = aux
                else:
                    data_ms[t] = entry_mse


        if self.consistency_tests:
            if (not isinstance(P,type(None))):
                NEES_i = (x_true - x_est).T @ np.linalg.inv(P) @ (x_true - x_est)
                if t not in self.data_NEES:
                    self.data_NEES[t] = NEES_i
                else:
                    self.data_NEES[t] += NEES_i

            #only do NIS if error is provided
            if isinstance(obs_err,np.ndarray):
                NIS_i = obs_err.T @ Sinv @ obs_err
                if t not in self.data_NIS:
                    self.data_NIS[t] = NIS_i
                else:
                    self.data_NIS[t] += NIS_i

    def _process_results(self,data_mse_dict,data_dict):
        """
        method to calculate RMSE for each time instant. This is for the plots
        """
        results = []
        if data_dict is not None and data_mse_dict is not None:
            for (t,mse),(_t,abs_e) in zip(data_mse_dict.items(),data_dict.items()):
                if self.sigmas:
                    sigmas = list(map(lambda x: np.sqrt(x),abs_e[6:]))
                    abs_e[6:] = sigmas[:]
                entry = [t] + list(map(lambda x: np.sqrt(x/self.M),mse[0:6])) +  list(
                    map(lambda x: np.sqrt(np.sum(x) / self.M)  , (mse[0:3],mse[3:6]))) + abs_e
                results.append(entry)
            return results
        elif data_dict is None:
            for (t,mse) in data_mse_dict.items():
                entry = [t] + list(map(lambda x: np.sqrt(x/self.M),mse[0:6])) +  list(
                    map(lambda x: np.sqrt(np.sum(x) / self.M)  , (mse[0:3],mse[3:6])))
                results.append(entry)
            return results
        elif data_mse_dict is None:
            for (t,abs_e) in data_dict.items():
                if self.sigmas:
                    sigmas = list(map(lambda x: np.sqrt(x),abs_e[6:]))
                    abs_e[6:] = sigmas[:]
                entry = [t] + abs_e
                results.append(entry)
            return results

    def process_results(self,path="",save=False):
        """
        this function calculates and saves the results in dataframes.
        INPUTS
            path -> the path where to save the results (/path/so/save/name_file.csv)
        """
        self.DF_results = {}

        for frame in self.frames:

            if frame == 'ECI':
                data = self.data_ECI if self.abs_erros else None
                data_mse = self.data_mse_ECI if self.RMSE_errors else None
            elif frame == 'RSW':
                data = self.data_RSW if self.abs_erros else None
                data_mse = self.data_mse_RSW if self.RMSE_errors else None

            #put everything in a dataframe and save it
            columns = ['t']
            data_results = self._process_results(data_mse,data)

            if self.RMSE_errors and len(data_mse) > 0:
                columns += ['rmse_x','rmse_y','rmse_z','rmse_vx','rmse_vy','rmse_vz','rmse_pos','rmse_vel']
            if self.abs_erros and len(data) > 0:
                columns += ['x','y','z','vx','vy','vz']
                if self.sigmas:
                    columns += ['sigma_x','sigma_y','sigma_z','sigma_vx','sigma_vy','sigma_vz']


            self.DF_results[frame] = pd.DataFrame(data_results,columns=columns)
            if save:
                self.DF_results[frame].to_csv(path+frame+".csv", index = False)

        if self.consistency_tests:
            data_NEES = []
            data_NIS = []
            for key, value in self.data_NEES.items():
                temp = [key,value]
                data_NEES.append(temp)
            self.data_NEES = data_NEES
            for key, value in self.data_NIS.items():
                temp = [key,value]
                data_NIS.append(temp)
            self.data_NIS = data_NIS
            DF_cons1 = pd.DataFrame(data_NEES,columns=['t','NEES'])
            DF_cons2 = pd.DataFrame(data_NIS,columns=['t','NIS'])
            self.DF_cons = pd.concat([DF_cons1, DF_cons2], axis=1, sort=False)
            if save:
                self.DF_cons.to_csv(path+"consistency.csv", index = False)



    def plot_results(self,len_sensors,len_state,n=-1,side_consistency="two-sided",prob_consistency=0.95,
                     filter_name="",plot=True):
        for frame,DF_results in self.DF_results.items():
            t = DF_results[['t']]
            xlabel = 'time [s]'
            j = 0

            if n == -1 or n < 0 or n > len(DF_results):
                n = len(DF_results)
            _pos,_vel,_x,_y,_z,_vx,_vy,_vz = self.get_full_RMSE(DF_results,('3Dpos','3Dvel'),n)
            self.rmse_pos = _pos
            self.rmse_vel = _vel

            if plot:
                #RMSE plots
                for arg,ite in zip(['rmse_x','rmse_y','rmse_z'],[0,1,2]):
                    y = DF_results[[arg]]
                    title = "Filter: "+ filter_name + ", " + frame +' Position components RMSE for %d Monte-Carlo runs' % self.M
                    ylabel = 'RMSE [m]'
                    plot_graphs(y,t,title,ylabel,xlabel,i=j,label=self.__class__.dict_axis[frame][ite],show_label=True)

                j+=1
                for arg,ite in zip(['rmse_vx','rmse_vy','rmse_vz'],[3,4,5]):
                    y = DF_results[[arg]]
                    title = "Filter: "+ filter_name + ", " + frame +' Velocity components RMSE for %d Monte-Carlo runs' % self.M
                    ylabel = 'RMSE [m/s]'
                    plot_graphs(y,t,title,ylabel,xlabel,i=j,label=self.__class__.dict_axis[frame][ite],show_label=True)

                j+=1
                for arg,ti,uni,full in zip(['rmse_pos','rmse_vel'],['Position','Velocity'],['[m]','[m/s]'],[_pos,_vel]):
                    y = DF_results[[arg]]
                    title = "Filter: "+ filter_name + ", " + ti + '3-D RMSE for %d Monte-Carlo runs' % self.M + "\nAlong full trajectory (last %d points): %.12f "%(n,full) + " " + uni
                    ylabel = 'RMSE ' + uni
                    j+=1
                    print(ti," RMSE Along full traj: ",full," ",uni)
                    plot_graphs(y,t,title,ylabel,xlabel,i=j,show_label=False)
                show_plots(True)

                #Absolute Error Plots
                if self.abs_erros:
                    for arg,ite,sigma in zip(['x','y','z'],[0,1,2],['sigma_x','sigma_y','sigma_z']):
                        y = DF_results[[arg]]
                        try:
                            sigmas = True
                            sig_pl = DF_results[[sigma]]
                        except KeyError:
                            sigmas = False
                        title = "Filter: "+ filter_name + ", " + frame +' absolute error in ' + arg + ' for %d Monte-Carlo runs' % self.M
                        ylabel = 'Error [m]'
                        plot_graphs(y,t,title,ylabel,xlabel,i=j,label=self.__class__.dict_axis[frame][ite],show_label=True)
                        if sigmas:
                            plot_grid(sig_pl,t,j,r'+$\sigma$')
                            plot_grid(-sig_pl,t,j,r'+$\sigma$')
                            j+=1
                    j+=1
                    for arg,ite,sigma in zip(['vx','vy','vz'],[3,4,5],['sigma_vx','sigma_vy','sigma_vz']):
                        y = DF_results[[arg]]
                        try:
                            sigmas = True
                            sig_pl = DF_results[[sigma]]
                        except:
                            sigmas = False
                        title = "Filter: "+ filter_name + ", " + frame +' absolute error in ' + arg + ' for %d Monte-Carlo runs' % self.M
                        ylabel = 'Error [m/s]'
                        plot_graphs(y,t,title,ylabel,xlabel,i=j,label=self.__class__.dict_axis[frame][ite],show_label=True)
                        if sigmas:
                            plot_grid(sig_pl,t,j,r'+$\sigma$')
                            plot_grid(-sig_pl,t,j,r'+$\sigma$')
                            j+=1
                    j+=1
                    show_plots(True)

            if self.consistency_tests:
                self.evaluate_consistency(len_sensors,len_state,side=side_consistency,prob=prob_consistency,filter_name=filter_name)


    def evaluate_consistency(self,n_z,n_x=6,side="two-sided",prob=0.95,filter_name=""):
        """
        computes the NEES and NIS plot along with consistent interval bound.
        n_z -> dimension of the measurements
        n_x -> dimension of the state
        side -> "two-sided" or "one-sided"
        prob -> probability for the test
        """
        #NIS
        self.consistency_plots(1,n_z,side,prob,"NIS",self.data_NIS,filter_name)
        #NEES
        self.consistency_plots(0,n_x,side,prob,"NEES",self.data_NEES,filter_name)



    def consistency_plots(self,i,n,side,prob,test,data,filter_name):
        if side is "two-sided":
            up = (1 - prob) / 2
            low = 1 - (1 - prob) / 2
            lower = chi2.isf(q=low, df=self.M*n) / self.M
        else:
            up = (1 - prob)
            lower = 0

        t = [entry[0] for entry in data]
        y = [entry[1]/self.M for entry in data]
        upper = chi2.isf(q=up, df=self.M*n) / self.M

        #counting the number of times when value goes outside bonds
        counter = 0
        for y_i in y:
            counter += _count(y_i,upper,lower)
        print(test+" test - %d points were outside the probability region"% counter)

        title = "Filter: "+ filter_name + ", " + test+" Test considering a %s %.2f%s window" % (side,100*prob,'%')+ "\n"+test+" test - %d points were outside the probability region"% counter
        plot_graphs(y,t,title,test,'Time [s]',i,label=test)
        if up:
            horizontal_line(upper,"Upper Bound",c='r')
        horizontal_line(lower,"Lower Bound",c='y')
        show_plots(True)


    def get_sigmas(self,P):
        sigma_x = 1000**2 * P[0,0]
        sigma_y = 1000**2 * P[1,1]
        sigma_z = 1000**2 * P[2,2]
        sigma_vx = 1000**2 * P[3,3]
        sigma_vy = 1000**2 * P[4,4]
        sigma_vz = 1000**2 * P[5,5]
        return [sigma_x,sigma_y,sigma_z,sigma_vx,sigma_vy,sigma_vz]


    def get_full_RMSE(self,data,args,n):
        """
        method to calculate the RMSE along the full trajectory. Outputs the final result (float)
        """
        pos,vel,x,y,z,vx,vy,vz = 0,0,0,0,0,0,0,0

        for index, row in data[-n:].iterrows():
            if '3Dpos' in args:
                pos += row['rmse_pos']**2
            if '3Dvel' in args:
                vel += row['rmse_vel']**2
            if 'pos' in args:
                x += row['rmse_x']**2
                y += row['rmse_y']**2
                z += row['rmse_z']**2
            if 'vel' in args:
                vx += row['rmse_vx']**2
                vy += row['rmse_vy']**2
                vz += row['rmse_vz']**2

        pos = np.sqrt(pos/n)
        vel = np.sqrt(vel/n)
        x = np.sqrt(x/n)
        y = np.sqrt(y/n)
        z = np.sqrt(z/n)
        vx = np.sqrt(vx/n)
        vy = np.sqrt(vy/n)
        vz = np.sqrt(vz/n)

        return pos,vel,x,y,z,vx,vy,vz
