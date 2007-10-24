import numpy

# For treatment of missing data, see:
#
# http://www.quantlet.com/mdstat/scripts/xfg/html/xfghtmlnode92.html
#
# Shumway, R.H. & Stoffer, D.S. (1982). An approach to time series
# smoothing and forecasting using the EM algorithm. Journal of Time
# Series Analysis, 3, 253-264. http://www.stat.pitt.edu/stoffer/em.pdf

class KalmanFilter:
    def __init__(self,A,C,Q,R,initial_x,initial_P):
        self.A = A # process update model
        self.C = C # observation model
        self.Q = Q # process covariance matrix
        self.R = R # measurement covariance matrix
        self.xhat_k1 = initial_x # a posteri state estimate from step (k-1)
        self.P_k1 = initial_P    # a posteri error estimate from step (k-1)

        self.ss = self.A.shape[0] # ndim in state space
        self.os = self.C.shape[0] # ndim in observation space
        self.AT = self.A.T
        self.CT = self.C.T

        if len(initial_x)!=self.ss:
            raise ValueError( 'initial_x must be a vector with ss components' )
        
    def step(self,y=None,isinitial=False):
        xhatminus, Pminus = self.step1__calculate_a_priori(isinitial=isinitial)
        return self.step2__calculate_a_posteri(xhatminus, Pminus, y=y)
    
    def step1__calculate_a_priori(self,isinitial=False):
        dot = numpy.dot # shorthand
        ############################################
        #          update state-space
        
        # compute a priori estimate of statespace
        if not isinitial:
            xhatminus = dot(self.A,self.xhat_k1)
            # compute a priori estimate of errors
            Pminus = dot(dot(self.A,self.P_k1),self.AT)+self.Q
        else:
            xhatminus = self.xhat_k1
            Pminus = self.P_k1

        return xhatminus, Pminus

    def step2__calculate_a_posteri(self,xhatminus,Pminus,y=None):
        """
        y represents the observation for this time-step
        """
        dot = numpy.dot # shorthand
        inv = numpy.linalg.inv
        
        ############################################
        #          incorporate observation

        # calculate Kalman gain
        Knumerator = dot(Pminus,self.CT)
        Kdenominator = dot(dot(self.C,Pminus),self.CT)+self.R
        K = dot(Knumerator,inv(Kdenominator))

        # calculate a posteri state estimate
        if y is not None:
            residuals = y-dot(self.C,xhatminus) # error/innovation
            xhat = xhatminus+dot(K, residuals)
        else:
            xhat = xhatminus
            
        one_minus_KC = numpy.eye(self.ss)-dot(K,self.C)
        
        # compute a posteri estimate of errors
        P = dot(one_minus_KC,Pminus)

        # this step (k) becomes next step's prior (k-1)
        self.xhat_k1 = xhat
        self.P_k1 = P

        return xhat, P

def kalman_smoother(y,A,C,Q,R,init_x,init_V,valid_data_idx=None):
    """

    arguments
    ---------
    y - observations
    A - process update matrix
    C - state-to-observation matrix
    Q - process covariance matrix
    R - observation covariance matrix
    init_x - initial state
    init_V - initial error estimate
    valid_data_idx - (optional) indices to rows of y that are valid (None if all data valid)

    returns
    -------
    xsmooth - smoothed state estimates
    Vsmooth - smoothed error estimates

    Kalman smoother based on Kevin Murphy's Kalman toolbox for
    MATLAB(tm).

    N.B. Axes are swapped relative to Kevin Murphy's example, because
    in all my data, time is the first dimension."""
    
    def smooth_update(xsmooth_future,Vsmooth_future,xfilt,Vfilt,Vfilt_future,A,Q):
        dot = numpy.dot
        inv = numpy.linalg.inv
        
        xpred = dot(A,xfilt)
        Vpred = dot(A,numpy.dot(Vfilt,A.T)) + Q
        J = dot(Vfilt,numpy.dot(A.T,inv(Vpred))) # smoother gain matrix
        xsmooth = xfilt + dot(J, xsmooth_future-xpred)
        Vsmooth = Vfilt + dot(J,dot(Vsmooth_future-Vpred,J.T))
        return xsmooth, Vsmooth
        
    T, os = y.shape
    ss = len(A)

    kfilt = KalmanFilter(A,C,Q,R,init_x,init_V)
    # Forward pass
    xfilt = numpy.zeros((T,ss))
    Vfilt = numpy.zeros((T,ss,ss))

    for i in range(T):
        isinitial = i==0
        if (valid_data_idx is None) or (i in valid_data_idx):
            y_i = y[i]
        else:
            y_i = None

        xfilt_i, Vfilt_i = kfilt.step(y=y_i,isinitial=isinitial)
        xfilt[i] = xfilt_i
        Vfilt[i] = Vfilt_i
        
    xsmooth = numpy.array(xfilt,copy=True)
    Vsmooth = numpy.array(Vfilt,copy=True)

    for t in range(T-2,-1,-1):
        xsmooth_t, Vsmooth_t = smooth_update(xsmooth[t+1,:],
                                             Vsmooth[t+1,:,:],
                                             xfilt[t,:],
                                             Vfilt[t,:,:],
                                             Vfilt[t,:,:],
                                             A,Q)
        xsmooth[t,:] = xsmooth_t
        Vsmooth[t,:,:] = Vsmooth_t

    return xsmooth, Vsmooth
