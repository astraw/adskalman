from __future__ import division
import numpy

# For treatment of missing data, see:
#
# http://www.quantlet.com/mdstat/scripts/xfg/html/xfghtmlnode92.html
#
# Shumway, R.H. & Stoffer, D.S. (1982). An approach to time series
# smoothing and forecasting using the EM algorithm. Journal of Time
# Series Analysis, 3, 253-264. http://www.stat.pitt.edu/stoffer/em.pdf

def gaussian_prob(x,m,C,use_log=False):
    m = numpy.atleast_1d(m)
    assert len(x.shape)==1
    N=1
    d = x.shape[0]
    M = numpy.dot(m.T,numpy.ones((1,N))) # replicate mean across columns
    denom = (2*numpy.pi)**(d/2)*numpy.sqrt(abs(numpy.linalg.det(C)))
    x = x[:,numpy.newaxis] # make column vector
    xMT = (x-M).T
    tmpXX = (numpy.dot(xMT,numpy.linalg.inv(C))) * xMT
    mahal = numpy.sum(tmpXX.flat)
    if numpy.any( mahal<0 ):
        raise ValueError("mahal < 0 => C is not psd")
    if use_log:
        p = -0.5*mahal - numpy.log(denom)
    else:
        eps=2**-52
        p = numpy.exp( -0.5*mahal ) / (denom+eps)
    return p

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

    def step(self,y=None,isinitial=False,full_output=False):
        xhatminus, Pminus = self.step1__calculate_a_priori(isinitial=isinitial)
        return self.step2__calculate_a_posteri(xhatminus, Pminus, y=y, full_output=full_output)

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

    def step2__calculate_a_posteri(self,xhatminus,Pminus,y=None,full_output=False):
        """
        y represents the observation for this time-step
        """
        dot = numpy.dot # shorthand
        inv = numpy.linalg.inv

        if y is not None:
            ############################################
            #          incorporate observation

            # calculate a posteri state estimate

            # calculate Kalman gain
            Knumerator = dot(Pminus,self.CT)
            Kdenominator = dot(dot(self.C,Pminus),self.CT)+self.R
            K = dot(Knumerator,inv(Kdenominator))

            residuals = y-dot(self.C,xhatminus) # error/innovation
            xhat = xhatminus+dot(K, residuals)

            one_minus_KC = numpy.eye(self.ss)-dot(K,self.C)

            # compute a posteri estimate of errors
            P = dot(one_minus_KC,Pminus)
        else:
            # no observation
            xhat = xhatminus
            P = Pminus

        if full_output:
            # calculate loglik and Pfuture
            VVnew = dot(one_minus_KC,dot(self.A,self.P_k1))
            loglik = gaussian_prob( residuals, numpy.zeros((1,len(residuals))), Kdenominator, use_log=True)
        # this step (k) becomes next step's prior (k-1)
        self.xhat_k1 = xhat
        self.P_k1 = P
        if full_output:
            return xhat, P, loglik, VVnew
        else:
            return xhat, P

def kalman_filter(y,A,C,Q,R,init_x,init_V,full_output=False):
    y = numpy.asarray(y)

    T, os = y.shape
    ss = len(A)

    kfilt = KalmanFilter(A,C,Q,R,init_x,init_V)
    # Forward pass
    xfilt = numpy.zeros((T,ss))
    Vfilt = numpy.zeros((T,ss,ss))
    if full_output:
        VVfilt =  numpy.zeros((T,ss,ss))
    loglik = 0

    for i in range(T):
        isinitial = i==0
        y_i = y[i]

        if full_output:
            xfilt_i, Vfilt_i, LL, VVfilt_i = kfilt.step(y=y_i,
                                                        isinitial=isinitial,
                                                        full_output=True)
            VVfilt[i] = VVfilt_i
            loglik += LL
        else:
            xfilt_i, Vfilt_i = kfilt.step(y=y_i,
                                          isinitial=isinitial,
                                          full_output=False)
        xfilt[i] = xfilt_i
        Vfilt[i] = Vfilt_i

    if full_output:
        return xfilt,Vfilt,VVfilt,loglik
    else:
        return xfilt,Vfilt

def kalman_smoother(y,A,C,Q,R,init_x,init_V,valid_data_idx=None,full_output=False):
    """Rauch-Tung-Striebel (RTS) smoother

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
    VVsmooth - (only when full_output==True)
    loglik - (only when full_output==True)

    Kalman smoother based on Kevin Murphy's Kalman toolbox for
    MATLAB(tm).

    N.B. Axes are swapped relative to Kevin Murphy's example, because
    in all my data, time is the first dimension."""

    def smooth_update(xsmooth_future,Vsmooth_future,xfilt,Vfilt,Vfilt_future,VVfilt_future,A,Q,full_output=False):
        dot = numpy.dot
        inv = numpy.linalg.inv

        xpred = dot(A,xfilt)
        Vpred = dot(A,numpy.dot(Vfilt,A.T)) + Q
        J = dot(Vfilt,numpy.dot(A.T,inv(Vpred))) # smoother gain matrix
        xsmooth = xfilt + dot(J, xsmooth_future-xpred)
        Vsmooth = Vfilt + dot(J,dot(Vsmooth_future-Vpred,J.T))
        VVsmooth_future = VVfilt_future + numpy.dot(
            (Vsmooth_future - Vfilt_future),
            numpy.dot(inv(Vfilt_future),VVfilt_future))
        return xsmooth, Vsmooth, VVsmooth_future

    T, os = y.shape
    ss = len(A)

    # Forward pass
    forward_results = kalman_filter(y,A,C,Q,R,init_x,init_V,full_output=full_output)
    if full_output:
        xfilt,Vfilt,VVfilt,loglik = forward_results
    else:
        xfilt,Vfilt = forward_results
        VVfilt = Vfilt # dummy value

    # Backward pass
    xsmooth = numpy.array(xfilt,copy=True)
    Vsmooth = numpy.array(Vfilt,copy=True)
    VVsmooth = numpy.empty(Vfilt.shape)

    for t in range(T-2,-1,-1):
        xsmooth_t, Vsmooth_t, VVsmooth_t = smooth_update(xsmooth[t+1,:],
                                                         Vsmooth[t+1,:,:],
                                                         xfilt[t,:],
                                                         Vfilt[t,:,:],
                                                         Vfilt[t+1,:,:],
                                                         VVfilt[t+1,:,:],
                                                         A,Q,
                                                         full_output=full_output)
        xsmooth[t,:] = xsmooth_t
        Vsmooth[t,:,:] = Vsmooth_t
        VVsmooth[t+1,:,:] = VVsmooth_t

    VVsmooth[0,:,:]=numpy.zeros((ss,ss))

    if full_output:
        return xsmooth, Vsmooth, VVsmooth, loglik
    else:
        return xsmooth, Vsmooth

def learn_kalman(data, A, C, Q, R, initx, initV,
                 max_iter=10, diagQ=False, diagR=False,
                 ARmode=False, constr_fun_dict={}):
    verbose=True
    inv = numpy.linalg.inv

    def em_converged(loglik, previous_loglik, threshold=1e-4, check_increased=True):
        converged = False
        decrease = False
        if check_increased:
            if loglik - previous_loglik < -1e-3:
                print '******likelihood decreased from %6.4f to %6.4f!'%(previous_loglik, loglik)
                decrease = True
                converged = False
                return converged, decrease
        delta_loglik = abs(loglik - previous_loglik)
        eps=2**-52
        avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2
        if (delta_loglik/avg_loglik) < threshold:
            converged = True
        return converged, decrease

    def Estep(y, A, C, Q, R, initx, initV, ARmode):
        T, os = y.shape
        ss = len(A)
        if ARmode:
            xsmooth = y
            Vsmooth = numpy.zeros((T,ss,ss))
            VVsmooth = numpy.zeros((T,ss,ss))
            loglik = 0
        else:
            xsmooth, Vsmooth, VVsmooth, loglik = kalman_smoother(y, A, C, Q, R, initx, initV, full_output=True)

        delta = numpy.zeros((os,ss))
        gamma = numpy.zeros((ss,ss))
        beta = numpy.zeros((ss,ss))
        for t in range(T):
            delta = delta + numpy.dot( y[t,:,numpy.newaxis], xsmooth[t,:,numpy.newaxis].T )
            gamma = gamma + numpy.dot(xsmooth[t,:,numpy.newaxis],xsmooth[t,:,numpy.newaxis].T) + Vsmooth[t,:,:]
            if t>0:
                beta = beta +  numpy.dot(xsmooth[t,:,numpy.newaxis],xsmooth[t-1,:,numpy.newaxis].T) + VVsmooth[t,:,:]
        gamma1 = gamma - numpy.dot(xsmooth[T-1,:,numpy.newaxis],xsmooth[T-1,:,numpy.newaxis].T) - Vsmooth[T-1,:,:]
        gamma2 = gamma - numpy.dot(xsmooth[0,:,numpy.newaxis],xsmooth[0,:,numpy.newaxis].T) - Vsmooth[0,:,:]

        x1 = xsmooth[0,:]
        V1 = Vsmooth[0,:,:]
        return beta, gamma, delta, gamma1, gamma2, x1, V1, loglik
    thresh = 1e-4

    N = 1
    ss = A.shape[0]
    os = C.shape[0]

    alpha = numpy.zeros((os,os))
    Tsum = 0

    if 1:
    #for ex in range(N):
        #y=data[ex]
        y = data
        T = y.shape[0]
        Tsum += T
        alpha_temp = numpy.zeros((os,os))
        for t in range(T):
            yt = y[t,:,numpy.newaxis]
            alpha_temp = alpha_temp + numpy.dot(yt,yt.T)
        alpha = alpha + alpha_temp
    previous_loglik = -numpy.inf
    converged = False
    num_iter = 0
    LL = []

    while (not converged) and (num_iter < max_iter):
        # E step
        delta = numpy.zeros((os,ss))
        gamma = numpy.zeros((ss,ss))
        gamma1 = numpy.zeros((ss,ss))
        gamma2 = numpy.zeros((ss,ss))
        beta = numpy.zeros((ss,ss))
        P1sum = numpy.zeros((ss,ss))
        x1sum = numpy.zeros((ss,))
        loglik = 0

        if 1:
            #for ex in range(N):
            y = data
            T = len(y)
            beta_t, gamma_t, delta_t, gamma1_t, gamma2_t, x1, V1, loglik_t = Estep(y, A, C, Q, R, initx, initV, ARmode)
            beta = beta + beta_t
            gamma = gamma + gamma_t
            delta = delta + delta_t
            gamma1 = gamma1 + gamma1_t
            gamma2 = gamma2 + gamma2_t
            P1sum = P1sum + V1 + numpy.dot(x1[:,numpy.newaxis],x1[:,numpy.newaxis].T)
            x1sum = x1sum + x1
            loglik = loglik + loglik_t
        LL.append( loglik )
        if verbose:
            print 'iteration %d, loglik = %f'%(num_iter, loglik)
        num_iter += 1

        # M step
        Tsum1 = Tsum-N
        A = numpy.dot(beta,inv(gamma1))
        Q = (gamma2 - numpy.dot(A,beta.T))/Tsum1
        if diagQ:
            Q = numpy.diag( numpy.diag(Q) )
        if not ARmode:
            C = numpy.dot(delta,inv(gamma))
            R = (alpha - numpy.dot(C,delta.T)) / Tsum
            if diagR:
                R = numpy.diag( numpy.diag( R ))
        initx = x1sum/N
        initV = P1sum/N - numpy.dot(initx[:,numpy.newaxis],initx[:,numpy.newaxis].T)
        if len(constr_fun_dict.keys()):
            raise NotImplementedError("")
        converged, em_decrease = em_converged(loglik, previous_loglik, thresh)
        previous_loglik = loglik
    return A, C, Q, R, initx, initV, LL
