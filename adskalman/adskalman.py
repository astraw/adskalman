from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import numpy.matlib
linalg = numpy.linalg
# import scikits.learn.machine.em.densities as densities

# For treatment of missing data, see:
#
# Shumway, R.H. & Stoffer, D.S. (1982). An approach to time series
# smoothing and forecasting using the EM algorithm. Journal of Time
# Series Analysis, 3, 253-264. http://www.stat.pitt.edu/stoffer/em.pdf


def rand_mvn(mu, sigma, N):
    """generate multivariate normal samples
input:
  mu - a length M vector, the mean of the distribution
  sigma - an M by M array, the covariance of the distribution
  N - a scalar N, the number of samples to generate
output:
  Y - an N by M array, N samples with mean mu and covariance sigma
"""
    M = sigma.shape[0]
    X = numpy.random.standard_normal((M, N))
    Y = numpy.dot(linalg.cholesky(sigma), X).T + mu
    return Y


def covar(x):
    """determine the sample covariance matrix of x

input:
  x - an N by M array, N samples of an M component vector
output:
  sigma - and M by M array, the covariance matrix
"""
    mu = numpy.mean(x, axis=0)
    N = x.shape[0]
    y = x - mu
    sigma = numpy.dot(y.T, y) / (N - 1)

    # Note, the maximum likelihood estimator is /N [not /(N-1)] as
    # above, but that works only for a multivariate normal.

    return sigma


def gaussian_prob(x, m, C, use_log=False):
    if 0:
        return numpy.asscalar(densities.gauss_den(x, m, C, log=use_log))
    # Kevin Murphy's implementation
    m = numpy.atleast_1d(m)
    assert x.ndim == 1
    N = 1
    d = x.shape[0]
    M = numpy.dot(m.T, numpy.ones((1, N)))  # replicate mean across columns
    denom = (2 * numpy.pi)**(d / 2) * numpy.sqrt(abs(numpy.linalg.det(C)))
    x = x[:, numpy.newaxis]  # make column vector
    xMT = (x - M).T
    tmpXX = (numpy.dot(xMT, numpy.linalg.inv(C))) * xMT
    mahal = numpy.sum(tmpXX.flat)
    if numpy.any(mahal < 0):
        raise ValueError("mahal < 0 => C is not psd")
    if use_log:
        p = -0.5 * mahal - numpy.log(denom)
    else:
        eps = numpy.finfo(numpy.float64).eps
        # eps=2**-52
        p = numpy.exp(-0.5 * mahal) / (denom + eps)
    return p


class VariableObservationNoiseKalmanFilter:

    def __init__(self, A, C, Q, initial_x, initial_P):
        ss = len(A)  # ndim in state space
        os = len(C)  # ndim in observation space

        assert A.shape == (ss, ss)
        assert C.shape == (os, ss)
        assert Q.shape == (ss, ss)
        assert initial_x.shape == (ss,)
        assert initial_P.shape == (ss, ss)

        self.A = A  # process update model
        self.C = C  # observation model
        self.Q = Q  # process covariance matrix

        # These 2 attributes are the only state that changes during
        # filtering:
        self.xhat_k1 = initial_x  # a posteriori state estimate from step (k-1)
        self.P_k1 = initial_P    # a posteriori error estimate from step (k-1)

        self.ss = ss
        self.os = os
        self.AT = self.A.T
        self.CT = self.C.T

        if len(initial_x) != self.ss:
            raise ValueError('initial_x must be a vector with ss components')

    def step(self, y=None, isinitial=False, full_output=False, **kw):
        xhatminus, Pminus = self.step1__calculate_a_priori(isinitial=isinitial)
        return self.step2__calculate_a_posteri(xhatminus, Pminus, y=y,
                                               full_output=full_output, **kw)

    def step1__calculate_a_priori(self, isinitial=False):
        dot = numpy.dot  # shorthand
        ############################################
        #          update state-space

        # compute a priori estimate of statespace
        if not isinitial:
            xhatminus = dot(self.A, self.xhat_k1)
            # compute a priori estimate of errors
            Pminus = dot(dot(self.A, self.P_k1), self.AT) + self.Q
        else:
            xhatminus = self.xhat_k1
            Pminus = self.P_k1

        return xhatminus, Pminus

    def _compute_prediction(self, xhatminus):
        return numpy.dot(self.C, xhatminus)

    def step2__calculate_a_posteri(self, xhatminus, Pminus, y=None,
                                   full_output=False, R=None):
        """
        y represents the observation for this time-step
        """
        if R is None:
            raise ValueError('R cannot be None')
        dot = numpy.dot  # shorthand
        inv = numpy.linalg.inv

        missing_data = False
        if y is None or numpy.any(numpy.isnan(y)):
            missing_data = True

        if not missing_data:
            ############################################
            #          incorporate observation

            # calculate a posteriori state estimate

            # calculate Kalman gain
            Knumerator = dot(Pminus, self.CT)
            Kdenominator = dot(dot(self.C, Pminus), self.CT) + R
            K = dot(Knumerator, inv(Kdenominator))  # Kalman gain

            residuals = y - self._compute_prediction(xhatminus)  # error/innovation
            xhat = xhatminus + dot(K, residuals)

            one_minus_KC = numpy.eye(self.ss) - dot(K, self.C)

            # compute a posteriori estimate of errors
            P = dot(one_minus_KC, Pminus)
        else:
            # no observation
            xhat = xhatminus
            P = Pminus

        if full_output:
            if missing_data:
                # XXX missing data, check literature!
                raise NotImplementedError(
                    "don't know how to compute VVnew with missing data")
                # VVnew = dot(self.A,self.P_k1)
                # loglik = 0
            else:
                # calculate loglik and Pfuture
                VVnew = dot(one_minus_KC, dot(self.A, self.P_k1))
                loglik = gaussian_prob(residuals,
                                       numpy.zeros((1, len(residuals))),
                                       Kdenominator, use_log=True)

        # this step (k) becomes next step's prior (k-1)
        self.xhat_k1 = xhat
        self.P_k1 = P
        if full_output:
            return xhat, P, loglik, VVnew
        else:
            return xhat, P


class KalmanFilter(VariableObservationNoiseKalmanFilter):

    def __init__(self, A, C, Q, R, initial_x, initial_P):
        self.R = R  # measurement covariance matrix
        VariableObservationNoiseKalmanFilter.__init__(
            self, A=A, C=C, Q=Q, initial_x=initial_x, initial_P=initial_P)
        assert R.shape == (self.os, self.os)

    def step2__calculate_a_posteri(self, xhatminus, Pminus, y=None,
                                   full_output=False):
        return VariableObservationNoiseKalmanFilter.step2__calculate_a_posteri(
            self, xhatminus=xhatminus, Pminus=Pminus, y=y, full_output=full_output,
            R=self.R)

class KalmanFilter_NonlinearObservation(KalmanFilter):
    def __init__(self, A, observation_function, Q, R, initial_x, initial_P, delta = 1e-5):
        self.delta = delta
        self.observation_function = observation_function
        C0 = linearize_at(observation_function,initial_x)
        KalmanFilter.__init__(
            self, A=A, C=C0, Q=Q, R=R, initial_x=initial_x, initial_P=initial_P)
        assert R.shape == (self.os, self.os)

    def _compute_prediction(self, xhatminus):
        result = self.observation_function(xhatminus)
        assert(result.ndim == 2)
        assert(result.shape[1] == 1)
        return result[:,0] # drop dimension

    def step2__calculate_a_posteri(self, xhatminus, Pminus, y=None,
                                   full_output=False):

        self.C = linearize_at(self.observation_function, xhatminus, delta=self.delta)
        self.CT = self.C.T

        return KalmanFilter.step2__calculate_a_posteri(
            self, xhatminus=xhatminus, Pminus=Pminus, y=y, full_output=full_output)

def linearize_at(f, x, delta = 1e-5):
    f0 = f(x)
    mat = []
    for i in range(len(x)):
        dxi = numpy.zeros((len(x),))
        dxi[i] = delta
        fi = f(x+dxi)
        dF_dxi = (fi-f0)/delta
        mat.append(dF_dxi[:,0])
    mat = numpy.array(mat).T
    return mat

def kalman_filter(y, A, C, Q, R, init_x, init_V, full_output=False):
    T = len(y)
    ss = len(A)
    R = numpy.array(R)

    for arr in (A, C, Q):
        if numpy.any(numpy.isnan(arr)):
            raise ValueError(
                "cannot do Kalman filtering with nan values in parameters")

    if R.ndim not in (2, 3):
        raise ValueError("R not 2 or 3 dimensions but %d" % R.ndim)

    if R.ndim == 2:
        kfilt = KalmanFilter(A, C, Q, R, init_x, init_V)
    else:
        assert R.ndim == 3
        if R.shape[0] != T:
            raise ValueError(
                'Per-observation noise must have same length as observations')
        kfilt = VariableObservationNoiseKalmanFilter(A, C, Q, init_x, init_V)

    # Forward pass
    xfilt = numpy.zeros((T, ss))
    Vfilt = numpy.zeros((T, ss, ss))
    if full_output:
        VVfilt = numpy.zeros((T, ss, ss))
    loglik = 0

    for i in range(T):
        isinitial = i == 0
        y_i = y[i]
        if R.ndim == 3:
            kw = dict(R=R[i])
        else:
            kw = {}

        if full_output:
            xfilt_i, Vfilt_i, LL, VVfilt_i = kfilt.step(y=y_i,
                                                        isinitial=isinitial,
                                                        full_output=True, **kw)
            VVfilt[i] = VVfilt_i
            loglik += LL
        else:
            xfilt_i, Vfilt_i = kfilt.step(y=y_i,
                                          isinitial=isinitial,
                                          full_output=False, **kw)
        xfilt[i] = xfilt_i
        Vfilt[i] = Vfilt_i

    if full_output:
        return xfilt, Vfilt, VVfilt, loglik
    else:
        return xfilt, Vfilt


def kalman_smoother(y, A, C, Q, R, init_x, init_V, valid_data_idx=None,
                    full_output=False):
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
    valid_data_idx - (optional) Indices to rows of y that are valid or
        boolean array of len(y). (None if all data valid.)  Note that
        this is not necessary if y is nan where data are invalid.

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

    if valid_data_idx is not None:
        if hasattr(valid_data_idx, 'dtype') and valid_data_idx.dtype == numpy.bool:
            assert len(valid_data_idx) == len(y)
            invalid_cond = ~valid_data_idx
            y[invalid_cond] = numpy.nan  # broadcast
        else:
            y = numpy.array(y, copy=True)
            valid_data_idx = set(valid_data_idx)
            all_idx = set(range(len(y)))
            bad_idx = list(all_idx - valid_data_idx)
            for i in bad_idx:
                y[i] = numpy.nan  # broadcast

    def smooth_update(xsmooth_future, Vsmooth_future, xfilt, Vfilt,
                      Vfilt_future, VVfilt_future, A, Q, full_output=False):
        dot = numpy.dot
        inv = numpy.linalg.inv

        xpred = dot(A, xfilt)
        Vpred = dot(A, dot(Vfilt, A.T)) + Q
        J = dot(Vfilt, dot(A.T, inv(Vpred)))  # smoother gain matrix
        xsmooth = xfilt + dot(J, xsmooth_future - xpred)
        Vsmooth = Vfilt + dot(J, dot(Vsmooth_future - Vpred, J.T))
        VVsmooth_future = VVfilt_future + dot(
            (Vsmooth_future - Vfilt_future),
            dot(inv(Vfilt_future), VVfilt_future))
        return xsmooth, Vsmooth, VVsmooth_future

    T = len(y)
    ss = len(A)

    # Forward pass
    forward_results = kalman_filter(y, A, C, Q, R, init_x, init_V,
                                    full_output=full_output)
    if full_output:
        xfilt, Vfilt, VVfilt, loglik = forward_results
    else:
        xfilt, Vfilt = forward_results
        VVfilt = Vfilt  # dummy value

    # Backward pass
    xsmooth = numpy.array(xfilt, copy=True)
    Vsmooth = numpy.array(Vfilt, copy=True)
    VVsmooth = numpy.empty(Vfilt.shape)

    for t in range(T - 2, -1, -1):
        xsmooth_t, Vsmooth_t, VVsmooth_t = smooth_update(xsmooth[t + 1, :],
                                                         Vsmooth[t + 1, :, :],
                                                         xfilt[t, :],
                                                         Vfilt[t, :, :],
                                                         Vfilt[t + 1, :, :],
                                                         VVfilt[t + 1, :, :],
                                                         A, Q,
                                                         full_output=full_output)
        xsmooth[t, :] = xsmooth_t
        Vsmooth[t, :, :] = Vsmooth_t
        VVsmooth[t + 1, :, :] = VVsmooth_t

    VVsmooth[0, :, :] = numpy.zeros((ss, ss))

    if full_output:
        return xsmooth, Vsmooth, VVsmooth, loglik
    else:
        return xsmooth, Vsmooth
