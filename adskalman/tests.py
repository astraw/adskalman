import unittest
import adskalman
import numpy
import scipy.io

class TestKalman(unittest.TestCase):
    def test_kalman1(self,time_steps=100,Qsigma=0.1,Rsigma=0.5):
        dt = 0.1
        # process model
        A = numpy.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]],
                        dtype=numpy.float64)
        # observation model
        C = numpy.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]],
                        dtype=numpy.float64)
        # process covariance
        Q = Qsigma*numpy.eye(4)
        # measurement covariance
        R = Rsigma*numpy.eye(2)
        
        x = numpy.array([0,0,0,0])
        x += Qsigma*numpy.random.standard_normal(x.shape)
        
        kf = adskalman.KalmanFilter(A,C,Q,R,x,Q)
        y = numpy.dot(C,x)
        y += Rsigma*numpy.random.standard_normal(y.shape)

        xs = []
        xhats = []
        for i in range(time_steps):
            if i==0: isinitial=True
            else: isinitial = False
            
            xhat,P = kf.step(y=y, isinitial=isinitial)

            # calculate new state
            x = numpy.dot(A,x) + Qsigma*numpy.random.standard_normal(x.shape)
            # and new observation
            y = numpy.dot(C,x) + Rsigma*numpy.random.standard_normal(y.shape)
        
            #print y,xhat
            xs.append(x)
            xhats.append( xhat )
        xs = numpy.array(xs)
        xhats = numpy.array(xhats)
        # XXX some comparison between xs and xhats
        
    def test_filt_KPM(self):
        kpm=scipy.io.loadmat('kpm_results')
        # process model
        A = numpy.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]],
                        dtype=numpy.float64)
        # observation model
        C = numpy.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]],
                        dtype=numpy.float64)
        ss=4; os=2
        # process covariance
        Q = 0.1*numpy.eye(ss)
        # measurement covariance
        R = 1.0*numpy.eye(os)
        initx = numpy.array([10, 10, 1, 0],dtype=numpy.float64)
        initV = 10.0*numpy.eye(ss)

        x = kpm['x'].T
        y = kpm['y'].T

        
        import warnings
        warnings.warn("not testing loglik implementation")
        # Because the code paths differ slightly, test both full_output conditions.

##        xfilt, Vfilt, VVfilt, loglik = adskalman.kalman_filter(y, A, C, Q, R, initx, initV,
##                                                               full_output=True)
        xfilt, Vfilt = adskalman.kalman_filter(y, A, C, Q, R, initx, initV)
        assert numpy.allclose(xfilt.T,kpm['xfilt'])
        assert numpy.allclose(Vfilt.T,kpm['Vfilt'])

        xsmooth, Vsmooth = adskalman.kalman_smoother(y,A,C,Q,R,initx,initV)
        assert numpy.allclose(xsmooth.T,kpm['xsmooth'])
        assert numpy.allclose(Vsmooth.T,kpm['Vsmooth'])

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestKalman),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()
