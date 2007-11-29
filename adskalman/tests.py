import unittest
import adskalman
import numpy
import scipy.io
import scipy.stats
import matplotlib.mlab
import pkg_resources

def assert_3d_vs_kpm_close(A,B,debug=False):
    """compare my matrices (where T dimension is first) vs. KPM's (where it's last)"""
    for i in range(A.shape[0]):
        if debug:
            print
            print 'i=%d'%i
            print 'A[i],',A[i].shape,A[i]
            print "B[:,:,i]",B[:,:,i].shape,B[:,:,i]
            diff = A[i]-B[:,:,i]
            print 'diff',diff
        try:
            assert numpy.allclose( A[i], B[:,:,i])
            if debug:
                print '(same)'
        except Exception,err:
            if debug:
                print '(different)'
            raise

class TestStats(unittest.TestCase):
    def test1(self):
        pi = numpy.pi
        exp = numpy.exp

        x=numpy.array([0,0])
        m=numpy.array([[0,0]])
        C=numpy.array([[1.0,0],[0,1]])
        lik = adskalman.gaussian_prob(x,m,C)
        lik_should = 1/(2*pi) * 1 * exp( 0 )
        assert numpy.allclose(lik,lik_should)

##         #lik2 = scipy.stats.norm.pdf(x,loc=m,scale=C)
##         lik2 = densities.gauss_den(x,m,C)
##         print
##         print 'lik',lik
##         print 'lik_should',lik_should
##         print 'lik2',lik2

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

            xs.append(x)
            xhats.append( xhat )
        xs = numpy.array(xs)
        xhats = numpy.array(xhats)
        # XXX some comparison between xs and xhats

    def test_filt_KPM(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_results'))
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

        xfilt, Vfilt = adskalman.kalman_filter(y, A, C, Q, R, initx, initV)
        assert numpy.allclose(xfilt.T,kpm['xfilt'])
        assert_3d_vs_kpm_close(Vfilt,kpm['Vfilt'])

        xsmooth, Vsmooth = adskalman.kalman_smoother(y,A,C,Q,R,initx,initV)
        assert numpy.allclose(xsmooth.T,kpm['xsmooth'])
        assert_3d_vs_kpm_close(Vsmooth,kpm['Vsmooth'])

    def test_filt_KPM_loglik(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_results'))
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

        xfilt, Vfilt, VVfilt, loglik = adskalman.kalman_filter(y, A, C, Q, R, initx, initV,
                                                               full_output=True)
        assert numpy.allclose(xfilt.T,kpm['xfilt'])
        assert numpy.allclose(Vfilt.T,kpm['Vfilt'])
        assert_3d_vs_kpm_close(Vfilt,kpm['Vfilt'])
        assert_3d_vs_kpm_close(VVfilt,kpm['VVfilt'])
        assert numpy.allclose(loglik,kpm['loglik'])

        xsmooth, Vsmooth, VVsmooth, loglik = adskalman.kalman_smoother(y,A,C,Q,R,initx,initV,
                                                                       full_output=True)
        assert numpy.allclose(xsmooth.T,kpm['xsmooth'])
        assert_3d_vs_kpm_close(Vsmooth,kpm['Vsmooth'])
        assert_3d_vs_kpm_close(VVsmooth,kpm['VVsmooth'])
        assert numpy.allclose(loglik,kpm['loglik_smooth'])

    def test_DRO_smooth(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_results'))
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

        xfilt, Vfilt = adskalman.DROsmooth(y,A,C,Q,R,initx,initV,mode='forward_only')
        if 0:
            xfilt_kpm, Vfilt_kpm = adskalman.kalman_filter(y, A, C, Q, R, initx, initV)
            print 'xfilt.T',xfilt.T
            print 'xfilt_kpm.T',xfilt_kpm.T
            print "kpm['xfilt']",kpm['xfilt']
        assert numpy.allclose(xfilt.T,kpm['xfilt'])
        assert_3d_vs_kpm_close(Vfilt,kpm['Vfilt'])

        xsmooth, Vsmooth = adskalman.DROsmooth(y,A,C,Q,R,initx,initV)
        if 0:
            xsmooth_kpm, Vsmooth_kpm = adskalman.kalman_smoother(y,A,C,Q,R,initx,initV)
            print 'xsmooth.T',xsmooth.T
            print 'xsmooth_kpm.T',xsmooth_kpm.T
            print "kpm['xsmooth']",kpm['xsmooth']

            print 'Vsmooth',Vsmooth
        assert numpy.allclose(xsmooth.T[:,:-1],kpm['xsmooth'][:,:-1]) # KPM doesn't update last timestep
        assert_3d_vs_kpm_close(Vsmooth[:-1],kpm['Vsmooth'][:,:,:-1])
        #assert numpy.allclose(xsmooth.T,kpm['xsmooth'])
        #assert_3d_vs_kpm_close(Vsmooth,kpm['Vsmooth'])

        xsmooth, Vsmooth, F, H, Q, R = adskalman.DROsmooth(y,A,C,Q,R,initx,initV,mode='EM')

    def test_learn_missing_DRO_nan(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_learn_results'))

        y = kpm['y'].T # data vector is transposed from KPM
        F1 = kpm['F1']
        H1 = kpm['H1']
        Q1 = kpm['Q1']
        R1 = kpm['R1']
        initx1 = kpm['initx']
        initV1 = kpm['initV']
        max_iter = kpm['max_iter']
        T,os = y.shape
        if 1:
            if T>2:
                y[2,:] = numpy.nan * numpy.ones( (os,) )
            if T>12:
                y[12,:] = numpy.nan * numpy.ones( (os,) )
        xsmooth, Vsmooth, F2, H2, Q2, R2 = adskalman.DROsmooth(y,F1, H1, Q1, R1, initx1, initV1,mode='EM',EM_max_iter=max_iter)
        #F2_kpm, H2, Q2, R2, initx2, initV2, LL = adskalman.learn_kalman(y, F1, H1, Q1, R1, initx1, initV1, max_iter)
        #print 'F2_kpm',kpm['F2']
        #print 'F2',F2

    def test_smooth_missing(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_learn_results'))

        y = kpm['y'].T # data vector is transposed from KPM
        F1 = kpm['F1']
        H1 = kpm['H1']
        Q1 = kpm['Q1']
        R1 = kpm['R1']
        initx1 = kpm['initx']
        initV1 = kpm['initV']

        T,os = y.shape
        # build data with missing observations specified by nan
        if T>2:
            y[2,:] = numpy.nan * numpy.ones( (os,) )
        if T>12:
            y[12,:] = numpy.nan * numpy.ones( (os,) )

        # build data with missing observations specified by None
        y_none = []
        for yy in y:
            if numpy.any( numpy.isnan( yy ) ):
                y_none.append( None )
            else:
                y_none.append( yy )

        # get results
        xsmooth_nan, Vsmooth_nan = adskalman.kalman_smoother(y,F1,H1,Q1,R1,initx1,initV1)
        xsmooth_none, Vsmooth_none = adskalman.kalman_smoother(y_none,F1,H1,Q1,R1,initx1,initV1)
        # compare
        assert numpy.allclose(xsmooth_nan, xsmooth_none)
        assert numpy.allclose(Vsmooth_nan, Vsmooth_none)

    def test_learn_missing_nan(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_learn_results'))

        y = kpm['y'].T # data vector is transposed from KPM
        F1 = kpm['F1']
        H1 = kpm['H1']
        Q1 = kpm['Q1']
        R1 = kpm['R1']
        initx1 = kpm['initx']
        initV1 = kpm['initV']
        max_iter = kpm['max_iter']
        T,os = y.shape
        if T>2:
            y[2,:] = numpy.nan * numpy.ones( (os,) )
        if T>12:
            y[12,:] = numpy.nan * numpy.ones( (os,) )
        F2, H2, Q2, R2, initx2, initV2, LL = adskalman.learn_kalman(y, F1, H1, Q1, R1, initx1, initV1, max_iter)

    def test_learn_KPM(self):
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_learn_results'))

        y = kpm['y'].T # data vector is transposed from KPM
        F1 = kpm['F1']
        H1 = kpm['H1']
        Q1 = kpm['Q1']
        R1 = kpm['R1']
        initx1 = kpm['initx']
        initV1 = kpm['initV']
        max_iter = kpm['max_iter']
        F2, H2, Q2, R2, initx2, initV2, LL = adskalman.learn_kalman(y, F1, H1, Q1, R1, initx1, initV1, max_iter)
        assert numpy.allclose(F2,kpm['F2'])
        assert numpy.allclose(H2,kpm['H2'])
        assert numpy.allclose(Q2,kpm['Q2'])
        assert numpy.allclose(R2,kpm['R2'])
        assert numpy.allclose(initx2,kpm['initx2'])
        assert numpy.allclose(initV2,kpm['initV2'])
        #print numpy.ravel(LL),kpm['LL']
        assert numpy.allclose(LL,kpm['LL'])

    def test_loglik_KPM(self):
        # this test broke the loglik calculation
        kpm=scipy.io.loadmat(pkg_resources.resource_filename(__name__,'kpm_learn_results'))
        y = kpm['y'].T # data vector is transposed from KPM
        F1 = kpm['F1']
        H1 = kpm['H1']
        Q1 = kpm['Q1']
        R1 = kpm['R1']
        initx1 = kpm['initx']
        initV1 = kpm['initV']
        xfilt, Vfilt, VVfilt, loglik_filt =  adskalman.kalman_filter(
            y, F1, H1, Q1, R1, initx1, initV1,full_output=True)
        assert numpy.allclose(xfilt, kpm['xfilt'].T)
        assert_3d_vs_kpm_close(Vfilt, kpm['Vfilt'])
        assert_3d_vs_kpm_close(VVfilt, kpm['VVfilt'])
        assert numpy.allclose(loglik_filt, kpm['loglik_filt'])

        xsmooth, Vsmooth, VVsmooth, loglik_smooth =  adskalman.kalman_smoother(
            y, F1, H1, Q1, R1, initx1, initV1,full_output=True)
        assert numpy.allclose(xsmooth, kpm['xsmooth'].T)
        assert_3d_vs_kpm_close(Vsmooth, kpm['Vsmooth'])
        assert_3d_vs_kpm_close(VVsmooth, kpm['VVsmooth'])
        assert numpy.allclose(loglik_smooth, kpm['loglik_smooth'])

    def test_DRO_on_Shumway_Stoffer(self):
        fname = pkg_resources.resource_filename(__name__,'table1.csv')
        table1 = matplotlib.mlab.csv2rec(fname)

        fname = pkg_resources.resource_filename(__name__,'table2.csv')
        table2 = matplotlib.mlab.csv2rec(fname)

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestKalman),
                           unittest.makeSuite(TestStats),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()
