import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport M_PI,log
from cython.parallel import prange
import numpy.ma as ma

cpdef decomposed_theta(np.ndarray[np.float64_t, ndim=2] theta, int lag=1):
    '''
    Decomposed theta into the mean, inter, coupling matrix, coef, and covariance matrix of the error, cov
    '''
    cdef int clag = lag
    cdef int dim = theta.shape[1]
    cdef double[:] inter = theta[0]
    cdef double[:, :] coef = theta[1:clag*dim+1]
    cdef double[:, :] cov = theta[clag*dim+1:]
    return inter,coef,cov

cpdef pseudo_det(np.ndarray[np.float64_t, ndim=2] X, thresh = 1e-15):
    '''
    Estimates the pseudo-determinant of X
    '''
    eig_values = np.linalg.eig(X)[0]
    if np.all(eig_values<thresh):
        return False
    cdef double eig_product = np.product(eig_values[eig_values>thresh]).real
    return eig_product


cpdef get_yp(np.ndarray[np.float64_t, ndim=2] y, int lag=1):
    cdef int clag = lag
    cdef np.ndarray[np.float64_t, ndim=2] yp
    if clag<2:
        yp = y[0:-clag]
    else:
        ypredictor=[y[clag-p:-p] for p in range(1,clag+1)]
        yp = np.hstack(ypredictor)
    return yp




cpdef get_theta(np.ndarray[np.float64_t, ndim=2] y, int lag=1):
    '''
    Fits a AR(lag) model to y
    '''
    cdef int clag = lag
    cdef np.ndarray[np.float64_t, ndim=2] yf = y[clag:]
    cdef np.ndarray[np.float64_t, ndim=2] yp = get_yp(y,clag)
    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]

    cdef int yp_len = yp.shape[0]
    cdef int yp_dim = yp.shape[1]
    cdef double[:, :] x_inter = np.vstack((np.ones(yp_len),yp.T)).T
    cdef double[:, :] pinverse = np.linalg.pinv(x_inter)
    cdef double[:, :] beta = np.dot(pinverse,yf)
    cdef double[:, :] pred = np.dot(x_inter,beta)
    eps = ma.zeros((y_len,y_dim))
    eps[clag:] = yf-pred
    eps[:clag] = ma.masked
    cdef double[:, :] cov = np.cov(eps[clag:].T)
    cdef double[:, :] theta = np.vstack((beta,cov))
    return np.array(theta,dtype=np.float64),eps

cpdef get_error(np.ndarray[np.float64_t, ndim=2] theta,np.ndarray[np.float64_t, ndim=2] y,int lag=1):
    '''
    Returns the error of a AR(lag) fit to y
    '''
    cdef int T = y.shape[0]
    cdef int dim = y.shape[1]
    cdef int clag = lag
    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]

    cdef double[:, :] sigmainv
    cdef double[:, :] x_inter
    cdef double[:, :] beta


    inter,coef,sigma=decomposed_theta(theta,clag)

    yf = y[clag:]
    yp = get_yp(y,clag)
    #get prediction
    beta=np.vstack((inter,coef))
    x_inter=np.vstack((np.ones(yp.shape[0]),yp.T)).T
    pred = np.dot(x_inter,beta)
    eps = ma.zeros((y_len,y_dim))
    eps[:clag] = ma.masked
    eps[clag:] = yf-pred

    return eps


cpdef get_pred(np.ndarray[np.float64_t, ndim=2] theta,np.ndarray[np.float64_t, ndim=2] y,int lag=1):
    '''
    Returns the prediction, y^hat, of the AR(lag) model, theta, found in y
    '''
    cdef int T = y.shape[0]
    cdef int dim = y.shape[1]
    cdef int clag = lag
    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]

    inter,coef,sigma=decomposed_theta(theta,clag)

    cdef np.ndarray[np.float64_t, ndim=2] yf = y[clag:]
    cdef np.ndarray[np.float64_t, ndim=2] yp = get_yp(y,clag)
    #get prediction
    cdef np.ndarray[np.float64_t, ndim=2] beta = np.vstack((inter,coef))
    cdef np.ndarray[np.float64_t, ndim=2] x_inter = np.vstack((np.ones(yp.shape[0]),yp.T)).T
    cdef np.ndarray[np.float64_t, ndim=2] pred = np.zeros((y_len,y_dim),dtype=np.float64)
    pred[clag:] = np.dot(x_inter,beta)
    pred[:clag] = y[:clag]

    return pred


cpdef loglik_mvn(np.ndarray[np.float64_t, ndim=2] theta, np.ndarray[np.float64_t, ndim=2] y,int lag=1):
    '''
    Computes the loglikelihood of the parameters theta fitting y
    '''

    cdef int T = y.shape[0]
    cdef int dim = y.shape[1]
    cdef int clag = lag

    inter,coef,sigma=decomposed_theta(theta,clag)
    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] yf = y[clag:]
    cdef np.ndarray[np.float64_t, ndim=2] yp = get_yp(y,clag)

    cdef int yp_len = yp.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] sigmainv=np.linalg.pinv(sigma)
    sign,value = np.linalg.slogdet(sigmainv)
    cdef double abs_log_det_inv_s = sign*value
    # cdef double abs_log_det_inv_s = np.abs(det_inv_s)

    cdef np.ndarray[np.float64_t, ndim=2] beta = np.vstack((inter,coef))
    cdef np.ndarray[np.float64_t, ndim=2] x_inter = np.vstack((np.ones(yp_len),yp.T)).T
    cdef np.ndarray[np.float64_t, ndim=2] pred = np.dot(x_inter,beta)
    cdef np.ndarray[np.float64_t, ndim=2] eps = yf-pred
    cdef int eps_len = eps.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] element_sum = np.zeros(eps_len,dtype=np.float64)
    cdef int[:] unmasked_time = np.arange(0,eps_len,dtype=np.dtype("i"))
    cdef int t
    cdef double last_term = 0
    for t in unmasked_time:
        eps_t=eps[t]
        last_term+=np.dot(eps_t.T,np.dot(sigmainv,eps_t))


    return (-0.5*dim*(T-1)*log(2*M_PI)
            +0.5*(T-1)*abs_log_det_inv_s
            -0.5*last_term)


cpdef gen_obs(np.ndarray[np.float64_t, ndim=2] theta,np.ndarray[np.float64_t, ndim=2] y,int lag=1):
    '''
    Returns a simulation of an AR process
    theta are the parameters of the model
    y is the interval over which we want to simulate
    '''
    cdef int clag = lag
    cdef int dim = y.shape[1]
    cdef int size_y = y.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] sim = np.zeros((size_y,dim),dtype=np.float64)
    #decompose theta
    inter,coef,cov=decomposed_theta(theta,clag)
    #draw an error vector with the given covariance and mean 0
    cdef np.ndarray[np.float64_t, ndim=2] eps = np.random.multivariate_normal(np.zeros(dim),cov,size_y)
    #define the initial condition for the simulation
    sim[:clag]=y[:clag]
    cdef int i
    for i in range(clag,len(sim)):
        sim[i,:]=inter+np.sum([np.dot(sim[i-p-1,:],coef[p*dim:(p+1)*dim]) for p in range(clag)],axis=0)+eps[i]
    return sim


cpdef R_null(np.ndarray[np.float64_t, ndim=2] theta,np.ndarray[np.float64_t, ndim=2] window1,np.ndarray[np.float64_t, ndim=2] window2,int N,int lag):
    '''
    Returns the likelihood distribution
    theta is the null model
    '''

    cdef int cN = N
    cdef int clag = lag
    cdef double[:] Rs = np.zeros(N)
    cdef int i
    for i in range(cN):
        #simulate a time series of the size of window2 using the null model, theta
        obs=gen_obs(theta,window2,clag)
        #fit a linear model to a window of size window1 and window2 in the simulation
        theta_g1,eps_g1=get_theta(obs[:len(window1)],clag)
        theta_g2,eps_g2=get_theta(obs,clag)
        #obtain the respective likelihood ratios
        loglik1=loglik_mvn(theta_g1,obs)
        loglik2=loglik_mvn(theta_g2,obs)
        r=loglik2-loglik1
        Rs[i] = r
    return Rs


#dealing with masked data

def get_yp_masked(y,lag=1):
    if lag<2:
        yp=y[0:-lag]
    else:
        ypredictor=[y[lag-p:-p] for p in range(1,lag+1)]
        yp=ma.hstack(ypredictor)
    return yp


def get_theta_masked(y,lag=1):
    cdef int clag = lag

    yf = y[clag:]
    yp = get_yp_masked(y,clag)
    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]
    cdef int yp_len
    cdef int yp_dim

    cdef double[:, :] x_inter
    cdef double[:, :] pinverse
    cdef double[:, :] beta
    cdef double[:, :] theta

    sel = ~np.logical_or(np.any(yp.mask, axis=1), np.any(yf.mask, axis=1))
    yp_len = yp[sel,:].shape[0]
    yp_dim = yp[sel,:].shape[1]

    x_inter=np.vstack((np.ones(yp_len),yp[sel,:].T)).T
    pinverse=np.linalg.pinv(x_inter)
    beta=np.dot(pinverse,yf[sel,:])
    pred=ma.zeros((yf.shape[0],yf.shape[1]))
    pred[sel,:]=np.dot(x_inter,beta)
    pred[~sel]=ma.masked
    pred=ma.vstack((y[:clag],pred))
    eps=y-pred
    eps[:clag]=ma.masked
    cov=ma.cov(eps.T)
    theta=ma.vstack((beta,cov))
    return np.array(theta),eps

def get_pred_masked(theta,y,lag=1):
    '''
    Computes the loglikelihood of the parameters theta fitting
    the interval in which eps was calculated, that starts with x1
    '''
    #eps is the error computed using coef, inter, Sigma over the whole interval

    cdef int T = y.shape[0]
    cdef int dim = y.shape[1]
    cdef double cond
    cdef double det_s
    cdef int clag = lag
    inter,coef,sigma=decomposed_theta(theta,clag)

    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]
    cdef int yp_len
    cdef int yp_dim
    cdef int t
    cdef double last_term
    cdef double abs_det

    cdef double[:, :] sigmainv
    cdef double[:, :] x_inter
    cdef double[:, :] beta
    cdef double[:] element_sum
    cdef int[:] unmasked_time

    yf = y[clag:]
    yp = get_yp_masked(y,clag)
    sigma=np.array(sigma,dtype=np.float64)
    sigmainv=np.array(np.linalg.pinv(sigma),dtype=np.float64)
    det_inv_s=pseudo_det(np.array(sigmainv,dtype=np.float64))
    sel = ~np.logical_or(np.any(yp.mask, axis=1), np.any(yf.mask, axis=1))
    yp_len = yp[sel,:].shape[0]
    yp_dim = yp[sel,:].shape[1]

    x_inter=np.vstack((np.ones(yp_len),yp[sel,:].T)).T
    beta=np.vstack((inter,coef))
    pred=ma.zeros((yf.shape[0],yf.shape[1]))
    pred[sel,:]=np.dot(x_inter,beta)
    pred[~sel]=ma.masked
    pred=ma.vstack((y[:clag],pred))
    return pred


def loglik_mvn_masked(theta,y,lag=1):
    '''
    Computes the loglikelihood of the parameters theta fitting
    the interval in which eps was calculated, that starts with x1
    '''
    #eps is the error computed using coef, inter, Sigma over the whole interval

    cdef int T = y.shape[0]
    cdef int dim = y.shape[1]
    cdef double cond
    cdef double det_s
    cdef int clag = lag
    inter,coef,sigma=decomposed_theta(theta,clag)

    cdef int y_len = y.shape[0]
    cdef int y_dim = y.shape[1]
    cdef int yp_len
    cdef int yp_dim
    cdef int t
    cdef double last_term
    cdef double abs_det

    cdef double[:, :] sigmainv
    cdef double[:, :] x_inter
    cdef double[:, :] beta
    cdef double[:] element_sum
    cdef int[:] unmasked_time

    yf = y[clag:]
    yp = get_yp_masked(y,clag)
    sigma=np.array(sigma,dtype=np.float64)
    sigmainv=np.array(np.linalg.pinv(sigma),dtype=np.float64)
    # det_inv_s=pseudo_det(np.array(sigmainv,dtype=np.float64))
    abs_log_det_inv_s = np.linalg.slogdet(sigmainv)[1]
    # sigma_inv_abs_det = np.abs(det_inv_s)
    sel = ~np.logical_or(np.any(yp.mask, axis=1), np.any(yf.mask, axis=1))
    yp_len = yp[sel,:].shape[0]
    yp_dim = yp[sel,:].shape[1]

    x_inter=np.vstack((np.ones(yp_len),yp[sel,:].T)).T
    beta=np.vstack((inter,coef))
    pred=ma.zeros((yf.shape[0],yf.shape[1]))
    pred[sel,:]=np.dot(x_inter,beta)
    pred[~sel]=ma.masked
    pred=ma.vstack((y[:clag],pred))
    eps=y-pred
    eps[:clag]=ma.masked
    element_sum=np.zeros(eps.shape[0],dtype=np.float64)
    unmasked_time=np.arange(0,eps.shape[0],dtype=np.dtype("i"))[~eps.mask[:,0]]
    for t in unmasked_time:
        eps_t=eps[t]
        element_sum[t]=np.dot(eps_t.T,ma.dot(sigmainv,eps_t))
    last_term=np.sum(element_sum)

    return (-0.5*dim*(T-1)*log(2*M_PI)
            +0.5*(T-1)*abs_log_det_inv_s
            -0.5*last_term)
