import numpy as np
import numpy.ma as ma
#Ignore warnings
import warnings
#cython script with main functions
import SPCR_calculations_final as lvar_c
import matplotlib.pyplot as plt

np.seterr(divide='ignore')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



def test(window1,window2,theta_1,theta_2,N,per,lag=1):
    #Assume theta_1 is the null model fitting window2
    r_1 = lvar_c.loglik_mvn(theta_2,window2,lag) - lvar_c.loglik_mvn(theta_1,window2,lag) #lik ratio between theta_2 and theta_1 modelling window2
    Rnull_1=lvar_c.R_null(theta_1,window1,window2,N,lag) #lik ratio dist with theta_1 as the null model
    thresh_1_max=np.percentile(Rnull_1,per) #threshold lik ratio from per-th percentile of null distribution
    is_in_1=r_1<thresh_1_max #check whether r_1 falls inside the distribution
    if is_in_1:
        return False
    else:
        return True

def r_window(tseries,t,windows,N,per,lag=1,cond_thresh=1e5):
    '''
    Returns the break found after t, iterating over a set of candidate windows
    tseries is the time series we want to segment
    windows is the set of candidate windows
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    for i in range(len(windows)-1):
        window1=np.array(tseries[t:t+windows[i]])
        window2=np.array(tseries[t:t+windows[i+1]])
        # print(window1,window2)
        X=window2-window2.mean(axis=0)
        cov=np.cov(X.T)
        eigvals,eigvecs=np.linalg.eig(cov)
        indices=np.argsort(eigvals.real)[::-1]
        eigvals=eigvals.real[indices]
        eigvecs=eigvecs.real[:,indices]
        dim=window2.shape[1]
        theta_1,eps=lvar_c.get_theta(window1)
        c1,A1,cov1=lvar_c.decomposed_theta(theta_1)
        while np.linalg.cond(cov1)>cond_thresh:
            dim-=1
            window1_pca=np.array(window1.dot(eigvecs[:,:dim]),dtype=np.float64)
            window2_pca=np.array(window2.dot(eigvecs[:,:dim]),dtype=np.float64)
            theta_1,eps=lvar_c.get_theta(window1_pca)
            c1,A1,cov1=lvar_c.decomposed_theta(theta_1)
        if dim==1:
            dim=2

        window1_pca=np.array(window1.dot(eigvecs[:,:dim]),dtype=np.float64)
        window2_pca=np.array(window2.dot(eigvecs[:,:dim]),dtype=np.float64)

        theta_1,eps=lvar_c.get_theta(window1_pca)
        theta_2,eps=lvar_c.get_theta(window2_pca)
        if test(window1_pca,window2_pca,theta_1,theta_2,N,per,lag):
            return windows[i]
        else:
            continue
    #if no break is found, return the last candidate window
    return windows[i+1]

def breakfinder(ts,br,w,N,per,lag=1,cond_thresh=1e6):
    '''
    Look around each break to identify weather it was real or artificial
    ts is the time series we which to segment
    br is the set of breaks found using r_window
    w_step is a dictionary containing steps and the respective candidate windows
    defined for the artifical break finder
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    steps=np.unique(np.diff(w))
    w_step={}
    w_step[steps[0]]=w[:np.arange(len(w)-1)[np.diff(w)==steps[0]][-1]+2]
    for i in range(len(steps)-1):
        w_step[steps[i+1]]=w[np.arange(len(w)-1)[np.diff(w)==steps[i]][-1]+1:np.arange(len(w)-1)[np.diff(w)==steps[i+1]][-1]+2]
    for step in w_step.keys():
        br_w=w_step[step]
        min_w=np.min(br_w)
        start=br-min_w
        #this step is probably not needed since br>=min(window_sizes)
        if start<0:
            start=0
        for i in range(len(br_w)-1) :
            w1=ts[start:br]
            w2=ts[start:br+step]
            if ma.count_masked(w1,axis=0)[0]<ma.count_masked(w2,axis=0)[0]:
                return br
            else:
                w1=np.array(w1)
                w2=np.array(w2)
                X=w2-w2.mean(axis=0)
                cov=np.cov(X.T)
                eigvals,eigvecs=np.linalg.eig(cov)
                indices=np.argsort(eigvals.real)[::-1]
                eigvals=eigvals.real[indices]
                eigvecs=eigvecs.real[:,indices]
                dim=window2.shape[1]
                theta_1,eps=lvar_c.get_theta(window1)
                c1,A1,cov1=lvar_c.decomposed_theta(theta_1)
                while np.linalg.cond(cov1)>cond_thresh:
                    dim-=1
                    window1_pca=np.array(w1.dot(eigvecs[:,:dim]),dtype=np.float64)
                    window2_pca=np.array(w2.dot(eigvecs[:,:dim]),dtype=np.float64)
                    theta_1,eps=lvar_c.get_theta(window1_pca)
                    c1,A1,cov1=lvar_c.decomposed_theta(theta_1)
                if dim==1:
                    dim=2
                theta_1,eps=lvar_c.get_theta(window1_pca)
                theta_2,eps=lvar_c.get_theta(window2_pca)
                first_test=test(window1_pca,window2_pca,theta_1,theta_2,N,per,lag)
                if first_test:
                    return br
                else:
                    continue
            start=br-br_w[i+1]
    return False



def segment_maskedArray(tseries,min_size=50):
    '''
    Segments  time series in case it has missing data
    '''
    segments=[]
    t0=0
    tf=1
    while tf<len(tseries):
        #if the first element is masked, increase it
        if np.any(tseries[t0].mask)==True:
            t0+=1
            tf=t0+1
        #if not, check tf
        else:
            #if tf is masked, the segment goes till tf
            if np.any(tseries[tf].mask)==True:
                segments.append([t0,tf])
                t0=tf+1
                tf=t0+1
            #if not increase tf
            else:
                tf+=1
    segments.append([t0,len(tseries)])
    #segments with less than min_size frames are excluded
    i=0
    while i<len(segments):
        t0,tf=segments[i]
        if tf-t0<min_size:
            segments.pop(i)
        else:
            i+=1
    return segments


def change_point(w,N,per,tseries,min_size,lag=1,cond_thresh=1e6):
    '''
    Segments an entire time series
    Returns the breaks found in each of the non masked segments, as well as the non masked segments
    tseries is the time series we which to segment
    w is the set of candidate windows
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    #segment the array into nonmasked segments with at least min_size observations
    segments=segment_maskedArray(tseries,min_size=min_size)
    windows_segment=[]
    for segment in segments:
        t0,tf=segment
        #define a new time series to be segmented
        ts=tseries[t0:tf]
        t=0
        windows=[]
        w_seg=np.copy(w)
        while t<len(ts)-w_seg[0]:
            while t>len(ts)-w_seg[-1]:
                w_seg=np.delete(w_seg,-1)
            if len(w_seg)<2:
                windows.append([t+t0,t0+len(ts)])
                break
            #estimate the first break in ts after t
            k=r_window(ts,t,w_seg,N,per,lag,cond_thresh)
            if k!=False:
                t+=k
                #If the t+min_step is larger than the len of the time series set it
                #to the end of the segment
                if len(ts)-t<=w_seg[0]:
                    #the last interval size is simply the distance from the previous break to
                    #the end of the time series
                    windows.append([t0+t-k,t0+len(ts)])
                else:
                    windows.append([t+t0-k,t+t0])
            else:
                t+=w_seg[0]
        #RemoveArtificialBreaks
        nwindows=list(windows)
        max_intervals=[]
        for i in range(len(windows)-1):
            if np.diff(windows[i])==max(w):
                max_intervals.append([i,windows[i]])
        i=0
        while i < len(max_intervals):
            k,interval=max_intervals[i]
            is_it=breakfinder(tseries,interval[-1],w,N,per,lag,cond_thresh) #last element might be articial
            if is_it!=False: #real break is found
                max_intervals.pop(i)
            else:
                nwindows[k+1][0]=nwindows[k][0]
                nwindows.pop(k)
                if len(max_intervals)>1:
                    for j in range(len(max_intervals[i:])-1):
                        max_intervals[i+j+1][0]=max_intervals[i+j+1][0]-1
                max_intervals.pop(i)
        windows_segment.append(nwindows)
    return windows_segment,segments


def getChangePoints(tseries_w,w,N,per,worm):
    results=[change_point(w,N,per,tseries) for tseries in tseries_w]
    return results

def transform_theta(theta,weigvecs):
    c,A,cov=lvar_c.decomposed_theta(theta)
    c_full=np.dot(weigvecs,c)
    A_full=np.dot(weigvecs,np.dot(A,weigvecs.T))
    cov_full=np.dot(weigvecs,np.dot(cov,weigvecs.T))
    return np.vstack((c_full,A_full,cov_full))

def pca_data(ts,cond_thresh=1e5):
    X=ts-ts.mean(axis=0)
    cov=np.cov(X.T)
    eigvals,eigvecs=np.linalg.eig(cov)
    indices=np.argsort(eigvals.real)[::-1]
    eigvals=eigvals.real[indices]
    eigvecs=eigvecs.real[:,indices]
    theta1,eps=lvar_c.get_theta(ts)
    c1,A1,cov1=lvar_c.decomposed_theta(theta1)
    dim=ts.shape[1]
    while np.linalg.cond(cov1)>cond_thresh:
        dim-=1

        window_pca=np.array(ts.dot(eigvecs[:,:dim]),dtype=np.float64)
        theta1,eps=lvar_c.get_theta(window_pca)
        c1,A1,cov1=lvar_c.decomposed_theta(theta1)
    if dim==1:
        dim=2
    window_pca=np.array(ts.dot(eigvecs[:,:dim]),dtype=np.float64)
    y=window_pca
    return y,eigvecs[:,:dim],ts.mean(axis=0),dim

def pca_theta_coef(ts,frameRate,lag=1,cond_thresh=1e5):
    y,weigvecs,mean,dim=pca_data(ts,cond_thresh)
    theta,eps=lvar_c.get_theta(y,lag)
    full_d_theta=transform_theta(theta,weigvecs)
    c,A,cov=lvar_c.decomposed_theta(theta)
    coef=(A-np.identity(A.shape[1]))*frameRate
    coef_full=np.dot(weigvecs,np.dot(coef,weigvecs.T))
    return full_d_theta,coef_full
