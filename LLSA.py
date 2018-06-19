import numpy as np
import numpy.ma as ma
#Ignore warnings
import warnings
#cython script with main functions
import LLSA_calculations as lvar_c

np.seterr(divide='ignore')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



def test(window1,window2,theta_1,theta_2,N,per,lag=1):
    #Assume theta_1 is the null model fitting window2
    r_1 = lvar_c.loglik_mvn(theta_2,window2) - lvar_c.loglik_mvn(theta_1,window2) #lik ratio between theta_2 and theta_1 modelling window2
    Rnull_1=lvar_c.R_null(theta_1,window1,window2,N,lag) #lik ratio dist with theta_1 as the null model
    thresh_1_max=np.nanpercentile(Rnull_1,per) #threshold lik ratio from per-th percentile of null distribution
    is_in_1=r_1<thresh_1_max #check whether r_1 falls inside the distribution
    if is_in_1:
        return False
    else:
        return True

def r_window(tseries,t,windows,N,per,lag=1,cond_thresh=1e6):
    '''
    Returns the break found after t, iterating over a set of candidate windows
    tseries is the time series w5 want to segment
    windows is the set of candidate windows
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    for i in range(len(windows)-1):
        window1=np.array(tseries[t:t+windows[i]])
        window2=np.array(tseries[t:t+windows[i+1]])
        #fit theta_1 and theta_2 to window1 and window2
        theta_1,eps1 = lvar_c.get_theta(window1,lag)
        theta_2,eps2 = lvar_c.get_theta(window2,lag)
        #decompose theta into its components
        c1,A1,cov1 = lvar_c.decomposed_theta(theta_1)
        c2,A2,cov2 = lvar_c.decomposed_theta(theta_2)
        #compute the condition number of the covariance
        cond1=np.linalg.cond(cov1)
        cond2=np.linalg.cond(cov2)
        #if the condition number of the covariance matrices is above cond_thresh, increase window size
        if cond1>cond_thresh or cond2>cond_thresh:
            continue
        else:
            #test whether there's a break between window1 and window2
            if test(window1,window2,theta_1,theta_2,N,per,lag):
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
                theta_1,eps1= lvar_c.get_theta(w1,lag)
                theta_2,eps2= lvar_c.get_theta(w2,lag)
                c1,A1,cov1= lvar_c.decomposed_theta(theta_1)
                c2,A2,cov2= lvar_c.decomposed_theta(theta_2)
                cond1=np.linalg.cond(cov1)
                cond2=np.linalg.cond(cov2)
                if cond1>cond_thresh or cond2>cond_thresh:
                    continue
                else:
                    first_test=test(w1,w2,theta_1,theta_2,N,per,lag)
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
