import sys
import SPCR_final as lvar
import SPCR_calculations_final as lvarc
import numpy as np
import numpy.ma as ma
import time as T
import h5py
import argparse

def timeStr(Time):
	ss=(Time%60)
	Time=Time/60
	mm=(Time%60)
	Time=Time/60
	hh=Time%24;
	return "{:0=2.0f}-{:0=2.0f}-{:0=2.0f}".format(hh,mm,ss);



def setupMetaData(f,w,N,per,min_size):
	meta_data=f.create_group('MetaData')
	c_windows=meta_data.create_dataset('windows',(len(w),))
	c_windows[...]=w
	n=meta_data.create_dataset('SizeNullDistribution',(1,))
	n[...]=N
	percent=meta_data.create_dataset('percentile',(1,))
	percent[...]=per
	min_s=meta_data.create_dataset('min_size',(1,))
	min_s[...]=min_size
	return 'Meta data is set up'



def main(argv):
    batch_start=T.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--N',help="Number of observations in the null distribution",default=1000,type=int)
    parser.add_argument('-s','--step',help="Step fraction",default=.1,type=float)
    parser.add_argument('-segment','--Seg',help='segment',default=1,type=int)
    parser.add_argument('-per','--Per',help='percentile',default=80.,type=float)
    parser.add_argument('-min_s','--min_size',help='minimum size',default=50,type=int)
    args=parser.parse_args()

    segment=args.Seg
    #extract time series
    f=h5py.File('AllenData_'+str(segment)+'.h5','r')
    tseries=ma.array(f['tseries'],dtype=np.float64)
    tseries=ma.masked_invalid(tseries)
    f.close()
    w0=10
    #Define candidate windows
    wmax=np.inf
    step_fraction=args.step
    i=w0
    w=[]
    while i<wmax:
    	w.append(i)
    	step=int(i*step_fraction)
    	if int(i*step_fraction)>w0:
    		break
    	if step<1:
    		step=1
    	i+=step

    N=args.N #number of samples in the null distribution
    per=args.Per
    lag=1
    print(N,per)

    #break finder
    min_size=args.min_size
    segs=lvar.segment_maskedArray(tseries,min_size)
    breaks_segments=lvar.change_point(w,N,per,tseries,min_size,cond_thresh=1e5)
    #save breaks and respective thetas
    windows_segment,segments=breaks_segments
    windows_f=np.concatenate(windows_segment)
    f=h5py.File("SegmentationAllen_"+str(segment)+".h5","w")
    setupMetaData(f,w,N,per,min_size)
    windows_seg=f.create_dataset('windows',windows_f.shape,dtype=int)
    windows_seg[...]=windows_f
    f.close()
    batch_end=T.time()
    print('Start time: '+str(timeStr(batch_start)))
    print('End time: '+str(timeStr(batch_end)))
    print('Total time: '+str(timeStr(batch_end-batch_start)))

if __name__ == "__main__":
    main(sys.argv)
