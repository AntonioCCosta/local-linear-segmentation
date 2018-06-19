#numpy
import numpy as np
import numpy.ma as ma
import sys
import argparse
import h5py
import LLSA as lvar
import LLSA_calculations as lvarc


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--N',help="Number of observations in the null distribution",default=1000,type=int)
    parser.add_argument('-min_w','--w0',help="Minimum window size candidate",default=10,type=int)
    parser.add_argument('-s','--step',help="Step fraction",default=.1,type=float)
    parser.add_argument('-per','--Per',help='percentile',default=97.5,type=float)
    parser.add_argument('-sim_i','--Sim_i',help='sim_i',default=0,type=int)
    parser.add_argument('-sim_f','--Sim_f',help='sim_f',default=5,type=int)
    args=parser.parse_args()

    #Define candidate windows
    print('Defining the candidate windows...')
    w0=args.w0
    step_fraction=args.step
    i=w0
    w=[]
    while i < np.inf:
    	w.append(i)
    	step=int(i*step_fraction)
    	if int(i*step_fraction)>w0:
    		break
    	if step<1:
    		step=1
    	i+=step
    print('Candidate windows: ', w)

    print('Loading the time series...')
    tseries_path='Sample_tseries.h5'
    f=h5py.File(tseries_path,'r')
    thetas_gen=np.array(f['MetaData/thetas_gen'])
    frameRate=np.array(f['MetaData/FrameRate'])[0]
    duration=np.array(f['MetaData/SimTime'])[0]
    Ns=np.array(f['MetaData/NumSurrogates'],dtype=int)[0]
    taus=np.sort(np.array(list(f.keys())[:-1],dtype=float))

    tau=taus[0]
    tseries_sim=[]
    for iteration in range(1,Ns+1):
        f=h5py.File(tseries_path,'r')
        ts=np.array(f[str(tau)][str(iteration)]['tseries'],dtype=np.float64)
        tseries_sim.append(ma.array(ts))
    f.close()

    N=args.N #number of samples in the null distribution
    per=args.Per
    sim_i=args.Sim_i
    sim_f=args.Sim_f
    lag=1 #first order AR model
    #break finder

    f=h5py.File('SegmentationResults.h5','w')


    min_size=20
    thetas_sim=[]
    windows_sim=[]
    for sim_idx in range(sim_i,sim_f): #this loop can, and should, be parallelized
        tseries=tseries_sim[sim_idx]
        print(tseries.shape)
        print('Finding the breaks for sim '+str(sim_idx)+'...')
        breaks_segments=lvar.change_point(w,N,per,tseries,min_size)
        #save breaks and respective thetas
        windows_segment,segments=breaks_segments
        #compute thetas in the obtained windows_segment
        thetas_final=[]
        for idx,seg in enumerate(segments):
        	segment_windows=np.copy(windows_segment[idx])
        	segments_windows=list(segment_windows)
        	thetas=[]
        	for seg_w in segment_windows:
        		i_0,i_f=seg_w
        		window_bw=tseries[i_0:i_f]
        		theta,eps=lvarc.get_theta(window_bw,lag)
        		thetas.append(np.vstack(theta))
        	thetas_final.append(thetas)
        thetas_final=np.concatenate(thetas_final)
        windows_final=np.concatenate(windows_segment)
        thetas_sim.append(thetas_final)
        windows_sim.append(windows_final)

        print('Breaks found at frames: ',windows_final)

        f_sim=f.create_group(str(sim_idx))
        windows_=f_sim.create_dataset('windows',windows_final.shape)
        thetas_=f_sim.create_dataset('thetas',thetas_final.shape)
        windows_[...]=windows_final
        thetas_[...]=thetas_final
    f.close()

if __name__ == "__main__":
	main(sys.argv)
