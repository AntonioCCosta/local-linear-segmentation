#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import argparse
import sys
import LLSA as lvar
import LLSA_calculations as lvarc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster as cl
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import fcluster

def symmetrize(a): #fix minor computing errors
    return (a + a.T)/2 - np.diag(a.diagonal())


def compute_master_theta(model_indices,all_models,windows_sim,tseries_sim):
    master_tseries=[]
    for model in all_models[model_indices]:
        sim_idx,kw=model
        t0,tf=windows_sim[sim_idx][kw]
        ts=tseries_sim[sim_idx][t0:tf]
        master_tseries.append(ts)
        master_tseries.append([np.nan]*ts.shape[1])
    master_tseries=ma.masked_invalid(ma.vstack(master_tseries))
    master_theta,eps=lvarc.get_theta_masked(master_tseries)
    return master_theta


def likelihood_distance(model_indices,all_models,windows_sim,thetas_sim,tseries_sim):
    master_theta=compute_master_theta(model_indices,all_models,windows_sim,tseries_sim)
    distances=[]
    for model in all_models[model_indices]:
        sim_idx,kw=model
        theta=thetas_sim[sim_idx][kw]
        t0,tf=windows_sim[sim_idx][kw]
        ts=ma.masked_invalid(tseries_sim[sim_idx][t0:tf])
        theta_here,eps=lvarc.get_theta(ts)
        if np.linalg.norm(theta-theta_here)>1e-3:
            print('alarm')
        distances.append(lvarc.loglik_mvn_masked(theta,ts)-lvarc.loglik_mvn_masked(master_theta,ts))
    return np.sum(distances)



def main(argv,):
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelx_i','--Modelx_i',help="First model row",default=0,type=int)
    parser.add_argument('-modelx_f','--Modelx_f',help="Last model row",default=10,type=int)
    parser.add_argument('-modely_i','--Modely_i',help="First model collumn",default=0,type=int)
    parser.add_argument('-modely_f','--Modely_f',help="Last model collumn",default=10,type=int)
    args=parser.parse_args()

    modelx_i=args.Modelx_i
    modelx_f=args.Modelx_f
    modely_i=args.Modely_i
    modely_f=args.Modely_f
    indices_x=[modelx_i,modelx_f]
    indices_y=[modely_i,modely_f]

    print('Loading the time series...')
    print('Loading the time series...')
    tseries_path='Sample_tseries.h5'
    f=h5py.File(tseries_path,'r')
    thetas_gen=np.array(f['MetaData/thetas_gen'],dtype=np.float64)
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

    #get segmentation results
    print('Loading segmentation results...')

    f=h5py.File('SegmentationResults.h5','r')
    sim_indices=np.sort(np.array(list(f.keys()),dtype=int))
    windows_sim=[]
    thetas_sim=[]
    for sim_idx in sim_indices:
        tseries=tseries_sim[sim_idx]
        f_sim=f[str(sim_idx)]
        thetas=np.array(f_sim['thetas'],dtype=np.float64)
        windows=np.array(f_sim['windows'],dtype=int)
        dim=tseries.shape[1]
        windows_sim.append(windows)
        thetas_sim.append(thetas)
    f.close()





    all_models=[]
    for sim_idx in sim_indices:
        windows=windows_sim[sim_idx]
        for kw in range(len(windows)):
            all_models.append([sim_idx,kw])
    all_models=np.array(all_models)

    points_x=np.arange(indices_x[0],indices_x[1])
    points_y=np.arange(indices_y[0],indices_y[1])

    print('Computing distance matrix...')
    distance_matrix=np.zeros((int(np.diff(indices_x)),int(np.diff(indices_x))))
    for idx_x,point_x in enumerate(points_x):
        for idx_y,point_y in enumerate(points_y):
            model_indices=[point_x,point_y]
            distance_matrix[idx_x,idx_y]=likelihood_distance(model_indices,all_models,windows_sim,thetas_sim,tseries_sim)

    indices=np.vstack((indices_x,indices_y))

    f=h5py.File('distance_'+str(indices_x)+'_'+str(indices_y)+'.h5','w')
    dist_mat=f.create_dataset('distance_matrix',distance_matrix.shape)
    indices_=f.create_dataset('indices',indices.shape)
    dist_mat[...]=distance_matrix
    indices_[...]=indices
    f.close()
    plt.title('Distance matrix')
    plt.imshow(distance_matrix,cmap='jet')
    plt.colorbar()
    plt.show()

    sym_distance_matrix=symmetrize(distance_matrix)
    plt.title('Symmetrized distance matrix')
    plt.imshow(sym_distance_matrix,cmap='jet')
    plt.colorbar()
    plt.show()

    #convert to vector of distances
    pdist=ssd.squareform(sym_distance_matrix,force='tovector')

    print('Performing hierarchical clustering...')
    #perform ward hierarchical clustering
    Z=cl.hierarchy.ward(pdist)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    ddgram=dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=12.,
    )
    plt.grid(False)
    plt.show()

    print('getting representative states')
    k = 2
    cluster_labels = fcluster(Z, k, criterion='maxclust')
    cluster_labels
    master_As=[]
    master_times=[]
    for idx in range(1,np.max(cluster_labels)+1):
        indices=np.arange(len(cluster_labels))[cluster_labels==idx]
        times=[]
        for model in all_models[indices]:
            sim_idx,kw=model
            times.append(np.mean(windows_sim[sim_idx][kw]))
            theta=thetas_sim[sim_idx][kw]
        print('State '+str(idx))
        print('Center window: '+str(int(np.median(times)))+' frames.')
        master_theta=compute_master_theta(indices,all_models,windows_sim,tseries_sim)
        c,A,cov=lvarc.decomposed_theta(master_theta)

        master_As.append(A)
        master_times.append(int(np.median(times)))

    ordered_As=np.array(master_As)[np.argsort(master_times)]
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.suptitle('Estimated couplings')
    master_eigs=[]
    for idx,A in enumerate(ordered_As):
        coef=(A-np.identity(2))*frameRate
        master_eigs.append(np.linalg.eigvals(coef))
        ax=axes.flat[idx]
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_title('State '+str(idx+1))
        im=ax.imshow(A,cmap='seismic',vmin=-.8,vmax=.8)
        plt.axis('off')
        plt.grid(False)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.suptitle('Original couplings')
    eigs_gen=[]
    for idx,theta in enumerate(thetas_gen):
        c,A,cov=lvarc.decomposed_theta(theta)
        coef=(A-np.identity(2))*frameRate
        eigs_gen.append(np.linalg.eigvals(coef))
        ax=axes.flat[idx]
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_title('State '+str(idx+1))
        im=ax.imshow(A,cmap='seismic',vmin=-.8,vmax=.8)
        plt.axis('off')
        plt.grid(False)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    colors=['k','b']
    for idx in range(np.max(cluster_labels)):
        eig=master_eigs[idx]
        eig_gen=eigs_gen[idx]
        plt.scatter(eig.real,eig.imag,c=colors[idx],marker='+',label='estimated')
        plt.scatter(eig_gen.real,eig_gen.imag,c=colors[idx],marker='x',label='original')
    plt.axhline(0,alpha=.7,c='k')
    plt.axvline(0,ls='--',c='r')
    plt.xlim(-5,5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
