#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt

from joblib                 import Parallel, delayed
from mpl_toolkits.mplot3d   import Axes3D
from scipy.spatial.distance import pdist      
from scipy.spatial.distance import cdist
from pyDOE3                 import ff2n
from pyDOE3                 import fracfact
from scipy.stats            import qmc
from dataclasses            import fields

mpl.rcParams.update({'font.size': 9,'axes.labelsize': 9,'axes.titlesize': 9,'xtick.labelsize': 9,'ytick.labelsize': 9,'legend.fontsize': 9,
                             'grid.alpha': 0.5, 'grid.linewidth': 0.7,'grid.linestyle': '-', 'xtick.minor.visible': True,
                             'ytick.minor.visible': True,})

class DoESampler:
    def __init__(self):
        self.descr = "Sample from the parameters' domain "
    def evaluate_points_spread(self,points, weights=None):
        distances = pdist(points, metric='euclidean')
        return [float(np.mean(distances)), float(np.min(distances)), float(np.std(distances))]

    def sampling_quality_obj(self,points, weights=None):
        default_weights = [0,1,0]
        n_dims = points.shape[1]
        n_sample = points.shape[0]
        if weights is None:
            weights = np.ones(n_dims)

        #weighted_points = points * np.sqrt(weights)

        distances = pdist(points, metric='euclidean')
        weighted_av  = np.mean([weights[ii] * self.dist_calculator(np.array(points[:,ii]).reshape(-1,1),n_sample)[0].mean() for ii in range(n_dims)])
        weighted_min = np.mean([weights[ii] * self.dist_calculator(np.array(points[:,ii]).reshape(-1,1),n_sample)[0].min() for ii in range(n_dims)])
        weighted_std = np.mean([weights[ii] * self.dist_calculator(np.array(points[:,ii]).reshape(-1,1),n_sample)[0].std() for ii in range(n_dims)])

        #return sum([default_weights[0]*float(np.mean(distances)), default_weights[1]*float(np.min(distances)), -default_weights[2]*float(np.std(distances))] ) 
        return sum([default_weights[0]*weighted_av, default_weights[1]*weighted_min, -default_weights[2]*weighted_std] ) 



    def greedy_minimax(self,n_samples, n_dims, n_candidates=1000): 
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=42)
        candidates = sampler.random_base2(m=int(np.ceil(np.log2(n_candidates))))[:n_candidates]
        points = [candidates[np.random.randint(n_candidates)]] # Choose an integer from the interval {0 ; n_candidates}

        for _ in range(n_samples - 1): 
            dist = cdist(candidates, np.array(points)) # Calculate the distance between the points and the candidates
            np.fill_diagonal(dist, np.inf)
            min_dist = dist.min(axis=1) # Minimal distance from         print
            next_idx = np.argmax(min_dist) 
            points.append(candidates[next_idx]) 

        return np.array(points)


    def dist_calculator(self,A,n_sample):
        dist = cdist(A, A,metric='euclidean') # Calculate the distance between the points and the candidates
        np.fill_diagonal(dist, np.inf)
        return dist[np.triu_indices(n = n_sample, k=1)], dist

    def dist_eval(self,B,name,n_sample,FIG = False):

        ddist, dist = self.dist_calculator(B,n_sample)

        if FIG:
            plt.figure(figsize=(5,2))
            plt.imshow(dist)
            plt.colorbar()

        return ddist.mean(), ddist.min(), ddist.std()

    def two_dim_plotter(self,POINTS,names):
        n_dims = POINTS[0].shape[1]
        n_sample = POINTS[0].shape[0]
        fig, ax = plt.subplots(1,n_dims,figsize=(n_dims*4,3))
        if n_dims == 1:
            ax = [ax]
        collor = ['k','b','r','b','orange','green','b','r','r','b','k']
        markerr = ['x','^','s','x','s','x','^','s','x','^','s']

        for i in range(n_dims):
            for j in range(len(POINTS)):
                D = self.dist_calculator( np.array(POINTS[j][:,i]).reshape(-1,1), n_sample)[0] 
                A,B,C = self.dist_eval(POINTS[j],names[j],n_sample,False)

                ax[i].scatter(POINTS[j][:,i],[j+1]*len(POINTS[j][:,i]),color=collor[j],marker=markerr[j],label=f'{names[j]}, {np.min(D):.3f},{np.mean(D):.3f}')

            ax[i].set_xlabel(f'X{i:.0f}')
            ax[i].grid(which='major', alpha=0.25)
            ax[i].grid(which='minor', alpha=0.1)
            ax[i].set_ylim([0,len(POINTS) + 10])
            ax[i].legend(fontsize = 9 , loc = 'upper right',ncol = 1)

        fig.tight_layout()
        pass

    def generate_and_evaluate(self,n_samples, n_dims, weights, types,Fraction=0):
        greedy_reduce = n_samples <= n_dims
        target_n = n_dims + 5 if greedy_reduce else n_samples

        if types == 'LHS':
            points = qmc.LatinHypercube(d=n_dims, seed=42).random(n=target_n)
        elif types == 'LHS_OPT':
            points = qmc.LatinHypercube(d=n_dims, optimization="lloyd", seed=42).random(n=target_n)
        elif types == 'RAND':
            points = np.random.rand(target_n, n_dims)
        elif types == 'SOB':
            sampler = qmc.Sobol(d=n_dims, seed=42, scramble=True)
            m = int(np.ceil(np.log2(target_n)))
            points = sampler.random_base2(m=m)[:target_n]
        elif types == 'GREEDY':
            points = self.greedy_minimax(target_n, n_dims)
        elif types == 'FACT':
            points = self.Factorial_Design(n_dims, Fraction)
            greedy_reduce = False
        else:
            raise ValueError(f"Unknown type: {types}")

        if greedy_reduce:
            points = self.reverse_greedy_maximin(points, n_samples)

        current = self.sampling_quality_obj(points, weights)

        return current, points

    def doe_generator(self,N,n_samples,n_dims,weights,types,Fraction=0):
        if types == 'FACT':
            N = 1

        results = Parallel( n_jobs=-1, verbose=0 )( delayed(self.generate_and_evaluate)(n_samples, n_dims, weights, types,Fraction) for _ in range(N) )
        best = float(-1e5)
        best_points = None

        for current, points in results:

            if current > best:
                best = current
                best_points = points

        return best_points, self.evaluate_points_spread(best_points)

    def reverse_greedy_maximin(self,points,n_samples,weights = None):
        n_dims = points.shape[1]
        if weights is None:
                weights = np.ones(n_dims)

        dist = self.dist_calculator(points,n_samples)[1]
        min_dist = dist.min(axis=1)
        idx_remove = np.argmin(min_dist)

        for _ in range(np.shape(points)[0] - n_samples):
            idx_remove = np.argmin(min_dist)

            points = np.delete(points, idx_remove, axis=0)
            dist = np.delete(dist, idx_remove, axis=0)
            dist = np.delete(dist, idx_remove, axis=1)

            min_dist = dist.min(axis=1)
        return points

    def figger(self,axx, LOCs, randd_p, markk, coll, labb,LS):
        axx[0].scatter(LOCs,randd_p[:,0],20,marker=markk,color=coll,label=labb)
        [axx[i].scatter(LOCs,randd_p[:,i],20,marker=markk,color=coll) for i in [1,2]]
        [axx[j].plot(LOCs,randd_p[:,j],color=coll,linestyle=LS) for j in [0,1,2]]

    def greedy_reducer(self,original_points, n_to_reduce, random_state=42): 
        rng = np.random.default_rng(random_state)
        candidates = original_points
        initial_idx = rng.integers(0, len(candidates))
        points = [candidates[initial_idx]]

        for _ in range(n_to_reduce - 1): 
            dist = cdist(candidates, np.array(points)) 
            min_dist = dist.min(axis=1) 
            next_idx = np.argmax(min_dist) 
            points.append(candidates[next_idx]) 

        return np.array(points)

    def dim_counter(self,Exp_par):
        dim_c = 0
        for f in fields(Exp_par):
            if f.type.__name__ == "DefineProcessParameter" and getattr(Exp_par, f.name).if_in_doe:
                    dim_c += 1
        return dim_c

    def three_dim_plotter(self,A):
        fig = plt.figure(figsize=(5,3))
        ax = fig.add_subplot(111, projection='3d')
        
        if np.shape(A)[1] >= 3:
            ax.scatter(A[:,0],A[:,1],A[:,-1])
            ax.set_xlabel('X0')
            ax.set_ylabel('X1')
            ax.set_zlabel('X2')
            fig.tight_layout()
            ax.text(0,0,0,f'No. of exp: {np.shape(A)[0]}',fontsize=20)
        else:
            ax.text(0,0,0,'n_dims < 3 ! ',fontsize=20)
        plt.show()
        return fig, ax

    def scaler(self,A):
        B = A - np.min(A,axis=0)
        if sum(np.max(B,axis=0)) != 0:
            C = B / np.max(B,axis=0)
        else:
            C = B
        return C

    def two_factplotter(self,A):
        plt.figure(figsize=(2,2))
        plt.scatter(A[:,0],A[:,2])
        plt.xlim([-0.01,1.01])
        plt.ylim([-0.01,1.01])
        return

    def factorial_design(self,n_dims, Fraction=0):
        factors = ['a','b','c','d','e','f']
        msg = ''
        if n_dims > 10:
            msg = 'please select n_dims to be <10'
            print(msg)
            return np.zeros([1,n_dims])
        if Fraction >= 3:
            Fraction = 0
            msg = 'please select Fraction to be 0 <= Fraction <= 2'
            print(msg)
            return np.zeros([1,n_dims])
        if n_dims == 4 and Fraction > 1:
            Fraction = 0
            msg = 'please select Fraction to be 0 <= Fraction <= 1 if n_dims==4'
            print(msg)
            return np.zeros([1,n_dims])
        if n_dims == 3 and Fraction>1:
            Fraction = 0
            msg = 'please select Fraction to be 0 <= Fraction <= 1 if n_dims==3'
            print(msg)
            return np.zeros([1,n_dims])

        else:       

            if n_dims < 3 and Fraction > 0:
                Fraction = 0

            base = factors[:n_dims-Fraction]
            generated = [base[0] + base[i+1] for i in range(Fraction)]
            design_string = ' '.join(base + generated)
            design_matrix = fracfact(design_string)
            factor_levels = dict()

            for i in range(n_dims):
                factor_levels[f"X{i+1}"] = np.array([0, 1])

            plan_ml = pd.DataFrame(design_matrix, columns = factor_levels.keys())
            for col, values in factor_levels.items():
                plan_ml[col] = plan_ml[col].map(lambda x: values[0] if x == -1 else values[1])

            return np.vstack(plan_ml.values)

    def descaler(self,DoE_level,minmaxvals):
        return [float(minmaxvals.min_val + (minmaxvals.max_val-minmaxvals.min_val)*DoE_level)] + [minmaxvals.min_val, minmaxvals.max_val]


# In[4]:


#Doe_sampler = DoE_sampler()


# In[ ]:




