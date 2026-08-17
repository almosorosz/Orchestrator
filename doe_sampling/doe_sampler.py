#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

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

    def two_dim_plotter(self, POINTS, names, name_strings=[],fig=None):
        n_dims   = POINTS[0].shape[1]
        n_sample = POINTS[0].shape[0]
        
        # --- 1. DYNAMIC FIGURE SIZE (Fixes Compression) ---
        # Give each factor ~0.85 inches of width so factors never squash
        fig_width = max(1.5, n_dims * 0.3)
        fig_height = 3.0
    
        # --- 2. DYNAMIC LEGEND COLUMNS ---
        # Automatically split legend into 1, 2, or 3 columns as factors grow
        legend_cols = 1 if n_dims <= 4 else (2 if n_dims <= 12 else 3)
        
        bbox_2 = dict(facecolor="white",edgecolor="black",boxstyle="round,pad=0.3",alpha=0.8)
        
        BG_COLOR = "#111827"
        if fig is None:
            fig = Figure(figsize=(fig_width, fig_height),facecolor=BG_COLOR)
        ax = fig.subplots(1, 1)
    
        collor = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6","#06b6d4","#ec4899","#f97316","#14b8a6","#6366f1","#d946ef",]
        markerr = ["x", "^", "s", "x", "s", "x", "^", "s", "x", "^", "s"]
    
        if not name_strings:
            name_strings = [f"X{i}" for i in range(n_dims)]

        for i in range(n_dims):
            disp_name = name_strings[i][:8]

            ax.text(i + 0.8,0.0, name_strings[i],rotation=90, ha="center",va="bottom",color="k",fontsize=6,clip_on=False,bbox = bbox_2)
            
            for j in range(len(POINTS)):
                D = self.dist_calculator(np.array(POINTS[j][:, i]).reshape(-1, 1), n_sample)[0]

                
                label_text = f"{disp_name:<8}, Min: {np.min(D):.3f}, Avg: {np.mean(D):.3f}"
                ax.scatter([i + 1] * len(POINTS[j][:, i]),POINTS[j][:, i],color=collor[i % len(collor)],
                               marker=markerr[i % len(markerr)],alpha=0.8,label=label_text if j == 0 else "")
                
    
        # --- 4. AXIS CONFIGURATION & PADDING ---
        ax.set_xlim(0.5, n_dims + 0.5)  # Adds clean padding on left and right
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(range(1, n_dims + 1))
        ax.set_xticklabels(['' for i in range(n_dims)])
        # Truncate long X-tick labels to prevent overlapping when rotated
        #ax.set_xticklabels(name_strings, rotation=25, ha="right", color="#9ca3af", fontsize=8)
        ax.set_xlabel("Samples by factors", color="#9ca3af",fontsize=8)  
        
        if names:
            ax.set_ylabel(f"Sampling strategy: {names}", color="#e5e7eb", fontsize=8)
    
        ax.grid(which="major", alpha=0.2)
        ax.grid(which="minor", alpha=0.075)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    
        # --- 5. LEGEND POSITIONING ---
        legend = ax.legend(fontsize=7,loc="lower center",bbox_to_anchor=(0.5, 1.02),ncol=legend_cols,frameon=True,)
        legend.get_frame().set_facecolor("#1f2937")
        legend.get_frame().set_edgecolor("#9ca3af")
        for text in legend.get_texts():
            text.set_color("#9ca3af")
    
        # --- 6. DARK MODE STYLING ---
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor("#1f2937")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
        ax.tick_params(colors="#9ca3af")
        ax.xaxis.label.set_color("#9ca3af")
        fig.subplots_adjust(top=0.72, bottom=0.22, left=0.08, right=0.95)
        return fig
        
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
            points = self.factorial_design(n_dims, Fraction)
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

    def three_dim_plotter(
        self, A, dimname1="X0", dimname2="X1", dimname3="X2", title_text="",fig=None):
        BG_COLOR = "#111827"
        bbox_2 = dict(facecolor="white",edgecolor="black",boxstyle="round,pad=0.3",alpha=0.8)
        
        if fig is None: 
            fig = Figure(figsize=(4.2, 4.2), facecolor=BG_COLOR, layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
    
        # --- TITLE WITH BACKGROUND BOX ---
        ax.set_title(
            " Representation of\nfirst three factors in 3D. ",
            color="#e5e7eb",
            fontsize=7,
            y=0.96,  # Height positioning
            pad=-1,  # Pull down slightly
            bbox=dict(boxstyle="round,pad=0.8",facecolor="#1f2937",edgecolor="#4b5563", linewidth=0.7,alpha=0.95,),
        )
    
        # ... rest of your 3D plot code ...
        if np.shape(A)[1] >= 3:
            ax.scatter(A[:, 0], A[:, 1], A[:, 2], color="#3b82f6", alpha=0.85, s=30)
            ax.text(0.5, 0, 0, dimname1, bbox=bbox_2, rotation=15, ha="center", va="center")
            ax.text(1, 0.5, 0, dimname2, bbox=bbox_2, rotation=-15, ha="center", va="center")
            ax.text(1, 1, 0.5, dimname3, bbox=bbox_2, rotation=90, ha="center", va="center")
            ax.set_xlim([-0.1, 1.1])
            ax.set_zlim([-0.1, 1.1])
    
        # Dark mode pane styling
        ax.set_facecolor(BG_COLOR)
        pane_color = (31 / 255, 41 / 255, 55 / 255, 1.0)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
    
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_color("#9ca3af")
    
        ax.tick_params(colors="#9ca3af", pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    
        return fig

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




