"""Standalone scientific figure from completed, fixed-budget audit tables."""
import sys
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);f=pd.read_csv(out/'INTERVENTION_SUMMARY.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
styles=[('full','PINA analytic BA','#2457aa'),('noR','Without Reliability','#cc6b22'),
        ('cart','Learned Cartesian','#7d47ad'),('table_ba','TRAIN table BA','#71842c'),
        ('table_bond','TRAIN table bonds','#929292'),('mmff_gradient','MMFF94s gradient','#098475')]
curve=[]
for prefix,label,color in styles:
    part=f[f['case'].str.startswith(prefix+'_') & f['case'].str.contains('@')] if prefix in ['full','noR','cart'] else f[f['case'].str.startswith(prefix+'@')]
    part=part.copy();part['budget']=part['case'].str.split('@').str[-1].astype(float)
    mean=part.groupby('budget')[['validity3d','actual_rms']].mean()
    axes[0].plot(mean.actual_rms,100*mean.validity3d,'o-',label=label,color=color,lw=1.8,ms=5)
    for budget,row in mean.iterrows():curve.append(dict(method=label,budget=budget,v3d_percent=100*row.validity3d,actual_rms=row.actual_rms))
axes[0].axhline(100*f.loc[f['case']=='source','validity3d'].iloc[0],ls=':',color='black',label='Source')
axes[0].axvline(.01,color='black',alpha=.12)
axes[0].set(xlabel='Cartesian RMS displacement (angstrom)',ylabel='Validity3D (%)',
            title='Fixed intervention budget: direction quality')
axes[0].legend(fontsize=8,loc='best');axes[0].grid(alpha=.15)
effects=pd.read_csv(out/'PAIRED_MOLECULE_BOOTSTRAP.csv')
effects=effects[np.isclose(effects.budget,.01)].set_index('comparator')
names=['noR','cart','table_ba','table_bond','mmff_gradient','coefficient_sigma_stat']
labels=['Without Reliability','Learned Cartesian','TRAIN table BA','TRAIN table bonds','MMFF94s gradient','Statistical coefficient sigma*']
e=effects.loc[names];x=e.delta_v3d_pp.to_numpy()
axes[1].errorbar(x,np.arange(len(x)),xerr=np.stack([x-e.ci_low_pp,e.ci_high_pp-x]),fmt='o',color='#2457aa',capsize=3)
axes[1].axvline(0,color='black',lw=1);axes[1].set_yticks(np.arange(len(x)),labels);axes[1].invert_yaxis()
axes[1].set(xlabel='PINA minus comparator (percentage points)',title='Paired molecule bootstrap at 0.01 angstrom')
axes[1].grid(axis='x',alpha=.15)
fig.suptitle('Exploratory audit: 256 molecules / 512 Source conformers',fontsize=13)
fig.supxlabel('Three trained seeds where available; *sigma intervention uses seed 307. Intervals describe molecule sampling.',fontsize=9)
fig.savefig(out/'MATCHED_BUDGET_AUDIT.png',dpi=190)
fig.savefig(out/'MATCHED_BUDGET_AUDIT.pdf')
pd.DataFrame(curve).to_csv(out/'PLOTTED_CURVES.csv',index=False)
