import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CCOL={'PD+g':'#3b82f6','CTC-PD':'#ef4444','CTC-SMC':'#a855f7','PBC':'#f59e0b'}
fig,ax=plt.subplots(figsize=(3.6,6.2)); ax.set_xlim(0,10); ax.set_ylim(0,26); ax.axis('off')

def box(y,h,txt,fc='#f3f4f6',ec='#374151',fs=8,bold=False,x=0.4,w=9.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.25',
                                fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',color='#111827')
def arrow(y0,y1):
    ax.add_patch(FancyArrowPatch((5,y0),(5,y1),arrowstyle='-|>',mutation_scale=13,lw=1.4,color='#374151'))

def chips(y,items,colors=None):
    n=len(items); w=9.2/n;
    for i,it in enumerate(items):
        c=(colors or {}).get(it,'#e5e7eb'); tc='#111827'
        ax.add_patch(FancyBboxPatch((0.4+i*w+0.04,y),w-0.08,1.0,boxstyle='round,pad=0.02,rounding_size=0.15',fc=c,ec='#374151',lw=0.7))
        ax.text(0.4+i*w+w/2,y+0.5,it,ha='center',va='center',fontsize=6.3,color=tc,fontweight='bold')

ax.text(5,25.4,'TwinBench evaluation grid',ha='center',fontsize=9.5,fontweight='bold')
box(23.2,1.3,'6 six-DOF manipulators',fc='#eef2ff',bold=True)
ax.text(0.5,22.55,'datasheet',fontsize=5.2,color='#6b7280',va='center',ha='left',rotation=90)
chips(21.9,['KR6','RS006L','MPX1950'])
ax.text(0.5,20.85,'URDF',fontsize=5.2,color='#6b7280',va='center',ha='left',rotation=90)
chips(20.2,['UR5e','UR10e','TX60'])
box(18.3,1.3,'4 trajectory types',fc='#eef2ff',bold=True); chips(17.0,['Cubic','Min-Jerk','FSMJ','FSMJ-noVia'])
box(15.1,1.3,'4 controllers',fc='#eef2ff',bold=True);       chips(13.8,['PD+g','CTC-PD','CTC-SMC','PBC'],CCOL)
box(11.9,1.3,'3 uncertainty levels',fc='#eef2ff',bold=True);chips(10.6,['light','nominal','heavy'])
box(8.7,1.3,'2 fidelity tiers',fc='#eef2ff',bold=True);     chips(7.4,['Decoupled proxy','Pinocchio rigid-body'])

arrow(7.3,6.3)
box(5.0,1.2,'Deterministic simulation\nSHA-256 seeding (target + noise)',fc='#ecfdf5',ec='#065f46',bold=True,fs=7.5)
arrow(4.9,3.9)
box(2.6,1.2,'RMSE$_{cart,norm}$ per (context, controller)\n→ per-context 4-controller ranking',fc='#fff7ed',ec='#9a3412',fs=7)
arrow(2.5,1.5)
box(0.2,1.2,'Cross-fidelity Kendall $\\tau_b$\nranking-stability analysis',fc='#faf5ff',ec='#6b21a8',bold=True,fs=7.5)

fig.savefig('figures/F1_pipeline.png',dpi=300,bbox_inches='tight',facecolor='white')
print('SAVED figures/F1_pipeline.png')

# under a different filename; same content, PDF instead of PNG.
fig.savefig('figures/twinbench_grid.pdf',bbox_inches='tight',facecolor='white')
print('SAVED figures/twinbench_grid.pdf')
