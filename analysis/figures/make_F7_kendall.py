import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from twinbench.metrics import statistics as RM

CTRLS=['pdg','ctc_pd','ctc_smc','pbc']
ds=pd.read_csv("data/all_results_datasheet.csv")
urdf=pd.read_csv("data/all_results_main.csv")

def ranked(df, ctx_cols):
    rows=[]
    for key,b in df.groupby(ctx_cols):
        m=b.groupby('controller').RMSE_cart_norm.mean(); m=m[m.index.isin(CTRLS)]; rk=m.rank()
        d=dict(zip(ctx_cols,key)) if isinstance(key,tuple) else {ctx_cols[0]:key}
        for c in m.index: rows.append({**d,'controller':c,'rank':float(rk[c])})
    return pd.DataFrame(rows)

# panel (a) datasheet spans BOTH scenes (Tier A + Tier C); URDF panels are single-scene by design
CTX_A=['robot','params','trajectory','scene']; CTX_URDF=['robot','params','trajectory']
panels=[
    ("(a) Datasheet / proxy", ranked(ds[ds.fidelity=='proxy'], CTX_A), CTX_A, "robot×unc×traj×scene"),
    ("(b) URDF / proxy",      ranked(urdf[urdf.fidelity=='proxy'], CTX_URDF), CTX_URDF, "robot×unc×traj"),
    ("(c) URDF / rigid-body", ranked(urdf[urdf.fidelity=='rigid_body'], CTX_URDF), CTX_URDF, "robot×unc×traj"),
]

fig,axes=plt.subplots(1,3,figsize=(13.5,4.6))
summ={}
for ax,(title,rdf,ctx,ctxlab) in zip(axes,panels):
    ctx_list,mat=RM.kendall_stability_matrix(rdf,ctx,['controller'],'rank')
    mu,sd,n=RM.kendall_offdiag_summary(mat)
    summ[title]=(mu,sd,len(ctx_list))
    im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_title(f"{title}\nmean off-diag $\\tau_b$ = {mu:.3f} $\\pm$ {sd:.3f}",fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{len(ctx_list)} contexts ({ctxlab})",fontsize=8)
cbar=fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02); cbar.set_label("Kendall $\\tau_b$")
fig.suptitle("Within-tier context-by-context ranking consistency (corrected sign, 4 controllers)",fontsize=11,y=1.02)
out="figures/F7_kendall_methods.png"; fig.savefig(out,dpi=300,bbox_inches='tight')
print("SAVED",out)
for t,(mu,sd,nctx) in summ.items():
    print(f"  {t:24s}: {mu:.3f} +/- {sd:.3f}  (n_contexts={nctx})")
