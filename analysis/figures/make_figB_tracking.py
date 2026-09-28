import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments import run_main_grid as R

JIDX=1
COL={'pdg':'#3b82f6','ctc_pd':'#ef4444','ctc_smc':'#a855f7','pbc':'#f59e0b'}
LAB={'pdg':'PD+g','ctc_pd':'CTC-PD','ctc_smc':'CTC-SMC','pbc':'PBC'}
CTRLS=['pdg','ctc_pd','ctc_smc','pbc']
PANELS=[("(a) KR6, proxy",'kr6','proxy'),("(b) UR5e, rigid-body",'ur5e','rigid_body')]

cap={}
_ocm=R.compute_metrics
def wrap_cm(*, q,dq_sim,tau_hist,ddq_hist,sat_hist,qd,dqd,t,T_eff,dh,R_max,payload_used,tau_max,dq_cap,ddq_cap,urdf_robot=None):
    p_des=R.ee_pos_batch(qd,dh=dh,urdf_robot=urdf_robot); p_act=R.ee_pos_batch(q,dh=dh,urdf_robot=urdf_robot)
    cart=np.linalg.norm(p_des-p_act,axis=1)/max(R_max,1e-12)
    cap['run']=dict(t=t,qd=qd.copy(),q=q.copy(),tau=tau_hist.copy(),sat=sat_hist.copy(),cart=cart)
    return _ocm(q=q,dq_sim=dq_sim,tau_hist=tau_hist,ddq_hist=ddq_hist,sat_hist=sat_hist,qd=qd,dqd=dqd,t=t,
                T_eff=T_eff,dh=dh,R_max=R_max,payload_used=payload_used,tau_max=tau_max,dq_cap=dq_cap,ddq_cap=ddq_cap,urdf_robot=urdf_robot)
R.compute_metrics=wrap_cm

NO_IMP={'joint_index0':1,'tau_frac_of_max':0.0,'onset':1.0,'duration':0.2}

def build(disturb_cfg,fname,title_extra):
    data={}; rmse={}; satmax=0.0
    for pi,(title,robot,fid) in enumerate(PANELS):
        for c in CTRLS:
            cap.clear()
            res=R.run_single(robot,'open_reach','cubic',c,'nominal',0,fidelity=fid,disturb_cfg=disturb_cfg)
            data[(pi,c)]=cap['run']; rmse[(pi,c)]=res['RMSE_cart_norm']; satmax=max(satmax,res['Sat_frac_any'])
    fig,ax=plt.subplots(4,2,figsize=(11,10),sharex='col')
    rows=['$q_2$ [rad]','$\\tau_2$ [Nm]','$e_2=q_d-q$ [rad]','$\\|p_{des}-p_{act}\\|/R_{max}$']
    for pi,(title,robot,fid) in enumerate(PANELS):
        d0=data[(pi,CTRLS[0])]; t=d0['t']
        ax[0,pi].plot(t,d0['qd'][:,JIDX],'k--',lw=1.3,zorder=1)
        for c in CTRLS:
            d=data[(pi,c)]; col=COL[c]
            ax[0,pi].plot(t,d['q'][:,JIDX],color=col,lw=1.1)
            ax[1,pi].plot(t,d['tau'][:,JIDX],color=col,lw=1.1)
            sat=d['sat'][:,JIDX].astype(bool)
            if sat.any(): ax[1,pi].plot(t[sat],d['tau'][sat,JIDX],'.',color=col,ms=4)
            ax[2,pi].plot(t,d['qd'][:,JIDX]-d['q'][:,JIDX],color=col,lw=1.1)
            ax[3,pi].plot(t,d['cart'],color=col,lw=1.1)
        ax[2,pi].axhline(0,color='0.7',lw=0.6,zorder=0)
        ax[0,pi].set_title(title,fontsize=12,fontweight='bold')
        for r in range(4): ax[r,pi].grid(alpha=0.25)
        ax[3,pi].set_xlabel('time [s]')
    for r in range(4): ax[r,0].set_ylabel(rows[r],fontsize=10)
    handles=[Line2D([0],[0],color='k',ls='--',label='desired')]+[Line2D([0],[0],color=COL[c],lw=2,label=LAB[c]) for c in CTRLS]
    fig.legend(handles=handles,loc='upper center',ncol=5,fontsize=10,frameon=False,bbox_to_anchor=(0.5,0.995))
    fig.suptitle(f"Joint-2 tracking, cubic / nominal / seed 0, corrected sign {title_extra}",y=0.955,fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(fname,dpi=300,bbox_inches='tight')
    return rmse,satmax

for cfg,fname,extra,ver in [(None,"figures/figB_tracking_withimpulse.png","(with 5% joint-2 impulse at t=1.0s)","WITH impulse"),
                            (NO_IMP,"figures/figB_tracking_noimpulse.png","(no disturbance)","NO impulse")]:
    rmse,satmax=build(cfg,fname,extra)
    print(f"\n=== {ver}: {fname} | Sat max {satmax} ===")
    for pi,(title,robot,fid) in enumerate(PANELS):
        print(f"  {title}: "+"  ".join(f"{LAB[c]}={rmse[(pi,c)]:.4f}" for c in CTRLS))
