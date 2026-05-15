"""
model_shap_interaction.py — FAST pairwise feature interaction (permutation method)
Only top-6 continuous features (15 pairs), 3 test samples, 3 permutations
"""
import torch, numpy as np, pandas as pd, os, sys, warnings, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from data_pipeline_v2 import _build_features, THRUST_SCALE, MFR_MAX
from model_train_v2 import DualOutputLSTM, INPUT_DIM

FN = ['ton','vl','test_pressure','cumulated_on_time','cumulated_throughput',
      'cumulated_pulses','ssf','health_check','ramp1','ramp2','ramp3','ramp4',
      'onmod','offmod','random_short','random_long','random_mixed']
FIG_DIR = "outputs/figures/v2"; NP_DIR = "outputs/predictions/v2"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(NP_DIR, exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.size':10,
    'axes.titlesize':14,'axes.labelsize':12,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

SEQ=200; G0=9.80665

def load_flat(md, dr, n=3, seq=SEQ):
    df=pd.read_csv(md); df['filename']=df['filename'].str.strip()
    ex=set(os.listdir(dr))
    sub=df[(df['anomalous']==False)&(df['sn']>12)&df['filename'].isin(ex)]
    xs,modes=[],[]
    for m in sub['test_mode'].unique():
        for _,r in sub[sub['test_mode']==m].head(1).iterrows():
            fp=os.path.join(dr,r['filename'])
            if not os.path.exists(fp): continue
            try: d=pd.read_csv(fp)
            except: continue
            if len(d)<seq: continue
            x3d,_,_=_build_features(d.iloc[:seq],r); xs.append(x3d.mean(axis=0)); modes.append(m)
            if len(xs)>=n: break
        if len(xs)>=n: break
    return np.stack(xs)[:n],modes[:n]

def rmse(model, x_t, y_t, idx=0):
    sc=THRUST_SCALE if idx==0 else MFR_MAX
    with torch.no_grad():
        p=model(x_t)
        return ((p[:,:,idx]-y_t[:,:,idx])**2).mean().sqrt().item()*sc

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\nFast pairwise interaction — top-6 features only")
    model=DualOutputLSTM(INPUT_DIM).to(device)
    model.load_state_dict(torch.load("outputs/models/v2/dual_output_lstm_v2.pth",map_location=device,weights_only=True))
    model.eval()

    # Load 3 test samples as flat [3,17]
    xs_flat,modes=load_flat("data/metadata.csv","data/dataset/dataset/test/",n=3)
    # Also load the full 3D version for actual model evaluation
    xs_full,ys_full=[],[]
    df=pd.read_csv("data/metadata.csv"); df['filename']=df['filename'].str.strip()
    ex=set(os.listdir("data/dataset/dataset/test/"))
    sub=df[(df['anomalous']==False)&(df['sn']>12)&df['filename'].isin(ex)]
    for m in sub['test_mode'].unique():
        for _,r in sub[sub['test_mode']==m].head(1).iterrows():
            fp=os.path.join("data/dataset/dataset/test/",r['filename'])
            if not os.path.exists(fp): continue
            try: d=pd.read_csv(fp)
            except: continue
            if len(d)<SEQ: continue
            x3d,y3d,_=_build_features(d.iloc[:SEQ],r); xs_full.append(x3d); ys_full.append(y3d)
            if len(xs_full)>=3: break
        if len(xs_full)>=3: break
    xs_arr=np.stack(xs_full); ys_arr=np.stack(ys_full)
    x_t=torch.from_numpy(xs_arr).float().to(device)
    y_t=torch.from_numpy(ys_arr).float().to(device)
    print(f"  Test: {x_t.shape} {y_t.shape}  modes={modes}")

    # Baseline
    bl_t=rmse(model,x_t,y_t,0); bl_m=rmse(model,x_t,y_t,1)
    print(f"  Baseline: thrust={bl_t:.4f}N  mfr={bl_m:.1f}mg/s\n")

    # ── Single-feature importance (top-6 only) ──
    N=6; N_PERM=3
    single_t=np.zeros(17); single_m=np.zeros(17)
    print("Single-feature importance (6 continuous)...")
    for f in range(N):
        lt=[]; lm=[]
        for _ in range(N_PERM):
            xp=xs_arr.copy()
            for t in range(SEQ): np.random.shuffle(xp[:,t,f])
            xtp=torch.from_numpy(xp).float().to(device)
            lt.append(rmse(model,xtp,y_t,0))
            lm.append(rmse(model,xtp,y_t,1))
        single_t[f]=np.mean(lt)-bl_t
        single_m[f]=np.mean(lm)-bl_m
    topt=[FN[i] for i in np.argsort(single_t)[-1:-4:-1]]
    topm=[FN[i] for i in np.argsort(single_m)[-1:-4:-1]]
    print(f"  Thrust top-3: {topt}")
    print(f"  MFR    top-3: {topm}")

    # ── Pairwise interaction (15 pairs of top-6) ──
    n_pairs=N*(N-1)//2
    inter_t=np.zeros((17,17)); inter_m=np.zeros((17,17))
    done=0; t0=time.time()
    print(f"\nPairwise interactions ({n_pairs} pairs)...")
    for i in range(N):
        for j in range(i+1,N):
            lt=[]; lm=[]
            for _ in range(N_PERM):
                xp=xs_arr.copy()
                for t in range(SEQ):
                    np.random.shuffle(xp[:,t,i])
                    np.random.shuffle(xp[:,t,j])
                xtp=torch.from_numpy(xp).float().to(device)
                lt.append(rmse(model,xtp,y_t,0))
                lm.append(rmse(model,xtp,y_t,1))
            rmse_ij_t=np.mean(lt); rmse_ij_m=np.mean(lm)
            inter_t[i,j]=rmse_ij_t-bl_t-single_t[i]-single_t[j]
            inter_t[j,i]=inter_t[i,j]
            inter_m[i,j]=rmse_ij_m-bl_m-single_m[i]-single_m[j]
            inter_m[j,i]=inter_m[i,j]
            done+=1
            elapsed=time.time()-t0
            eta=elapsed/done*(n_pairs-done)
            print(f"  {done}/{n_pairs}  {elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"\nDone in {time.time()-t0:.0f}s")

    # ── Top pairs ──
    pairs_t=[(i,j,inter_t[i,j]) for i in range(N) for j in range(i+1,N)]
    pairs_t.sort(key=lambda x:abs(x[2]),reverse=True)
    print(f"\n{'='*60}\n  Top Interaction Pairs (Thrust)\n{'='*60}")
    for rank,(i,j,v) in enumerate(pairs_t,1):
        tp="SYNERGY" if v>0 else "ANTAGONY"
        print(f"  {rank:2d}. {FN[i]:>22s} x {FN[j]:<22s} {v:+.4f} N  {tp}")

    pairs_m=[(i,j,inter_m[i,j]) for i in range(N) for j in range(i+1,N)]
    pairs_m.sort(key=lambda x:abs(x[2]),reverse=True)
    print(f"\n{'='*60}\n  Top Interaction Pairs (MFR)\n{'='*60}")
    for rank,(i,j,v) in enumerate(pairs_m,1):
        tp="SYNERGY" if v>0 else "ANTAGONY"
        print(f"  {rank:2d}. {FN[i]:>22s} x {FN[j]:<22s} {v:+.1f} mg/s  {tp}")

    # ── Plots ──
    fig,axes=plt.subplots(1,2,figsize=(18,7))
    for ax,mat,title in [(axes[0],inter_t,'Thrust'),(axes[1],inter_m,'MFR')]:
        vmax=max(abs(mat[:N,:N].max()),abs(mat[:N,:N].min()))
        im=ax.imshow(mat[:N,:N],cmap='RdBu_r',aspect='auto',vmin=-vmax,vmax=vmax)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(FN[:N],rotation=45,ha='right',fontsize=9)
        ax.set_yticklabels(FN[:N],fontsize=9)
        ax.set_title(f'{title} Feature Interactions\nRed=Synergy Blue=Antagony',fontsize=12,fontweight='bold')
        thresh=vmax*0.3
        for r in range(N):
            for c in range(N):
                v=mat[r,c]
                ax.text(c,r,f'{v:.3f}',ha='center',va='center',fontsize=7,
                       color='white' if abs(v)>thresh else 'black')
        plt.colorbar(im,ax=ax,shrink=0.8)
    plt.suptitle('Feature Interaction Matrix (Permutation Method)',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"shap_interaction_heatmap.png"),dpi=300); plt.close()
    print(f"\n  -> shap_interaction_heatmap.png")

    # Top-10 bar chart (thrust)
    fig,ax=plt.subplots(figsize=(12,5))
    labels=[f'{FN[i]} x {FN[j]}' for i,j,_ in pairs_t]
    values=[v for _,_,v in pairs_t]
    colors=['#c0392b' if v>0 else '#2980b9' for v in values]
    ax.barh(range(len(values)),values,color=colors,height=0.6,ec='white')
    ax.set_yticks(range(len(values))); ax.set_yticklabels(labels,fontsize=10); ax.invert_yaxis()
    ax.set_xlabel('Interaction Strength (N RMSE)'); ax.axvline(0,color='black',lw=0.5)
    ax.set_title('Feature Interaction Pairs — Thrust',fontsize=14,fontweight='bold')
    ax.grid(axis='x',alpha=0.3,ls='--')
    for b,v in zip(ax.containers[0],values):
        xpos=v+(0.002 if v>=0 else -0.002)
        ax.text(xpos,b.get_y()+b.get_height()/2,f'{v:+.3f}',va='center',
               ha='left' if v>=0 else 'right',fontsize=9,fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"shap_interaction_top10.png"),dpi=300); plt.close()
    print(f"  -> shap_interaction_top10.png")
    np.save(os.path.join(NP_DIR,"shap_interaction_thrust.npy"),inter_t)
    np.save(os.path.join(NP_DIR,"shap_interaction_mfr.npy"),inter_m)
    print("Done.")

if __name__=="__main__": main()
