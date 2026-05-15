"""
model_test_multistep.py — FAST batch evaluation of multi-step model
Only first window per file, batch processing on GPU
"""
import os, sys, torch, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data_pipeline_v2 import _build_features, THRUST_SCALE, MFR_MAX, INPUT_DIM
from model_train_multistep import DualOutputLSTMMultiStep, INPUT_LEN, MAX_HORIZON, SEQ_LEN, HORIZONS

class MultiStepTestDS(Dataset):
    def __init__(self, md, dr, n_max=500):
        self.samples=[]
        df=pd.read_csv(md); df['filename']=df['filename'].str.strip()
        ex=set(os.listdir(dr))
        df=df[(df['anomalous']==False)&(df['sn']>=13)&df['filename'].isin(ex)]
        for _,r in df.iterrows():
            fp=os.path.join(dr,r['filename'])
            if not os.path.exists(fp): continue
            try: d=pd.read_csv(fp)
            except: continue
            T=len(d)
            if T<SEQ_LEN: continue
            xf,yf,_=_build_features(d.iloc[:SEQ_LEN],r)
            self.samples.append((xf[:INPUT_LEN].astype(np.float32),
                                 yf.astype(np.float32), r['test_mode'], r['sn'], r['test_pressure']))
            if len(self.samples)>=n_max: break
        print(f"  Test samples: {len(self.samples)}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y,mode,sn,pr=self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y), mode, sn, pr

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    mp="outputs/models/v2/dual_output_lstm_multistep.pth"
    model=DualOutputLSTMMultiStep().to(device)
    model.load_state_dict(torch.load(mp,map_location=device)); model.eval()

    ds=MultiStepTestDS("data/metadata.csv","data/dataset/dataset/test/")
    ld=DataLoader(ds,64,False)

    # Per-horizon accumulators
    metrics={h:{'t_rmse':[],'t_mae':[],'m_rmse':[],'m_mae':[]} for h in HORIZONS}
    per_mode={h:{} for h in HORIZONS}

    with torch.no_grad():
        for bi,(x,y,modes,sns,prs) in enumerate(ld):
            if bi%10==0: print(f"  [{bi*64}/{len(ds)}]",flush=True)
            x=x.to(device); pred=model(x).cpu().numpy()  # [B,200,8]
            y=y.numpy()
            for h_idx,h in enumerate(HORIZONS):
                vl=INPUT_LEN-h
                if vl<=0: continue
                for b in range(len(x)):
                    # thrust
                    pt=pred[b,:vl,2*h_idx]*THRUST_SCALE
                    tt=y[b,h:INPUT_LEN,0]*THRUST_SCALE
                    metrics[h]['t_rmse'].append(np.sqrt(np.mean((pt-tt)**2)))
                    metrics[h]['t_mae'].append(np.mean(np.abs(pt-tt)))
                    # mfr
                    pm=pred[b,:vl,2*h_idx+1]*MFR_MAX
                    tm=y[b,h:INPUT_LEN,1]*MFR_MAX
                    metrics[h]['m_rmse'].append(np.sqrt(np.mean((pm-tm)**2)))
                    metrics[h]['m_mae'].append(np.mean(np.abs(pm-tm)))
                    # per-mode
                    mode=modes[b]
                    if mode not in per_mode[h]:
                        per_mode[h][mode]={'t_rmse':[],'m_rmse':[]}
                    per_mode[h][mode]['t_rmse'].append(np.sqrt(np.mean((pt-tt)**2)))
                    per_mode[h][mode]['m_rmse'].append(np.sqrt(np.mean((pm-tm)**2)))

    n=len(metrics[HORIZONS[0]]['t_rmse'])
    print(f"\n  Evaluated {n} samples\n")
    print(f"{'='*70}")
    print(f"  Multi-Step Results ({n} windows, SN13-24)")
    print(f"{'='*70}")
    print(f"  {'Horizon':>8s}  {'Thrust RMSE':>13s}  {'Thrust MAE':>12s}  {'MFR RMSE':>13s}  {'MFR MAE':>12s}")
    trm=[]; mrm=[]
    for h in HORIZONS:
        m=metrics[h]
        tr=np.mean(m['t_rmse']); tm=np.mean(m['t_mae'])
        mr=np.mean(m['m_rmse']); mm=np.mean(m['m_mae'])
        trm.append(tr); mrm.append(mr)
        print(f"  t+{h:5d}  {tr:13.4f} N  {tm:12.4f} N  {mr:13.1f} mg/s  {mm:12.1f} mg/s")

    print(f"\n  Error growth:")
    for i in range(1,len(HORIZONS)):
        print(f"    t+1 -> t+{HORIZONS[i]}: thrust x{trm[i]/trm[0]:.2f}, MFR x{mrm[i]/mrm[0]:.2f}")
    print(f"  v2.0 reference: thrust=0.0832N, mfr=49.1mg/s (t+0, current-step)")
    print("="*70)

    # ── Plots ──
    fig_dir="outputs/figures/v2"; os.makedirs(fig_dir,exist_ok=True)

    # Plot 1: RMSE vs Horizon
    fig,ax1=plt.subplots(figsize=(10,6))
    ax1.plot(HORIZONS,trm,'o-',color='#2196F3',lw=2,ms=8,label='Thrust RMSE (N)')
    ax1.set_xlabel('Prediction Horizon (steps)'); ax1.set_ylabel('Thrust RMSE (N)',color='#2196F3')
    ax1.tick_params(axis='y',labelcolor='#2196F3')
    ax2=ax1.twinx()
    ax2.plot(HORIZONS,mrm,'s--',color='#FF5722',lw=2,ms=8,label='MFR RMSE (mg/s)')
    ax2.set_ylabel('MFR RMSE (mg/s)',color='#FF5722'); ax2.tick_params(axis='y',labelcolor='#FF5722')
    ax1.set_title('Multi-Step Prediction: RMSE vs Horizon\n(Attention-LSTM, SN13-24, 100 Hz)',fontsize=14,fontweight='bold')
    ax1.grid(True,ls=':',alpha=0.5)
    for i,h in enumerate(HORIZONS):
        ax1.annotate(f'{trm[i]:.4f}',(h,trm[i]),textcoords="offset points",xytext=(0,12),ha='center',fontsize=8,color='#2196F3')
        ax2.annotate(f'{mrm[i]:.1f}',(h,mrm[i]),textcoords="offset points",xytext=(0,-15),ha='center',fontsize=8,color='#FF5722')
    lines1,labels1=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2,labels1+labels2,loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir,"multistep_rmse_curve.png"),dpi=300); plt.close()
    print(f"Figure: {fig_dir}/multistep_rmse_curve.png")

    # Plot 2: Per-mode breakdown at t+5
    h=5
    modes_sorted=sorted(per_mode[HORIZONS[1]].keys())
    fig,ax=plt.subplots(figsize=(14,5))
    xp=np.arange(len(modes_sorted)); w=0.35
    t_vals=[np.mean(per_mode[HORIZONS[1]][m]['t_rmse']) for m in modes_sorted]
    m_vals=[np.mean(per_mode[HORIZONS[1]][m]['m_rmse']) for m in modes_sorted]
    b1=ax.bar(xp-w/2,t_vals,w,color='#2196F3',alpha=.85,label='Thrust RMSE (N)')
    ax2b=ax.twinx()
    b2=ax2b.bar(xp+w/2,m_vals,w,color='#FF5722',alpha=.85,label='MFR RMSE (mg/s)')
    ax.set_xticks(xp); ax.set_xticklabels(modes_sorted,rotation=45,ha='right')
    ax.set_ylabel('Thrust RMSE (N)',color='#2196F3'); ax2b.set_ylabel('MFR RMSE (mg/s)',color='#FF5722')
    ax.set_title(f'Multi-Step RMSE by Test Mode (t+{h}, {n} windows)',fontsize=13,fontweight='bold')
    ax.grid(axis='y',alpha=0.3,ls='--')
    fig.legend(loc='upper right',bbox_to_anchor=(0.95,0.85))
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir,"multistep_by_mode.png"),dpi=300); plt.close()
    print(f"Figure: {fig_dir}/multistep_by_mode.png")
    print("Done.")

if __name__=="__main__": main()
