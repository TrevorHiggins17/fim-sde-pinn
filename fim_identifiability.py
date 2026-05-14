import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

RESULTS_DIR = "./results_fim"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif","Times New Roman","Times","serif"],
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     8.5,
    "axes.titleweight":  "bold",
    "axes.linewidth":     0.7,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   0,
    "ytick.major.size":   0,
    "legend.fontsize":    7,
    "legend.framealpha":  0.9,
    "legend.edgecolor":  "0.75",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

COLOUR_IDENT  = "#2166ac"   # blue  — identifiable
COLOUR_CODEP  = "#d6604d"   # muted red-orange — co-dependent
COLOUR_NONID  = "#999999"   # grey  — non-identifiable
COLOUR_THRESH = "#1b7837"   # green — threshold line

# All 31 model parameters from lit
p = dict(
    mu_max=0.106, kiA=0.109, KsI=1.4,
    KsA=1.789, kiI=186.52, sigma_light=1.0,
    qN0=0.876, YXA=0.059,
    rhoN_max=40.445, Kstar=0.3125,
    n=18.183, phiN=137.455,
    KsN=0.162, kiN=0.113,
    KsA_N=1.004, kiA_N=1.098,
    r1=0.0420, r2=0.00410,
    r4=0.0049, KsS=0.0,
    kiS=0.2079, nS=3.6205,
    k1=0.1771, phiS=0.6675,
    r3=0.162, KsL=0.0227,
    kiL=0.0861, nL=1.8117,
    k2=0.2135, phiL=0.0,
    KH=4.653,
)

ALL_PARAMS_31 = [
    "mu_max","KsA","kiA","KsI","kiI","sigma_light",
    "rhoN_max","Kstar","n","phiN","KsN","kiN",
    "KsA_N","kiA_N","qN0","YXA",
    "r1","r2","r4","KsS","kiS","nS","k1","phiS",
    "r3","KsL","kiL","nL","k2","phiL","KH",
]
assert len(ALL_PARAMS_31) == 31

def make_label(name):
    label_map = {
        "mu_max":r"$\mu_{\mathrm{max}}$","KsA":r"$K_{sA}$","kiA":r"$k_{iA}$",
        "KsI":r"$K_{sI}$","kiI":r"$k_{iI}$","sigma_light":r"$\sigma_\ell$",
        "rhoN_max":r"$\rho_{N,\mathrm{max}}$","Kstar":r"$K^*$","n":r"$n$",
        "phiN":r"$\varphi_N$","KsN":r"$K_{sN}$","kiN":r"$k_{iN}$",
        "KsA_N":r"$K_{sA,N}$","kiA_N":r"$k_{iA,N}$","qN0":r"$q_{N,0}$",
        "YXA":r"$Y_{XA}$","r1":r"$r_1$","r2":r"$r_2$","r4":r"$r_4$",
        "KsS":r"$K_{sS}$","kiS":r"$k_{iS}$","nS":r"$n_S$","k1":r"$k_1$",
        "phiS":r"$\varphi_S$","r3":r"$r_3$","KsL":r"$K_{sL}$","kiL":r"$k_{iL}$",
        "nL":r"$n_L$","k2":r"$k_2$","phiL":r"$\varphi_L$","KH":r"$K_H$",
    }
    return label_map.get(name, name)

PARAM_LABELS = {pname: make_label(pname) for pname in ALL_PARAMS_31}

CONDITIONS = {
    "TAP":   {"A0":0.41913,"N0":0.38187},
    "N-":    {"A0":0.41913,"N0":0.35497},
    "A++":   {"A0":1.26341,"N0":0.38187},
    "A'-N'": {"A0":1.16514,"N0":0.31520},
}
I0=125.0; z=0.05; X_SEED=5e-4; QN_IC=0.99185; H_IC=6.99818; T_END=200.0
IDX7=[0,1,2,3,4,5,7]
STATE_LABELS=["Biomass","Nitrogen","N quota","Acetate","Starch","Lipids","pH"]
STATE_LABELS_SHORT=["X","N",r"$q_N$","A","S","L","pH"]
DIGIT_MAP={"Biomass":0,"Nitrogen":1,"N quota":2,"Acetate":3,"Starch":4,"Lipids":5,"pH":6}
DIGITISED_CSV="fig3_digitized_tidy.csv"

# ODE 
def _pos(x): return x if x>0.0 else 0.0
def andrews_np(u,Ks,ki):
    u=_pos(u); return u/(u+Ks+u*u/ki+1e-20)
def g_int_N_np(Ni,Ks,ki,n_exp):
    Ni=_pos(Ni)
    if Ni<1e-20: return 0.0
    Nin=Ni**n_exp; Ksn=Ks**n_exp if Ks>1e-20 else 0.0
    denom=Nin+Ksn+(Ni*Ni/ki)**n_exp
    return Nin/denom if denom>1e-20 else 0.0
def ode_rhs(t,y,A0,N0,params):
    X,N,qN,A,S,L,xact,H=y
    Xw=_pos(X);Nw=_pos(N);Aw=_pos(A);xactw=_pos(xact)
    qNw=max(qN,params["qN0"]+1e-12)
    I=I0*np.exp(-params["sigma_light"]*Xw*z)
    wH_n=Aw/params["KsA"];wI_n=I/params["KsI"];dw=wH_n+wI_n
    wH=0.5 if dw<1e-14 else wH_n/dw
    muH=andrews_np(Aw,params["KsA"],params["kiA"])
    muI=andrews_np(I,params["KsI"],params["kiI"])
    mu=max(params["mu_max"]*(wH*muH+(1-wH)*muI)*(1-params["qN0"]/qNw),0.0)
    rho_bar=(params["rhoN_max"]*N0**params["n"]
             /(N0**params["n"]+params["Kstar"]**params["n"])
             *np.exp(-params["phiN"]*Xw))
    rho=(rho_bar*andrews_np(Nw,params["KsN"],params["kiN"])
         *andrews_np(Aw,params["KsA_N"],params["kiA_N"]))
    Ni=qNw*Xw;Aint=max(A0-Aw,0.0);Nn=Nw/N0
    gS=g_int_N_np(Ni,params["KsS"],params["kiS"],params["nS"])
    gL=g_int_N_np(Ni,params["KsL"],params["kiL"],params["nL"])
    R1=params["r1"]*gS*(params["k1"]/(params["k1"]+Nn+1e-12))*(mu+np.exp(params["phiS"]*Aint))*xactw
    R3=params["r3"]*gL*(params["k2"]/(params["k2"]+Nn+1e-12))*(mu+np.exp(params["phiL"]*Aint))*xactw
    R2=(params["r2"]/qNw)*Xw;R4=(params["r4"]/qNw)*Xw
    dX=mu*Xw;dN=-rho*Xw;dqN=rho-mu*qNw
    dA=-(1/params["YXA"])*muH/(muH+muI+1e-14)*dX
    dxact=dX+R2+R4-R1-R3;dS=R1-R2;dL=R3-R4;dH=params["KH"]*dxact
    return [dX,dN,dqN,dA,dS,dL,dxact,dH]

def run_ode(params,cfg,t_eval):
    y0=np.array([X_SEED,cfg["N0"],QN_IC,cfg["A0"],0.0,0.0,X_SEED,H_IC])
    sol=solve_ivp(lambda t,y:ode_rhs(t,y,cfg["A0"],cfg["N0"],params),
                  t_span=(0.0,T_END),y0=y0,method="LSODA",
                  rtol=1e-8,atol=1e-10,dense_output=True,t_eval=t_eval)
    assert sol.status==0
    return sol.y

# Load data
df_all=pd.read_csv(DIGITISED_CSV)
df_all.columns=df_all.columns.str.strip()
for col in ["condition","variable"]:
    df_all[col]=df_all[col].astype(str).str.strip()
df_all["time_h"]=pd.to_numeric(df_all["time_h"],errors="coerce")
df_all["value"]=pd.to_numeric(df_all["value"],errors="coerce")
df_all=df_all.dropna(subset=["time_h","value"])
cond_map={}
for c in df_all["condition"].unique():
    cond_map[c]=(c.replace("\u2019","'").replace("\u2018","'")
                  .replace("\u2013","-").replace("\u2014","-")
                  .replace("\u2032","'"))
df_all["condition"]=df_all["condition"].map(cond_map).fillna(df_all["condition"])

# Nominal trajectories 
print("Running nominal LSODA trajectories...")
t_dense=np.linspace(0,T_END,2001)
nominal={}
for cname,cfg in CONDITIONS.items():
    nominal[cname]=run_ode(p,cfg,t_dense)
    print(f"  {cname}: X_fin={nominal[cname][0,-1]:.4f}")
scale7=np.array([max(nominal[c][i,:].max() for c in CONDITIONS)+1e-10 for i in IDX7],dtype=float)

data_points={}
for cname in CONDITIONS:
    data_points[cname]={}
    for varname,state_idx in DIGIT_MAP.items():
        pts=df_all[(df_all["condition"]==cname)&(df_all["variable"]==varname)].sort_values("time_h")
        if not pts.empty:
            data_points[cname][state_idx]=pts["time_h"].values

EPS=1e-4


#Stage 0 - Full 31-Parameter Sens Screen
print("\n"+"="*65)
print("STAGE 0 — Full 31-parameter sensitivity screen")
print("="*65)

t_screen=np.linspace(1,T_END,200)
sens_total_31={}
sens_heatmap_31=np.zeros((31,7))

for pi,pname in enumerate(ALL_PARAMS_31):
    theta0=p[pname]
    if abs(theta0)<1e-12:
        p_up=dict(p);p_up[pname]=1e-6
        p_dn=dict(p);p_dn[pname]=-1e-6
        denom=2e-6
    else:
        p_up=dict(p);p_up[pname]=theta0*(1+EPS)
        p_dn=dict(p);p_dn[pname]=theta0*(1-EPS)
        denom=2*EPS*theta0
    total_sens=0.0
    for cname,cfg in CONDITIONS.items():
        y_up=run_ode(p_up,cfg,t_screen)
        y_dn=run_ode(p_dn,cfg,t_screen)
        for si,ode_idx in enumerate(IDX7):
            dydp=(y_up[ode_idx,:]-y_dn[ode_idx,:])/denom
            s=np.abs(dydp*theta0/scale7[si]) if abs(theta0)>1e-12 else np.abs(dydp/scale7[si])
            sens_heatmap_31[pi,si]+=s.mean()
            total_sens+=s.sum()
    sens_total_31[pname]=total_sens
    print(f"  {pname:<14} total_sens={total_sens:.4f}")

sens_heatmap_31/=len(CONDITIONS)
max_sens=max(sens_total_31.values())
SCREEN_THRESH=0.01*max_sens
retained=[p_ for p_,s in sens_total_31.items() if s>=SCREEN_THRESH]
excluded=[p_ for p_,s in sens_total_31.items() if s<SCREEN_THRESH]
print(f"\nRetained ({len(retained)}): {retained}")
print(f"Excluded ({len(excluded)}): {excluded}")

CANDIDATE_PARAMS=retained
N_PARAMS=len(CANDIDATE_PARAMS)

# Figure 0 — 31-parameter screen 
sort_31=sorted(range(31),key=lambda i:sum(sens_heatmap_31[i,:]),reverse=True)
sens_sorted_31=sens_heatmap_31[sort_31,:]
params_sorted_31=[ALL_PARAMS_31[i] for i in sort_31]
status_31=["excluded" if ALL_PARAMS_31[i] in excluded else "candidate" for i in sort_31]

fig0,ax0=plt.subplots(figsize=(4.5,8.5))
vmax_31=np.percentile(sens_sorted_31[sens_sorted_31>0],98) if sens_sorted_31.max()>0 else 1.0
im0=ax0.imshow(sens_sorted_31,aspect="auto",cmap="Blues",interpolation="nearest",vmin=0,vmax=vmax_31)
ax0.set_xticks(range(7)); ax0.set_xticklabels(STATE_LABELS_SHORT,fontsize=7.5)
ax0.xaxis.set_ticks_position("top"); ax0.xaxis.set_label_position("top")
ax0.set_yticks(range(31))
ax0.set_yticklabels([make_label(p_) for p_ in params_sorted_31],fontsize=7)
for tick,status in zip(ax0.get_yticklabels(),status_31):
    tick.set_color(COLOUR_IDENT if status=="candidate" else COLOUR_NONID)
    if status=="candidate": tick.set_fontweight("bold")
for i in range(31):
    for j in range(7):
        v=sens_sorted_31[i,j]
        if v>0.005:
            ax0.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=4.5,
                     color="white" if v>vmax_31*0.6 else "#1a1a1a")
n_ret=sum(1 for s in status_31 if s=="candidate")
ax0.axhline(n_ret-0.5,color=COLOUR_CODEP,lw=1.5,ls="--")
ax0.text(6.6,n_ret-0.5,"",fontsize=6,color=COLOUR_CODEP,va="center")
for i in range(32): ax0.axhline(i-0.5,color="white",lw=0.4)
for j in range(8):  ax0.axvline(j-0.5,color="white",lw=0.4)
ax0.set_title("Full 31-parameter sensitivity screen",loc="left",pad=10)
ax0.legend(handles=[
    Patch(facecolor=COLOUR_IDENT,edgecolor="none",label=f"Retained ({len(retained)})"),
    Patch(facecolor=COLOUR_NONID,edgecolor="none",label=f"Excluded ({len(excluded)})"),
],fontsize=6.5,loc="lower right",frameon=True,handlelength=1.0)
plt.colorbar(im0,ax=ax0,orientation="vertical",pad=0.03,fraction=0.025,aspect=40
    ).set_label(r"Mean $|\partial y/\partial\theta \cdot \theta/\mathrm{scale}|$",fontsize=6.5)
plt.tight_layout()
fig0.savefig(os.path.join(RESULTS_DIR,"fim_fig0_31param_screen.png"),dpi=300)
plt.show()
plt.close(fig0)
print("Saved: fim_fig0_31param_screen.png")

# Stage - FIM on the candidate set
print("\n"+"="*65)
print(f"STAGE 1 — FIM on {N_PARAMS}-parameter candidate set")
print("="*65)

sens_matrix_full={}
for pi,pname in enumerate(CANDIDATE_PARAMS):
    theta0=p[pname]
    if abs(theta0)<1e-12:
        p_up=dict(p);p_up[pname]=1e-6
        p_dn=dict(p);p_dn[pname]=-1e-6
        denom_s=2e-6; zero_param=True
    else:
        p_up=dict(p);p_up[pname]=theta0*(1+EPS)
        p_dn=dict(p);p_dn[pname]=theta0*(1-EPS)
        denom_s=2*EPS*theta0; zero_param=False
    sens_matrix_full[pname]={}
    for cname,cfg in CONDITIONS.items():
        all_times=np.unique(np.concatenate([ts for ts in data_points[cname].values()])) if data_points[cname] else np.array([])
        if len(all_times)==0: continue
        y_up=run_ode(p_up,cfg,all_times); y_dn=run_ode(p_dn,cfg,all_times)
        S=np.zeros((7,len(all_times)))
        for si,ode_idx in enumerate(IDX7):
            dydp=(y_up[ode_idx,:]-y_dn[ode_idx,:])/denom_s
            S[si,:]=dydp/scale7[si] if zero_param else dydp*theta0/scale7[si]
        sens_matrix_full[pname][cname]=(all_times,S)
    print(f"  {pname:<12} done")

datapoint_keys=[]
for ci,(cname,cfg) in enumerate(CONDITIONS.items()):
    for si,(varname,state_idx) in enumerate(DIGIT_MAP.items()):
        if state_idx not in data_points[cname]: continue
        for t in data_points[cname][state_idx]:
            datapoint_keys.append((ci,cname,si,state_idx,t))
N_DATAPOINTS=len(datapoint_keys)
S_matrix=np.zeros((N_DATAPOINTS,N_PARAMS))
for pi,pname in enumerate(CANDIDATE_PARAMS):
    for ci,(cname,cfg) in enumerate(CONDITIONS.items()):
        if cname not in sens_matrix_full[pname]: continue
        all_times,S=sens_matrix_full[pname][cname]
        for si,(varname,state_idx) in enumerate(DIGIT_MAP.items()):
            if state_idx not in data_points[cname]: continue
            t_data=data_points[cname][state_idx]
            s_interp=np.interp(t_data,all_times,S[si,:])
            for k,(kci,kcname,ksi,kstate,kt) in enumerate(datapoint_keys):
                if kci==ci and ksi==si and kstate==state_idx:
                    t_idx=np.searchsorted(t_data,kt)
                    if t_idx<len(s_interp):
                        S_matrix[k,pi]=s_interp[t_idx]

FIM=S_matrix.T@S_matrix
diag_fim=np.diag(FIM)
D=np.sqrt(diag_fim); D_safe=np.where(D<1e-20,1.0,D)
corr=FIM/np.outer(D_safe,D_safe); np.fill_diagonal(corr,1.0)
eigvals,eigvecs=np.linalg.eigh(FIM)
eigvals=np.maximum(eigvals,0.0)
order=np.argsort(eigvals)[::-1]; eigvals=eigvals[order]; eigvecs=eigvecs[:,order]
pos_eigs=eigvals[eigvals>1e-30]
cond_number=pos_eigs[0]/pos_eigs[-1] if len(pos_eigs)>1 else np.inf
norm_scores=np.where(np.isnan(diag_fim),0.0,diag_fim/(np.nanmax(diag_fim)+1e-30))

THRESH_SCREEN=0.05; THRESH_COLLINEAR=0.90
classifications=[]
for i,pname in enumerate(CANDIDATE_PARAMS):
    score=norm_scores[i]
    max_corr=max(abs(corr[i,j]) for j in range(N_PARAMS) if j!=i)
    if score<THRESH_SCREEN: cls="Non-identifiable"
    elif max_corr>THRESH_COLLINEAR: cls="Co-dependent"
    else: cls="Identifiable"
    classifications.append(cls)

identifiable=[CANDIDATE_PARAMS[i] for i in range(N_PARAMS) if classifications[i]=="Identifiable"]
codependent=[CANDIDATE_PARAMS[i]  for i in range(N_PARAMS) if classifications[i]=="Co-dependent"]
nonid=[CANDIDATE_PARAMS[i]        for i in range(N_PARAMS) if classifications[i]=="Non-identifiable"]

print(f"\n  κ(F) = {cond_number:.2e}")
print(f"  Identifiable   ({len(identifiable)}): {identifiable}")
print(f"  Co-dependent   ({len(codependent)}):  {codependent}")
print(f"  Non-identifiable ({len(nonid)}): {nonid}")

rows=[]
for i,pname in enumerate(CANDIDATE_PARAMS):
    max_corr=max(abs(corr[i,j]) for j in range(N_PARAMS) if j!=i)
    mc_idx=int(np.argmax([abs(corr[i,j]) if j!=i else 0 for j in range(N_PARAMS)]))
    rows.append({"parameter":pname,"fim_diagonal":diag_fim[i],
                 "normalised_score":norm_scores[i],"max_abs_correlation":max_corr,
                 "most_correlated_with":CANDIDATE_PARAMS[mc_idx],
                 "classification":classifications[i],"literature_value":p[pname]})
pd.DataFrame(rows).sort_values("fim_diagonal",ascending=False
    ).to_csv(os.path.join(RESULTS_DIR,"fim_results.csv"),index=False)

# Figure 1 - Sensitivity heatmap 
sens_heatmap=np.zeros((N_PARAMS,7))
for pi,pname in enumerate(CANDIDATE_PARAMS):
    for si in range(7):
        vals=[]
        for cname in CONDITIONS:
            if cname not in sens_matrix_full[pname]: continue
            all_times,S=sens_matrix_full[pname][cname]
            vals.extend(np.abs(S[si,:]).tolist())
        sens_heatmap[pi,si]=np.mean(vals) if vals else 0.0

sort_order=np.argsort(sens_heatmap.sum(axis=1))[::-1]
sens_sorted=sens_heatmap[sort_order,:]
params_sorted=[CANDIDATE_PARAMS[i] for i in sort_order]
cls_sorted=[classifications[i] for i in sort_order]

fig1,ax1=plt.subplots(figsize=(3.5,5.5))
im1=ax1.imshow(sens_sorted,aspect="auto",cmap="Blues",interpolation="nearest",
               vmin=0,vmax=np.percentile(sens_sorted,98))
for i in range(N_PARAMS):
    for j in range(7):
        v=sens_sorted[i,j]
        if v>0.005:
            ax1.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=5.5,
                     color="white" if v>np.percentile(sens_sorted,72) else "#1a1a1a")
ax1.set_yticks(range(N_PARAMS))
ax1.set_yticklabels([PARAM_LABELS[p_] for p_ in params_sorted],fontsize=7.5)
for tick,cls in zip(ax1.get_yticklabels(),cls_sorted):
    if cls=="Identifiable": tick.set_color(COLOUR_IDENT); tick.set_fontweight("bold")
    elif cls=="Co-dependent": tick.set_color(COLOUR_CODEP); tick.set_fontweight("bold")
    else: tick.set_color(COLOUR_NONID)
ax1.set_xticks(range(7)); ax1.set_xticklabels(STATE_LABELS_SHORT,fontsize=7.5)
ax1.xaxis.set_ticks_position("top"); ax1.xaxis.set_label_position("top")
for i in range(N_PARAMS+1): ax1.axhline(i-0.5,color="white",lw=0.5)
for j in range(8): ax1.axvline(j-0.5,color="white",lw=0.5)
for i,cls in enumerate(cls_sorted):
    col=COLOUR_IDENT if cls=="Identifiable" else COLOUR_CODEP if cls=="Co-dependent" else None
    if col: ax1.add_patch(plt.Rectangle((-0.5,i-0.5),7,1,fill=False,edgecolor=col,lw=1.2,clip_on=False))
ax1.set_title(f"(a)  Normalised sensitivity heatmap\n17 candidates  |  κ(F) = {cond_number:.2e}",pad=10,loc="left")
ax1.legend(handles=[
    Patch(facecolor=COLOUR_IDENT,edgecolor="none",label="Identifiable"),
    Patch(facecolor=COLOUR_CODEP,edgecolor="none",label="Co-dependent"),
    Patch(facecolor=COLOUR_NONID,edgecolor="none",label="Non-identifiable"),
],fontsize=6.5,loc="upper left",bbox_to_anchor=(0.0,-0.03),ncol=1,frameon=True,handlelength=1.0)
plt.colorbar(im1,ax=ax1,orientation="vertical",pad=0.03,fraction=0.03,aspect=30
    ).set_label(r"Mean $|\partial y/\partial\theta \cdot \theta/\mathrm{scale}|$",fontsize=7,labelpad=6)
plt.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR,"fim_fig1_sensitivity_heatmap_pub.png"),dpi=300)
plt.show()
plt.close(fig1)
print("Saved: fim_fig1_sensitivity_heatmap_pub.png")

# Figure 2 - Correlation matrix 
corr_sorted=corr[np.ix_(sort_order,sort_order)]
param_labels_sorted=[PARAM_LABELS[p_] for p_ in params_sorted]
fig2,ax2=plt.subplots(figsize=(4.2,5.0))
corr_display=np.abs(corr_sorted)
im2=ax2.imshow(corr_display,aspect="auto",cmap="Reds",interpolation="nearest",vmin=0,vmax=1)
for i in range(N_PARAMS):
    for j in range(N_PARAMS):
        v=corr_display[i,j]
        ax2.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=5.0,
                 color="white" if v>0.75 else "#1a1a1a")
        if i!=j and v>THRESH_COLLINEAR:
            ax2.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor=COLOUR_CODEP,lw=1.5))
for i in range(N_PARAMS):
    ax2.add_patch(plt.Rectangle((i-0.5,i-0.5),1,1,facecolor="#e8e8e8",edgecolor="white",lw=0.5))
    ax2.text(i,i,"1.00",ha="center",va="center",fontsize=5.0,color="#999999")
ax2.set_xticks(range(N_PARAMS)); ax2.set_xticklabels(param_labels_sorted,fontsize=7,rotation=45,ha="right")
ax2.set_yticks(range(N_PARAMS)); ax2.set_yticklabels(param_labels_sorted,fontsize=7)
for i in range(N_PARAMS+1): ax2.axhline(i-0.5,color="white",lw=0.5); ax2.axvline(i-0.5,color="white",lw=0.5)
cb2=plt.colorbar(im2,ax=ax2,orientation="horizontal",pad=0.22,fraction=0.04,aspect=25)
cb2.set_label(r"$|r_{ij}|$ — absolute correlation",fontsize=7); cb2.ax.tick_params(labelsize=6.5)
cb2.set_ticks([0,0.25,0.50,0.75,1.0])
ax2.set_title(f"(b)  FIM correlation matrix  |  co-dependent pairs: |r| > {THRESH_COLLINEAR}",pad=6,loc="left")
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR,"fim_fig2_correlation_matrix_pub.png"),dpi=300)
plt.show()
plt.close(fig2)
print("Saved: fim_fig2_correlation_matrix_pub.png")

# Figure 3 - Eigenvalue spectrum 
fig3,(ax3a,ax3b)=plt.subplots(1,2,figsize=(7.0,3.2))
fig3.subplots_adjust(left=0.09,right=0.97,bottom=0.18,top=0.88,wspace=0.38)
colors3=[COLOUR_IDENT if v/eigvals[0]>0.05 else COLOUR_CODEP if v/eigvals[0]>0.005 else COLOUR_NONID for v in eigvals]
ax3a.bar(np.arange(N_PARAMS),eigvals,color=colors3,edgecolor="white",width=0.7)
ax3a.set_yscale("log")
ax3a.axhline(eigvals[0]*0.05, color=COLOUR_IDENT,lw=1.0,ls="--",label=r"5% of $\lambda_{\max}$")
ax3a.axhline(eigvals[0]*0.005,color=COLOUR_NONID, lw=1.0,ls=":", label=r"0.5% of $\lambda_{\max}$")
ax3a.set_xlabel("Eigenvalue index (descending)"); ax3a.set_ylabel(r"$\lambda_i$ (log scale)")
ax3a.set_title(r"(a)  Eigenvalue spectrum  |  $\kappa(\mathbf{F})$"+f" = {cond_number:.2e}",loc="left")
ax3a.legend(fontsize=7); ax3a.spines["top"].set_visible(False); ax3a.spines["right"].set_visible(False)
n_show=min(5,N_PARAMS); colors_ev=plt.cm.tab10(np.linspace(0,0.5,n_show)); bottom=np.zeros(N_PARAMS)
for k in range(n_show-1,-1,-1):
    ev=eigvecs[:,-(k+1)]
    ax3b.bar(np.arange(N_PARAMS),ev**2,bottom=bottom,color=colors_ev[k],alpha=0.85,
             label=rf"$\mathbf{{v}}_{{{N_PARAMS-k}}}$  ($\lambda={eigvals[-(k+1)]:.1e}$)")
    bottom+=ev**2
ax3b.set_xticks(np.arange(N_PARAMS))
ax3b.set_xticklabels([PARAM_LABELS[p_] for p_ in CANDIDATE_PARAMS],fontsize=7,rotation=45,ha="right")
ax3b.set_ylabel(r"Squared component $v_{ij}^2$")
ax3b.set_title("(b)  Null-space structure — 5 smallest eigenvectors",loc="left")
ax3b.legend(fontsize=6.5,loc="upper right"); ax3b.spines["top"].set_visible(False); ax3b.spines["right"].set_visible(False)
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR,"fim_fig3_eigenvalues_pub.png"),dpi=300)
plt.show()
plt.close(fig3)
print("Saved: fim_fig3_eigenvalues_pub.png")

# ── Figure 4 - Parameter ranking ─────────────────────────────────
sort_idx4=np.argsort(norm_scores)[::-1]
sorted_names4=[CANDIDATE_PARAMS[i] for i in sort_idx4]
sorted_scores4=norm_scores[sort_idx4]
sorted_cls4=[classifications[i] for i in sort_idx4]
bar_colors4=[COLOUR_IDENT if c=="Identifiable" else COLOUR_CODEP if c=="Co-dependent" else COLOUR_NONID for c in sorted_cls4]
fig4,ax4=plt.subplots(figsize=(6.5,3.0))
fig4.subplots_adjust(left=0.08,right=0.97,bottom=0.22,top=0.88)
bars4=ax4.bar(range(N_PARAMS),sorted_scores4,color=bar_colors4,edgecolor="white",width=0.65)
ax4.axhline(THRESH_SCREEN,color=COLOUR_THRESH,lw=1.2,ls="--")
ax4.text(N_PARAMS-0.5,THRESH_SCREEN+0.02,
         rf"$\delta_i = {THRESH_SCREEN:.2f}$ — identifiability threshold",
         ha="right",va="bottom",fontsize=7,color=COLOUR_THRESH)
for bar,score in zip(bars4,sorted_scores4):
    if score>0.002:
        ax4.text(bar.get_x()+bar.get_width()/2,score+0.012,f"{score:.3f}",
                 ha="center",va="bottom",fontsize=6,color="#1a1a1a")
ax4.set_xticks(range(N_PARAMS)); ax4.set_xticklabels([PARAM_LABELS[n] for n in sorted_names4],fontsize=8)
ax4.set_ylabel(r"Normalised FIM diagonal score $\delta_i$"); ax4.set_ylim(0,1.18)
ax4.set_title("(c)  Parameter identifiability ranking — FIM diagonal scores",loc="left")
ax4.legend(handles=[
    Patch(facecolor=COLOUR_IDENT,edgecolor="white",label=f"Identifiable ({len(identifiable)})"),
    Patch(facecolor=COLOUR_CODEP,edgecolor="white",label=f"Co-dependent ({len(codependent)})"),
    Patch(facecolor=COLOUR_NONID,edgecolor="white",label=f"Non-identifiable ({len(nonid)})"),
],fontsize=7.5,loc="upper right")
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
ax4.grid(True,axis="y",lw=0.4,alpha=0.35)
plt.tight_layout()
fig4.savefig(os.path.join(RESULTS_DIR,"fim_fig4_ranking_pub.png"),dpi=300)
plt.show()
plt.close(fig4)
print("Saved: fim_fig4_ranking_pub.png")

print(f"\n{'='*65}")
print("COMPLETE")
print(f"  Stage 0: {len(retained)}/31 retained, {len(excluded)} excluded")
print(f"  Stage 1: κ(F) = {cond_number:.2e}")
print(f"  Identifiable:    {identifiable}")
print(f"  Co-dependent:    {codependent}")
print(f"  Non-identifiable: {nonid}")
print(f"  Outputs: {RESULTS_DIR}/") 