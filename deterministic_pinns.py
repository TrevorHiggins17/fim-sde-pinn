import os, time, copy
import numpy as np
import deepxde as dde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
import pandas as pd
from scipy.integrate import solve_ivp

dde.backend.set_default_backend("pytorch")

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

matplotlib.rcParams.update({
    "font.family":"serif","font.serif":["DejaVu Serif","Times New Roman"],
    "font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
    "axes.titleweight":"bold","axes.linewidth":0.7,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":7,"ytick.labelsize":7,
    "xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.major.size":3.0,"ytick.major.size":3.0,
    "xtick.direction":"out","ytick.direction":"out",
    "legend.fontsize":7,"legend.framealpha":0.9,"legend.edgecolor":"0.75",
    "lines.linewidth":1.4,"figure.dpi":150,
    "savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05,
    "grid.linewidth":0.4,"grid.alpha":0.35,"grid.color":"0.70",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    N_COL=1200; N_ANCHORS=200; N_ADAM=12000
else:
    N_COL=400;  N_ANCHORS=120; N_ADAM=8000

# Parameters
p = dict(
    mu_max=0.106, kiA=0.109, KsI=1.4, KsA=1.789, kiI=186.52,
    sigma_light=1.0, qN0=0.876, YXA=0.059, rhoN_max=40.445,
    Kstar=0.3125, n=18.183, phiN=137.455, KsN=0.162, kiN=0.113,
    KsA_N=1.004, kiA_N=1.098, r1=0.0420, r2=0.00410, r3=0.162,
    r4=0.0049, KsS=0.0, kiS=0.2079, nS=3.6205, k1=0.1771,
    phiS=0.6675, KsL=0.0227, kiL=0.0861, nL=1.8117, k2=0.2135,
    phiL=0.0, KH=4.653,
)

PARAM_NAMES = [
    "rhoN_max","mu_max","r3","r1","Kstar",
    "phiN","KsA_N","kiA_N","YXA","r2",
]
GUESSES = {
    "rhoN_max":30.0, "mu_max":0.08,  "r3":0.12,  "r1":0.055,
    "Kstar":0.40,    "phiN":100.0,   "KsA_N":0.75,"kiA_N":0.82,
    "YXA":0.044,     "r2":0.003,
}

CONDITIONS = {
    "TAP":   {"A0":0.41913,"N0":0.38187},
    "N-":    {"A0":0.41913,"N0":0.35497},
    "A++":   {"A0":1.26341,"N0":0.38187},
    "A'-N'": {"A0":1.16514,"N0":0.31520},
}
A0_MAX=max(c["A0"] for c in CONDITIONS.values())
N0_MAX=max(c["N0"] for c in CONDITIONS.values())

COND_STYLE = {
    "TAP":   dict(color="#2C2C2C",ls="-",  marker="o",label="TAP"),
    "N-":    dict(color="#1A6FAF",ls="--", marker="s",label=r"N$^{-}$"),
    "A++":   dict(color="#C0392B",ls="-.", marker="^",label=r"A$^{++}$"),
    "A'-N'": dict(color="#27855A",ls=":",  marker="D",label=r"A$'$–N$'$"),
}

STATE_LABELS = ["Biomass","Nitrogen","N quota","Acetate","Starch","Lipids","pH"]
DIGIT_MAP = {
    "Biomass":(0,0),"Nitrogen":(1,1),"N quota":(2,2),
    "Acetate":(3,3),"Starch":(4,4),"Lipids":(5,5),"pH":(6,6),
}
PRIMARY = {"Nitrogen","Biomass","Lipids","pH","Starch"}
IDX7    = [0,1,2,3,4,5,7]
I0=125.0; z=0.05; X_SEED=5e-4; QN_IC=0.99185; H_IC=6.99818; T_END=200.0
DIGITISED_CSV = "fig3_digitized_tidy.csv"

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
    R2=(params["r2"]/qNw)*Xw; R4=(params["r4"]/qNw)*Xw
    dX=mu*Xw; dN=-rho*Xw; dqN=rho-mu*qNw
    dA=-(1/params["YXA"])*muH/(muH+muI+1e-14)*dX
    dxact=dX+R2+R4-R1-R3; dS=R1-R2; dL=R3-R4; dH=params["KH"]*dxact
    return [dX,dN,dqN,dA,dS,dL,dxact,dH]

# LSODA baseline
print("Running LSODA baselines...")
t_eval=np.linspace(0,T_END,2001)
sols={}
for cname,cfg in CONDITIONS.items():
    y0=np.array([X_SEED,cfg["N0"],QN_IC,cfg["A0"],0.0,0.0,X_SEED,H_IC])
    sol=solve_ivp(lambda t,y,A0=cfg["A0"],N0=cfg["N0"]:ode_rhs(t,y,A0,N0,p),
                  t_span=(0.0,T_END),y0=y0,method="LSODA",
                  rtol=1e-8,atol=1e-10,dense_output=True,t_eval=t_eval)
    assert sol.status==0
    sols[cname]=sol
    print(f"  {cname}: X_fin={sol.y[0,-1]:.4f}")
scale7=np.array([max(sols["TAP"].y[i].max(),1e-10) for i in IDX7],dtype=np.float32)

# Data
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
                  .replace("\u2013","-").replace("\u2014","-").replace("\u2032","'"))
df_all["condition"]=df_all["condition"].map(cond_map).fillna(df_all["condition"])

# Torch helpers
p_t={k:torch.tensor(v,dtype=torch.float32) for k,v in p.items()}
sc=torch.tensor(scale7,dtype=torch.float32)
T_t=torch.tensor(T_END,dtype=torch.float32)
I0t=torch.tensor(I0,dtype=torch.float32)
zt=torch.tensor(z,dtype=torch.float32)
A0_MAX_t=torch.tensor(A0_MAX,dtype=torch.float32)
N0_MAX_t=torch.tensor(N0_MAX,dtype=torch.float32)

def andrews_t(u,Ks,ki):
    u=torch.clamp(u,min=0.0); return u/(u+Ks+u*u/ki+1e-12)
def g_int_N_t(Ni,Ks,ki,n_exp):
    Ni=torch.clamp(Ni,min=0.0)
    Nin=Ni**n_exp; Ksn=Ks**n_exp; inner=(Ni*Ni/(ki+1e-12))**n_exp
    return Nin/(Nin+Ksn+inner+1e-20)

def make_ode_residuals(log_vars):
    def get(name):
        return torch.exp(log_vars[name]) if name in log_vars else p_t[name]
    def ode_residuals(x_in,u_hat):
        A0=x_in[:,1:2]*A0_MAX_t; N0=x_in[:,2:3]*N0_MAX_t
        X=u_hat[:,0:1]*sc[0]; N=u_hat[:,1:2]*sc[1]
        qN=u_hat[:,2:3]*sc[2]; A=u_hat[:,3:4]*sc[3]
        S=u_hat[:,4:5]*sc[4]; L=u_hat[:,5:6]*sc[5]; H=u_hat[:,6:7]*sc[6]
        xact=torch.clamp(X-S-L,min=0.0)
        Xw=torch.clamp(X,min=0.0); Nw=torch.clamp(N,min=0.0)
        Aw=torch.clamp(A,min=0.0); xw=torch.clamp(xact,min=0.0)
        qNw=torch.clamp(qN,min=p_t["qN0"]+1e-12)
        I_avg=I0t*torch.exp(-p_t["sigma_light"]*Xw*zt)
        wH_n=Aw/(p_t["KsA"]+1e-12); wI_n=I_avg/(p_t["KsI"]+1e-12)
        dw=wH_n+wI_n+1e-14; wH=wH_n/dw
        muH=andrews_t(Aw,p_t["KsA"],p_t["kiA"])
        muI=andrews_t(I_avg,p_t["KsI"],p_t["kiI"])
        mu_bar=get("mu_max")*(wH*muH+(1-wH)*muI)
        mu=torch.clamp(mu_bar*(1.0-p_t["qN0"]/qNw),min=0.0)
        N0n=N0**p_t["n"]; Ksn=get("Kstar")**p_t["n"]
        rho_bar=(get("rhoN_max")*N0n/(N0n+Ksn+1e-12)*torch.exp(-get("phiN")*Xw))
        rho=(rho_bar*andrews_t(Nw,p_t["KsN"],p_t["kiN"])
             *andrews_t(Aw,get("KsA_N"),get("kiA_N")))
        Ni=qNw*Xw; Aint=torch.clamp(A0-Aw,min=0.0)
        gS=g_int_N_t(Ni,p_t["KsS"],p_t["kiS"],p_t["nS"])
        gL=g_int_N_t(Ni,p_t["KsL"],p_t["kiL"],p_t["nL"])
        Nn=Nw/(N0+1e-12)
        gate_S=p_t["k1"]/(p_t["k1"]+Nn+1e-12)
        gate_L=p_t["k2"]/(p_t["k2"]+Nn+1e-12)
        drive_S=mu+torch.exp(p_t["phiS"]*Aint)
        drive_L=mu+torch.exp(p_t["phiL"]*Aint)
        R1=get("r1")*gS*gate_S*drive_S*xw
        R3=get("r3")*gL*gate_L*drive_L*xw
        R2=(get("r2")/qNw)*Xw; R4=(p_t["r4"]/qNw)*Xw
        dX=mu*Xw; dN=-rho*Xw; dqN=rho-mu*qNw
        dA=-(1.0/get("YXA"))*muH/(muH+muI+1e-14)*dX
        dxact=dX+R2+R4-R1-R3; dS=R1-R2; dL=R3-R4
        dH=p_t["KH"]*dxact
        f_hat=torch.cat([dX/sc[0],dN/sc[1],dqN/sc[2],dA/sc[3],
                         dS/sc[4],dL/sc[5],dH/sc[6]],dim=1)
        return [dde.grad.jacobian(u_hat,x_in,i=i,j=0)-T_t*f_hat[:,i:i+1]
                for i in range(7)]
    return ode_residuals

def zero_residuals(x_in,u_hat):
    return [u_hat[:,i:i+1]*0.0 for i in range(7)]

def r2_score(y_true,y_pred):
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-np.mean(y_true))**2)
    return 1.0-ss_res/ss_tot if ss_tot>1e-20 else float("nan")

# Geometry + BCs
geom=dde.geometry.Hypercube([0.0,0.0,0.0],[1.0,1.0,1.0])
all_ics=[]; all_anchor_bcs=[]; all_data_bcs=[]; anchor_arrays={}
for cname,cfg in CONDITIONS.items():
    A0_n=cfg["A0"]/A0_MAX; N0_n=cfg["N0"]/N0_MAX
    y0_7=np.array([X_SEED,cfg["N0"],QN_IC,cfg["A0"],0.0,0.0,H_IC],dtype=np.float32)
    ic_norm=(y0_7/scale7).astype(np.float32)
    x_ic=np.array([[0.0,A0_n,N0_n]],dtype=np.float32)
    for i in range(7):
        all_ics.append(dde.PointSetBC(x_ic,np.array([[ic_norm[i]]],dtype=np.float32),component=i))
    t_anch=np.linspace(0,T_END,N_ANCHORS); tau_anch=(t_anch/T_END).astype(np.float32)
    x_anch=np.stack([tau_anch,np.full(N_ANCHORS,A0_n,dtype=np.float32),
                     np.full(N_ANCHORS,N0_n,dtype=np.float32)],axis=1)
    y_anch=(sols[cname].sol(t_anch)[IDX7,:]/scale7[:,None]).T.astype(np.float32)
    anchor_arrays[cname]=(x_anch,y_anch)
    for i in range(7):
        all_anchor_bcs.append(dde.PointSetBC(x_anch,y_anch[:,i:i+1],component=i))
    for varname,(comp,sc_idx) in DIGIT_MAP.items():
        pts=df_all[(df_all["condition"]==cname)&
                   (df_all["variable"]==varname)].sort_values("time_h")
        if pts.empty: continue
        tau_pts=(pts["time_h"].values/T_END).astype(np.float32)
        x_pts=np.stack([tau_pts,np.full(len(tau_pts),A0_n,dtype=np.float32),
                        np.full(len(tau_pts),N0_n,dtype=np.float32)],axis=1)
        val_norm=(pts["value"].values.astype(np.float32)/scale7[sc_idx])[:,None]
        all_data_bcs.append((x_pts,val_norm,comp,80.0 if varname in PRIMARY else 30.0))

all_bcs_for_pde=all_ics+all_anchor_bcs
for x_pts,val_norm,comp,w in all_data_bcs:
    all_bcs_for_pde.append(dde.PointSetBC(x_pts,val_norm,component=comp))
x_all_anch=np.vstack([anchor_arrays[c][0] for c in CONDITIONS])
n_ic=len(all_ics); n_anchor=len(all_anchor_bcs)
lw_phase2=[1.0]*7+[5.0]*n_ic+[2.0]*n_anchor+[d[3] for d in all_data_bcs]
lw_ws=[0.0]*7+[0.0]*n_ic+[1.0]*n_anchor

# Network
net=dde.maps.FNN([3,64,64,64,7],"tanh","Glorot normal")
net.apply_output_transform(lambda x,y:torch.nn.functional.softplus(y))

# Phase 1: Warm start (no physics) 
print("\nPhase 1: Warm start (anchors + ICs only)...")
t0=time.time()
data_ws=dde.data.PDE(geom,zero_residuals,all_ics+all_anchor_bcs,
                     num_domain=512,num_boundary=4,anchors=x_all_anch)
model_ws=dde.Model(data_ws,net)
model_ws.compile("adam",lr=1e-3,loss_weights=lw_ws)
model_ws.train(iterations=8000,display_every=4000)
model_ws.compile("L-BFGS",loss_weights=lw_ws)
model_ws.train()
warmup_state=copy.deepcopy(net.state_dict())
print(f"Warm start done in {time.time()-t0:.0f}s")

# Phase 2: Inverse problem
log_vars={name:dde.Variable(float(np.log(GUESSES[name]))) for name in PARAM_NAMES}

class ParamTracker(dde.callbacks.Callback):
    def __init__(self,lv,every=500):
        super().__init__(); self.lv=lv; self.every=every; self.history=[]
    def on_epoch_end(self):
        step=self.model.train_state.step
        if step%self.every==0:
            vals={k:float(torch.exp(v).detach().cpu()) for k,v in self.lv.items()}
            self.history.append((step,vals))
            parts=[f"{k}={v:.4f}" for k,v in list(vals.items())[:4]]
            print(f"  step {step:>5d}  |  "+"  |  ".join(parts)+"  ...")

tracker=ParamTracker(log_vars,every=500)
net.load_state_dict(warmup_state)
ext_vars=list(log_vars.values())
ode_res=make_ode_residuals(log_vars)
data_inv=dde.data.PDE(geom,ode_res,all_bcs_for_pde,
                      num_domain=N_COL,num_boundary=4,anchors=x_all_anch)
model_inv=dde.Model(data_inv,net)

print("\nPhase 2a: Adam...")
t1=time.time()
model_inv.compile("adam",lr=5e-4,loss_weights=lw_phase2,
                  external_trainable_variables=ext_vars)
model_inv.train(iterations=N_ADAM,display_every=N_ADAM//4,callbacks=[tracker])
print(f"Adam done in {time.time()-t1:.0f}s")

print("Phase 2b: L-BFGS...")
t2=time.time()
model_inv.compile("L-BFGS",loss_weights=lw_phase2,
                  external_trainable_variables=ext_vars)
model_inv.train()
print(f"L-BFGS done in {time.time()-t2:.0f}s")

inferred={k:float(torch.exp(v).detach().cpu()) for k,v in log_vars.items()}

# Predictions + R²
t_dense=np.linspace(0,T_END,501)
pinn_preds={}; r2s_all={}
for cname in CONDITIONS:
    cfg=CONDITIONS[cname]; A0_n=cfg["A0"]/A0_MAX; N0_n=cfg["N0"]/N0_MAX
    tau_d=(t_dense/T_END).astype(np.float32)
    x_d=np.stack([tau_d,np.full(501,A0_n,dtype=np.float32),
                  np.full(501,N0_n,dtype=np.float32)],axis=1)
    y_d=model_inv.predict(x_d)
    pinn_preds[cname]={"t":t_dense,
                       "y":np.column_stack([y_d[:,i]*scale7[i] for i in range(7)])}
    r2s_all[cname]={}
    for varname,(comp,sc_idx) in DIGIT_MAP.items():
        pts=df_all[(df_all["condition"]==cname)&
                   (df_all["variable"]==varname)].sort_values("time_h")
        if pts.empty: continue
        tau_p=(pts["time_h"].values/T_END).astype(np.float32)
        x_p=np.stack([tau_p,np.full(len(tau_p),A0_n,dtype=np.float32),
                      np.full(len(tau_p),N0_n,dtype=np.float32)],axis=1)
        y_pred=model_inv.predict(x_p)[:,comp]*scale7[sc_idx]
        r2s_all[cname][varname]=r2_score(pts["value"].values,y_pred)

all_r2=[v for c in r2s_all.values() for v in c.values() if not np.isnan(v)]
mean_r2=np.mean(all_r2); n_ok=0
print("\n"+"="*64)
print(f"  {'Parameter':<12} {'Inferred':>10} {'Literature':>12} {'Dev%':>8}  Status")
print(f"  {'':{'─'}<56}")
for name in PARAM_NAMES:
    inf=inferred[name]; lit=p[name]; delta=100*(inf-lit)/lit
    st="OK" if abs(delta)<20 else "WARN" if abs(delta)<40 else "FAIL"
    if abs(delta)<20: n_ok+=1
    print(f"  {name:<12} {inf:>10.5f} {lit:>12.5f} {delta:>+7.1f}%  {st}")
print(f"\n  Mean R²={mean_r2:.4f}   {n_ok}/10 within 20%")
for cname in CONDITIONS:
    print(f"    {cname}: R²={np.nanmean(list(r2s_all[cname].values())):.4f}")
print("="*64)

pd.DataFrame([{
    "parameter":name,"guess":GUESSES[name],"inferred":inferred[name],
    "literature":p[name],"deviation_pct":100*(inferred[name]-p[name])/p[name]}
    for name in PARAM_NAMES]).to_csv(
    os.path.join(RESULTS_DIR,"deterministic_results.csv"),index=False)

# Figure 1: Trajectories
PANEL_DEFS=[
    ("Biomass",  r"(a)  Biomass",  r"$X$ (g$_\mathrm{C}$ L$^{-1}$)",               0),
    ("Nitrogen", r"(b)  Nitrogen", r"$N$ (g$_\mathrm{N}$ L$^{-1}$)",               1),
    ("N quota",  r"(c)  N quota",  r"$q_N$ (g$_\mathrm{N}$ g$_\mathrm{C}^{-1}$)", 2),
    ("Acetate",  r"(d)  Acetate",  r"$A$ (g$_\mathrm{C}$ L$^{-1}$)",               3),
    ("Starch",   r"(e)  Starch",   r"$S$ (g$_\mathrm{C}$ L$^{-1}$)",               4),
    ("Lipids",   r"(f)  Lipids",   r"$L$ (g$_\mathrm{C}$ L$^{-1}$)",               5),
    ("Starch",   r"(g)  Starch",   r"$S$ (g$_\mathrm{C}$ L$^{-1}$)",               4),
    ("pH",       r"(h)  pH",       r"pH",                                            6),
]
NO_CLIP={"pH","N quota"}; MSTYLE=dict(s=14,zorder=6,linewidths=0.7)
fig1,axes1=plt.subplots(2,4,figsize=(6.8,6.2))
fig1.subplots_adjust(left=0.11,right=0.98,bottom=0.14,top=0.96,wspace=0.58,hspace=0.52)
for ax_idx,(varname,title,ylabel,comp) in enumerate(PANEL_DEFS):
    ax=axes1.flat[ax_idx]
    for cname in CONDITIONS:
        sty=COND_STYLE[cname]
        ax.plot(t_eval,sols[cname].y[IDX7[comp],:],color="0.78",lw=0.8,ls="--",zorder=1)
        ax.plot(pinn_preds[cname]["t"],pinn_preds[cname]["y"][:,comp],
                color=sty["color"],lw=1.4,ls=sty["ls"],zorder=2)
        pts=df_all[(df_all["condition"]==cname)&(df_all["variable"]==varname)].sort_values("time_h")
        if not pts.empty:
            ax.scatter(pts["time_h"].values,pts["value"].values,
                       facecolors="white",edgecolors=sty["color"],marker=sty["marker"],**MSTYLE)
    r2_vals=[r2s_all[c].get(varname,np.nan) for c in CONDITIONS]
    r2_vals=[v for v in r2_vals if not np.isnan(v)]
    if r2_vals:
        ax.text(0.97,0.96,rf"$\bar{{R}}^2={np.mean(r2_vals):.3f}$",
                transform=ax.transAxes,ha="right",va="top",fontsize=6.5,color="#333333",
                bbox=dict(boxstyle="round,pad=0.22",fc="white",ec="0.80",lw=0.5,alpha=0.88))
    ax.set_title(title,pad=3,loc="left"); ax.set_ylabel(ylabel,labelpad=2)
    ax.set_xlabel("Time (h)",labelpad=2); ax.set_xlim(0,T_END)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=3))
    ax.grid(True,which="major")
    if varname not in NO_CLIP: ax.set_ylim(bottom=0)
legend_handles=[
    Line2D([0],[0],color=COND_STYLE[c]["color"],lw=1.4,ls=COND_STYLE[c]["ls"],
           label=COND_STYLE[c]["label"]) for c in CONDITIONS
]+[
    Line2D([0],[0],marker="o",color="0.45",lw=0,markerfacecolor="white",
           markeredgecolor="0.45",markersize=4,label="Experimental data"),
    Line2D([0],[0],color="0.78",lw=0.8,ls="--",label="LSODA (literature)"),
]
fig1.legend(legend_handles,[h.get_label() for h in legend_handles],
            loc="lower center",ncol=6,bbox_to_anchor=(0.525,0.01),
            frameon=True,fontsize=7,handlelength=2.2,columnspacing=1.0,borderpad=0.4)
fig1.savefig(os.path.join(RESULTS_DIR,"fig1_trajectories_pub.png"),dpi=300)
plt.close(fig1); print("Saved: fig1_trajectories_pub.png")

# Figure 2: Deviation bar 
deltas=[100*(inferred[n]-p[n])/p[n] for n in PARAM_NAMES]
bar_colors=["#27855A" if abs(d)<10 else "#f39c12" if abs(d)<20 else "#c0392b" for d in deltas]
fig2,ax2=plt.subplots(figsize=(7.0,3.2))
fig2.subplots_adjust(left=0.09,right=0.97,bottom=0.18,top=0.88)
x=np.arange(len(PARAM_NAMES))
bars=ax2.bar(x,deltas,color=bar_colors,width=0.55,edgecolor="white",linewidth=0.6)
ax2.axhline(0,color="#333333",linewidth=0.8)
ax2.axhline( 20,color="#f39c12",linewidth=0.7,linestyle="--",alpha=0.6)
ax2.axhline(-20,color="#f39c12",linewidth=0.7,linestyle="--",alpha=0.6)
for bar,d in zip(bars,deltas):
    ax2.text(bar.get_x()+bar.get_width()/2,d+(1.2 if d>=0 else -3.5),
             f"{d:+.1f}%",ha="center",va="bottom",fontsize=7,color="#1a1a1a",fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([rf"$\mathit{{{n}}}$" for n in PARAM_NAMES],fontsize=8)
ax2.set_ylabel("Deviation from literature (%)")
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f"{v:+.0f}%"))
ax2.set_title(rf"Parameter recovery  |  Mean $R^2={mean_r2:.4f}$  |  {n_ok}/10 within 20\%",
              fontsize=8.5,pad=6,loc="left")
ax2.legend(handles=[Patch(facecolor=c,edgecolor="white",label=l)
    for c,l in [("#27855A",r"$<10\%$"),("#f39c12",r"$10$–$20\%$"),("#c0392b",r"$>20\%$")]],
    title="Absolute deviation",fontsize=7,title_fontsize=7,loc="upper right")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.grid(True,axis="y",lw=0.4,alpha=0.35)
fig2.savefig(os.path.join(RESULTS_DIR,"fig2_deviation_bar.png"),dpi=300)
plt.close(fig2); print("Saved: fig2_deviation_bar.png")

# Figure 3: Convergence 
PARAM_COLORS=[
    "#2C2C2C","#C0392B","#27855A","#1A6FAF","#8e44ad",
    "#1abc9c","#795548","#607d8b","#e91e63","#ff9800",
]
if tracker.history:
    steps=[h[0] for h in tracker.history]
    fig3,ax3=plt.subplots(figsize=(7.0,3.2))
    fig3.subplots_adjust(left=0.09,right=0.97,bottom=0.14,top=0.88)
    for ci,name in enumerate(PARAM_NAMES):
        vals=[h[1][name]/p[name] for h in tracker.history]
        ax3.plot(steps,vals,color=PARAM_COLORS[ci],lw=1.4,label=rf"$\mathit{{{name}}}$")
    ax3.axhline(1.0,color="#333333",lw=0.8,ls="--",zorder=0)
    ax3.axhspan(0.8,1.2,color="#f5f5f5",zorder=0)
    ax3.text(0.98,0.97,rf"$\bar{{R}}^2={mean_r2:.4f}$  |  {n_ok}/10 within 20\%",
             transform=ax3.transAxes,ha="right",va="top",fontsize=7,color="#444444",
             bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="#cccccc",lw=0.6))
    ax3.set_xlabel("Training step"); ax3.set_ylabel("Inferred / Literature")
    ax3.set_ylim(0.2,2.2); ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax3.legend(ncol=5,fontsize=7,loc="upper left",handlelength=1.8,columnspacing=0.9)
    ax3.set_title("Parameter convergence during training",fontsize=8.5,pad=6,loc="left")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    fig3.savefig(os.path.join(RESULTS_DIR,"fig3_convergence.png"),dpi=300)
    plt.close(fig3); print("Saved: fig3_convergence.png")

# Figure 4: R² heatmap
r2_matrix=np.full((len(CONDITIONS),7),np.nan)
cond_list=list(CONDITIONS.keys())
for ci,cname in enumerate(cond_list):
    for si,slabel in enumerate(STATE_LABELS):
        r2_matrix[ci,si]=r2s_all[cname].get(slabel,np.nan)
fig4,ax4=plt.subplots(figsize=(6.5,2.8))
fig4.subplots_adjust(left=0.10,right=0.97,bottom=0.18,top=0.88)
im=ax4.imshow(r2_matrix,aspect="auto",cmap="YlGnBu",vmin=0.90,vmax=1.0)
ax4.set_xticks(range(7)); ax4.set_xticklabels(STATE_LABELS,fontsize=8)
ax4.set_yticks(range(len(cond_list))); ax4.set_yticklabels(cond_list,fontsize=8)
for i in range(len(cond_list)):
    for j in range(7):
        v=r2_matrix[i,j]
        if not np.isnan(v):
            ax4.text(j,i,f"{v:.3f}",ha="center",va="center",
                     fontsize=7,color="white" if v<0.95 else "#1a1a1a")
cb=plt.colorbar(im,ax=ax4,pad=0.02,fraction=0.03,aspect=20)
cb.set_label(r"$R^2$",fontsize=8); cb.ax.tick_params(labelsize=7)
ax4.set_title(r"$R^2$ per state per condition",fontsize=8.5,pad=6,loc="left")
ax4.tick_params(axis="both",length=0)
fig4.savefig(os.path.join(RESULTS_DIR,"fig4_r2_heatmap.png"),dpi=300)
plt.close(fig4); print("Saved: fig4_r2_heatmap.png")

mae=np.mean([abs(100*(inferred[n]-p[n])/p[n]) for n in PARAM_NAMES])
print(f"\nAll outputs in: {RESULTS_DIR}/")
print(f"Mean R²: {mean_r2:.4f}  |  {n_ok}/10 within 20%  |  MAE: {mae:.2f}%")