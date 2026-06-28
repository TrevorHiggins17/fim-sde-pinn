import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import copy
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p = dict(
    mu_max=0.106,  kiA=0.109,    KsI=1.4,
    KsA=1.789,     kiI=186.52,   sigma_light=1.0,
    qN0=0.876,     YXA=0.059,
    rhoN_max=40.445, Kstar=0.3125, n=18.183,
    phiN=137.455,  KsN=0.162,    kiN=0.113,
    KsA_N=1.004,   kiA_N=1.098,
    r1=0.0420,     r2=0.00410,
    r3=0.162,      r4=0.0049,
    KsS=0.0,       kiS=0.2079,   nS=3.6205,
    k1=0.1771,     phiS=0.6675,
    KsL=0.0227,    kiL=0.0861,   nL=1.8117,
    k2=0.2135,     phiL=0.0,
    KH=4.653,
)

# Level 5 inferred values (from deterministic run) — used as warm start
LEVEL5_INFERRED = {
    "rhoN_max": 44.84,
    "mu_max":   0.1025,
    "r3":       0.1507,
    "KH":       4.039,
    "r1":       0.0394,
}

# Experiment constants
I0 = 125.0; z = 0.05; X_SEED = 5.0e-4; QN_IC = 0.99185; H_IC = 6.99818
T_END = 200.0; A0_TAP = 0.41913; N0_TAP = 0.38187
DIGITISED_CSV = "fig3_digitized_tidy.csv"

# KKL parameters
N_KKL = 5          # number of KKL expansion terms
N_NOISE_SAMPLES = 4  # reduced for CPU — 4 noise samples per step # noise realisations per training batch
N_ENSEMBLE = 200    # realisations for final uncertainty bands

def _pos(x): return x if x > 0.0 else 0.0

def andrews_np(u, Ks, ki):
    u = _pos(u)
    return u / (u + Ks + u*u/ki + 1e-20)

def g_int_N_np(Ni, Ks, ki, n_exp):
    Ni = _pos(Ni)
    if Ni < 1e-20: return 0.0
    Nin = Ni**n_exp
    Ksn = Ks**n_exp if Ks > 1e-20 else 0.0
    inner = (Ni*Ni/ki)**n_exp
    denom = Nin + Ksn + inner
    return Nin/denom if denom > 1e-20 else 0.0

def ode_rhs(t, y, A0, N0, params):
    X,N,qN,A,S,L,xact,H = y
    Xw=_pos(X); Nw=_pos(N); Aw=_pos(A); xactw=_pos(xact)
    qNw = max(qN, params["qN0"]+1e-12)
    I = I0*np.exp(-params["sigma_light"]*Xw*z)
    wH_n=Aw/params["KsA"]; wI_n=I/params["KsI"]; dw=wH_n+wI_n
    wH = 0.5 if dw < 1e-14 else wH_n/dw
    muH = andrews_np(Aw, params["KsA"], params["kiA"])
    muI = andrews_np(I,  params["KsI"], params["kiI"])
    mu_bar = params["mu_max"]*(wH*muH + (1-wH)*muI)
    mu = max(mu_bar*(1.0 - params["qN0"]/qNw), 0.0)
    rho_bar = (params["rhoN_max"]
               * N0**params["n"] / (N0**params["n"]+params["Kstar"]**params["n"])
               * np.exp(-params["phiN"]*Xw))
    rho = (rho_bar * andrews_np(Nw, params["KsN"], params["kiN"])
           * andrews_np(Aw, params["KsA_N"], params["kiA_N"]))
    Ni=qNw*Xw; Aint=max(A0-Aw, 0.0)
    gS=g_int_N_np(Ni, params["KsS"], params["kiS"], params["nS"])
    gL=g_int_N_np(Ni, params["KsL"], params["kiL"], params["nL"])
    Nn=Nw/N0
    gate_S=params["k1"]/(params["k1"]+Nn+1e-12)
    gate_L=params["k2"]/(params["k2"]+Nn+1e-12)
    drive_S=mu+np.exp(params["phiS"]*Aint)
    drive_L=mu+np.exp(params["phiL"]*Aint)
    R1=params["r1"]*gS*gate_S*drive_S*xactw
    R3=params["r3"]*gL*gate_L*drive_L*xactw
    R2=(params["r2"]/qNw)*Xw; R4=(params["r4"]/qNw)*Xw
    dX=mu*Xw; dN=-rho*Xw; dqN=rho-mu*qNw
    fH=muH/(muH+muI+1e-14)
    dA=-(1.0/params["YXA"])*fH*dX
    dxact=dX+R2+R4-R1-R3; dS=R1-R2; dL=R3-R4; dH=params["KH"]*dxact
    return [dX,dN,dqN,dA,dS,dL,dxact,dH]

print("Running LSODA baseline...")
p_run = {**p, **LEVEL5_INFERRED}
y0_8 = np.array([X_SEED, N0_TAP, QN_IC, A0_TAP, 0.0, 0.0, X_SEED, H_IC])
sol_lit = solve_ivp(
    lambda t,y: ode_rhs(t,y,A0_TAP,N0_TAP,p),
    t_span=(0.0,T_END), y0=y0_8, method="LSODA",
    rtol=1e-8, atol=1e-10, dense_output=True,
    t_eval=np.linspace(0,T_END,2001))
sol_L5 = solve_ivp(
    lambda t,y: ode_rhs(t,y,A0_TAP,N0_TAP,p_run),
    t_span=(0.0,T_END), y0=y0_8, method="LSODA",
    rtol=1e-8, atol=1e-10, dense_output=True,
    t_eval=np.linspace(0,T_END,2001))
assert sol_lit.status == 0 and sol_L5.status == 0
print("LSODA OK")

IDX7 = [0,1,2,3,4,5,7]  # X,N,qN,A,S,L,H (skip xact)
scale7 = np.array([max(sol_lit.y[i].max(), 1e-10) for i in IDX7], dtype=np.float32)
y0_7   = y0_8[IDX7].astype(np.float32)
ic_norm = (y0_7/scale7).astype(np.float32)

def load_digit(csv_path, condition="TAP"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for c in ["condition","variable"]:
        df[c] = df[c].astype(str).str.strip()
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df["value"]  = pd.to_numeric(df["value"],  errors="coerce")
    return df[df["condition"]==condition].dropna(subset=["time_h","value"])

digit = load_digit(DIGITISED_CSV)
DIGIT_MAP = {
    "Biomass":(0,0), "Nitrogen":(1,1), "N quota":(2,2),
    "Acetate":(3,3), "Starch":(4,4),   "Lipids":(5,5), "pH":(6,6)
}
PRIMARY = {"Nitrogen","Biomass","Lipids","pH","Starch"}

p_t  = {k: torch.tensor(v, dtype=torch.float32, device=device) for k,v in p.items()}
sc   = torch.tensor(scale7, dtype=torch.float32, device=device)
T    = torch.tensor(T_END,  dtype=torch.float32, device=device)
I0t  = torch.tensor(I0,     dtype=torch.float32, device=device)
zt   = torch.tensor(z,      dtype=torch.float32, device=device)
A0t  = torch.tensor(A0_TAP, dtype=torch.float32, device=device)
N0t  = torch.tensor(N0_TAP, dtype=torch.float32, device=device)
EPS  = torch.tensor(1e-6,   dtype=torch.float32, device=device)

def kkl_wiener_deriv(tau: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
    """
    Compute dW/dtau using KKL expansion.
    
    tau  : (batch,) normalised time in [0,1]
    zeta : (batch, N_KKL) standard normal samples
    
    Returns dW/dtau : (batch,) 
    
    dW/dtau = sqrt(2T) * sum_k zeta_k * cos((k-0.5)*pi*tau)
    """
    k_vals = torch.arange(1, N_KKL+1, dtype=torch.float32, device=device)
    # (batch, N_KKL)
    phases = (k_vals - 0.5) * torch.pi * tau.unsqueeze(-1)
    cos_terms = torch.cos(phases)
    # sqrt(2T) factor — scales variance to physical time T
    scale_factor = torch.sqrt(2.0 * T)
    # (batch,)
    dW = scale_factor * (zeta * cos_terms).sum(dim=-1)
    return dW

def andrews_t(u, Ks, ki):
    u = torch.clamp(u, min=0.0)
    return u / (u + Ks + u*u/ki + 1e-12)

def g_int_N_t(Ni, Ks, ki, n_exp):
    Ni = torch.clamp(Ni, min=0.0)
    Nin = Ni**n_exp; Ksn = Ks**n_exp
    inner = (Ni*Ni/(ki+1e-12))**n_exp
    return Nin/(Nin+Ksn+inner+1e-20)

def compute_drift(y_norm: torch.Tensor, theta: dict) -> torch.Tensor:
    """
    Compute f_i(y, theta) for all 7 states.
    y_norm : (batch, 7) — normalised state
    Returns f_norm : (batch, 7) — normalised drift (divided by scale)
    """
    def get(name):
        return torch.exp(theta[name]) if name in theta else p_t[name]

    X  = y_norm[:,0:1]*sc[0]; N  = y_norm[:,1:2]*sc[1]
    qN = y_norm[:,2:3]*sc[2]; A  = y_norm[:,3:4]*sc[3]
    S  = y_norm[:,4:5]*sc[4]; L  = y_norm[:,5:6]*sc[5]
    H  = y_norm[:,6:7]*sc[6]
    xact = torch.clamp(X - S - L, min=0.0)

    Xw = torch.clamp(X,    min=0.0)
    Nw = torch.clamp(N,    min=0.0)
    Aw = torch.clamp(A,    min=0.0)
    xw = torch.clamp(xact, min=0.0)
    qNw = torch.clamp(qN,  min=p_t["qN0"]+1e-12)

    I_avg = I0t * torch.exp(-p_t["sigma_light"]*Xw*zt)
    wH_n  = Aw/(p_t["KsA"]+1e-12)
    wI_n  = I_avg/(p_t["KsI"]+1e-12)
    dw    = wH_n + wI_n + 1e-14
    wH    = wH_n / dw

    muH  = andrews_t(Aw,    p_t["KsA"], p_t["kiA"])
    muI  = andrews_t(I_avg, p_t["KsI"], p_t["kiI"])
    mu_bar = get("mu_max") * (wH*muH + (1-wH)*muI)
    mu     = torch.clamp(mu_bar*(1.0 - p_t["qN0"]/qNw), min=0.0)

    rho_bar = (get("rhoN_max")
               * N0t**p_t["n"] / (N0t**p_t["n"] + p_t["Kstar"]**p_t["n"] + 1e-12)
               * torch.exp(-p_t["phiN"]*Xw))
    rho = (rho_bar
           * andrews_t(Nw, p_t["KsN"],   p_t["kiN"])
           * andrews_t(Aw, p_t["KsA_N"], p_t["kiA_N"]))

    Ni   = qNw * Xw
    Aint = torch.clamp(A0t - Aw, min=0.0)
    gS   = g_int_N_t(Ni, p_t["KsS"], p_t["kiS"], p_t["nS"])
    gL   = g_int_N_t(Ni, p_t["KsL"], p_t["kiL"], p_t["nL"])
    Nn   = Nw / (N0t + 1e-12)
    gate_S = p_t["k1"] / (p_t["k1"] + Nn + 1e-12)
    gate_L = p_t["k2"] / (p_t["k2"] + Nn + 1e-12)
    drive_S = mu + torch.exp(p_t["phiS"] * Aint)
    drive_L = mu + torch.exp(p_t["phiL"] * Aint)

    R1 = get("r1")*gS*gate_S*drive_S*xw
    R3 = get("r3")*gL*gate_L*drive_L*xw
    R2 = (p_t["r2"]/qNw)*Xw
    R4 = (p_t["r4"]/qNw)*Xw

    dX     = mu * Xw
    dN     = -rho * Xw
    dqN    = rho - mu*qNw
    fH     = muH / (muH+muI+1e-14)
    dA     = -(1.0/p_t["YXA"])*fH*dX
    dxact  = dX + R2 + R4 - R1 - R3
    dS     = R1 - R2
    dL     = R3 - R4
    dH     = get("KH") * dxact

    # Return normalised drift: f_i / scale_i
    f_norm = torch.cat([
        dX/sc[0], dN/sc[1], dqN/sc[2], dA/sc[3],
        dS/sc[4], dL/sc[5], dH/sc[6]
    ], dim=-1)
    return f_norm

def compute_diffusion_C(y_norm: torch.Tensor, log_sigma: dict) -> torch.Tensor:
    """
    Option C: state-dependent diffusion functions h_i(y).
    Returns sigma_i * h_i(y) / scale_i for each state.
    y_norm : (batch, 7)
    """
    X  = y_norm[:,0:1]*sc[0]
    N  = y_norm[:,1:2]*sc[1]
    qN = y_norm[:,2:3]*sc[2]
    A  = y_norm[:,3:4]*sc[3]
    S  = y_norm[:,4:5]*sc[4]
    L  = y_norm[:,5:6]*sc[5]
    xact = torch.clamp(X - S - L, min=0.0)

    sig_X  = torch.exp(log_sigma["X"])
    sig_N  = torch.exp(log_sigma["N"])
    sig_qN = torch.exp(log_sigma["qN"])
    sig_A  = torch.exp(log_sigma["A"])
    sig_S  = torch.exp(log_sigma["S"])
    sig_L  = torch.exp(log_sigma["L"])
    sig_H  = torch.exp(log_sigma["H"])

    # Option C diffusion functions
    hX  = torch.sqrt(torch.clamp(X,    min=0.0) + EPS)          # CIR
    hN  = torch.sqrt(torch.clamp(N,    min=0.0) + EPS)          # CIR
    hqN = torch.clamp(qN - p_t["qN0"], min=0.0) + EPS           # quota excess
    hA  = torch.sqrt(torch.clamp(A,    min=0.0) + EPS)          # CIR
    hS  = torch.sqrt(torch.clamp(S,    min=0.0) + EPS)          # CIR
    hL  = torch.sqrt(torch.clamp(L,    min=0.0) + EPS)          # CIR
    hH  = torch.clamp(xact, min=0.0) + EPS                       # active biomass

    # Normalised diffusion: sigma_i * h_i / scale_i
    diff_norm = torch.cat([
        sig_X  * hX  / sc[0],
        sig_N  * hN  / sc[1],
        sig_qN * hqN / sc[2],
        sig_A  * hA  / sc[3],
        sig_S  * hS  / sc[4],
        sig_L  * hL  / sc[5],
        sig_H  * hH  / sc[6],
    ], dim=-1)
    return diff_norm

class SDEPINN(nn.Module):
    def __init__(self, n_kkl=N_KKL, hidden=64):
        super().__init__()
        in_dim = 1 + n_kkl  # tau + zeta_1...zeta_n
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 7)
        )
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)
        nn.init.xavier_normal_(self.net[6].weight)

    def forward(self, tau, zeta):
        """
        tau  : (batch, 1) — requires_grad=True for autograd
        zeta : (batch, N_KKL) — noise vector
        Returns y_norm : (batch, 7)
        """
        x = torch.cat([tau, zeta], dim=-1)
        out = self.net(x)
        return torch.nn.functional.softplus(out)

def loss_ic(net, zeta_batch):
    """
    IC loss: y(0, zeta) = ic_norm for ALL noise realisations.
    IC is deterministic — the batch starts from the same point regardless of noise.
    """
    B = zeta_batch.shape[0]
    tau0 = torch.zeros(B, 1, device=device, requires_grad=False)
    y_pred = net(tau0, zeta_batch)
    ic = torch.tensor(ic_norm, device=device).unsqueeze(0).expand(B,-1)
    return torch.mean((y_pred - ic)**2)

def loss_anchor(net, tau_anchor, y_anchor_norm):
    """
    At zeta=0, mean trajectory should match LSODA Level5 anchors.
    """
    B = tau_anchor.shape[0]
    zeta_zero = torch.zeros(B, N_KKL, device=device)
    y_pred = net(tau_anchor.unsqueeze(-1) if tau_anchor.dim()==1 else tau_anchor, 
                 zeta_zero)
    return torch.mean((y_pred - y_anchor_norm)**2)

def loss_data(net, data_list):
    """
    At zeta=0, fit mean trajectory to digitised experimental data.
    data_list: list of (tau_pts, val_norm, comp, weight)
    """
    total = torch.tensor(0.0, device=device)
    for (tau_pts, val_norm, comp, w) in data_list:
        B = tau_pts.shape[0]
        zeta_zero = torch.zeros(B, N_KKL, device=device)
        y_pred = net(tau_pts, zeta_zero)
        total = total + w * torch.mean((y_pred[:,comp:comp+1] - val_norm)**2)
    return total

def loss_sde_residual(net, tau_col, zeta_batch, theta_dict, log_sigma_dict):
    """
    Stochastic physics residual over M noise realisations.

    Residual_i = dy_i/dtau - T*f_i(y,theta) - sigma_i*h_i(y)*dW/dtau

    Uses forward-mode JVP (Jacobian-vector product) to compute dy/dtau
    for ALL 7 outputs in a SINGLE pass — ~7x faster than the 7-loop
    reverse-mode approach.

    Maths: since tau is a scalar input, JVP with tangent=1 gives
    d(y)/d(tau) = [dy_1/dtau, ..., dy_7/dtau] in one forward sweep.

    tau_col   : (N_col,) collocation points
    zeta_batch: (M, N_KKL) noise samples for this step
    """
    M = zeta_batch.shape[0]
    N_col = tau_col.shape[0]
    total_res = torch.tensor(0.0, device=device)

    for m in range(M):
        zeta_m = zeta_batch[m:m+1].expand(N_col, -1)  # (N_col, N_KKL)

        tau_1d = tau_col.clone().detach().requires_grad_(True)  # (N_col,)

        def net_fn(tau_flat):
            """Network forward: tau (N_col,) -> y_norm (N_col, 7)"""
            return net(tau_flat.unsqueeze(-1), zeta_m)  # (N_col, 7)

        # Forward-mode JVP: tangent=ones computes d(net_fn)/d(tau) * 1
        # = dy_i/dtau for all i simultaneously in ONE pass.
        # create_graph=True keeps the computation graph for backprop
        # through the residual loss with respect to theta and sigma.
        tangent = torch.ones_like(tau_1d)
        y_pred, dy_dtau = torch.autograd.functional.jvp(
            net_fn, (tau_1d,), (tangent,), create_graph=True
        )
        # y_pred  : (N_col, 7) — network output
        # dy_dtau : (N_col, 7) — all derivatives in one pass

        # Drift: T * f_i(y, theta) / scale_i (already normalised)
        drift = T * compute_drift(y_pred, theta_dict)  # (N_col, 7)

        # KKL Wiener derivative: dW/dtau — (N_col,)
        dW_dtau = kkl_wiener_deriv(tau_col, zeta_m)  # (N_col,)

        # Diffusion: sigma_i * h_i(y) / scale_i * dW/dtau
        diff_norm = compute_diffusion_C(y_pred, log_sigma_dict)  # (N_col, 7)
        diffusion = diff_norm * dW_dtau.unsqueeze(-1)             # (N_col, 7)

        # SDE residual
        residual = dy_dtau - drift - diffusion  # (N_col, 7)
        total_res = total_res + torch.mean(residual**2)

    return total_res / M

# Prior: log(sigma_i) ~ Normal(PRIOR_MU, PRIOR_STD^2)
# Mode in sigma-space: exp(PRIOR_MU) = 0.01
# Replaces L2-around-zero which caused collapse.
#
# Physical reasoning: biological CV ~5-20% of state magnitude.
# With CIR diffusion h_i ~ sqrt(state), target sigma ~ 0.01
# gives ~1-5% noise contribution. PRIOR_STD = 1.5 allows
# sigma to range over [0.001, 0.1] within 1 std of prior.

PRIOR_MU  = float(torch.log(torch.tensor(0.01)).item())  # log(0.01) ~ -4.61
PRIOR_STD = 1.5   # allows ~2 orders of magnitude variation
# KH-specific prior — prevents KH drifting when sigma_H takes over pH fitting
# Prior centred at Level 5 inferred value (4.039), std=0.3 allows ±30% max
KH_PRIOR_MU  = float(torch.log(torch.tensor(LEVEL5_INFERRED["KH"])).item())
KH_PRIOR_STD = 0.3   # tight — KH should not drift far from Level 5 estimate


# Full Variational Inference on sigma — set True for posterior on sigma
# Slower but gives uncertainty ON the diffusion coefficient itself.
USE_VI = False

# VI posterior std parameters (only used if USE_VI=True)
log_sigma_std = {
    name: nn.Parameter(torch.tensor(-2.0, device=device))
    for name in ["X","N","qN","A","S","L","H"]
}

def sample_sigma_vi(log_sigma_mu_dict, log_sigma_std_dict):
    """Reparameterisation trick: sample log(sigma) ~ N(mu, s^2)."""
    sampled = {}
    for name in log_sigma_mu_dict:
        mu  = log_sigma_mu_dict[name]
        s   = torch.exp(log_sigma_std_dict[name])
        eps = torch.randn(1, device=device)
        sampled[name] = mu + s * eps
    return sampled

def loss_sigma_prior(log_sigma_dict, log_sigma_std_dict=None):
    """
    Bayesian prior on sigma replacing L2-around-zero regulariser.

    MAP (USE_VI=False):
        L = sum_i (log_sigma_i - PRIOR_MU)^2 / (2 * PRIOR_STD^2)
        Mode at sigma = exp(PRIOR_MU) = 0.01 — anchors sigma away from zero.

    VI (USE_VI=True):
        KL(q(log_sigma) || p(log_sigma)) where
        q = N(mu_i, s_i^2),  p = N(PRIOR_MU, PRIOR_STD^2)
        Rewards posterior entropy — prevents sigma collapsing to delta function.
    """
    total = torch.tensor(0.0, device=device)
    prior_var = PRIOR_STD ** 2

    if not USE_VI:
        # MAP: L2 around PRIOR_MU in log space
        for v in log_sigma_dict.values():
            total = total + (v - PRIOR_MU)**2 / (2.0 * prior_var)
    else:
        # VI: analytic KL between two Gaussians
        for name in log_sigma_dict:
            mu_i = log_sigma_dict[name]
            s_i  = torch.exp(log_sigma_std_dict[name])
            s_i2 = s_i ** 2
            kl_i = 0.5 * (
                s_i2 / prior_var
                + (mu_i - PRIOR_MU)**2 / prior_var
                - 1.0
                + torch.log(torch.tensor(prior_var, device=device) / (s_i2 + 1e-12))
            )
            total = total + kl_i
    return total


def loss_kh_prior(log_theta_dict):
    """
    Log-normal prior on KH specifically.
    Prevents KH drifting when sigma_H absorbs pH fitting burden.
    L = (log(KH) - KH_PRIOR_MU)^2 / (2 * KH_PRIOR_STD^2)
    """
    log_kh = log_theta_dict["KH"]
    return (log_kh - KH_PRIOR_MU)**2 / (2.0 * KH_PRIOR_STD**2)

# LSODA anchors at Level 5 inferred values
N_ANCHORS = 250
t_anchor = np.linspace(0, T_END, N_ANCHORS)
tau_anchor_np = (t_anchor/T_END).astype(np.float32)
y_anchor_np = (sol_L5.sol(t_anchor)[IDX7,:]/scale7[:,None]).T.astype(np.float32)
tau_anchor = torch.tensor(tau_anchor_np, device=device)  # (250,)
y_anchor   = torch.tensor(y_anchor_np,   device=device)  # (250, 7)

# Digitised data points
data_list = []
for varname,(comp,sc_idx) in DIGIT_MAP.items():
    pts = digit[digit["variable"]==varname].sort_values("time_h")
    if pts.empty: continue
    tau_pts = torch.tensor(
        (pts["time_h"].values/T_END).astype(np.float32),
        device=device
    ).unsqueeze(-1)  # (n_pts, 1)
    val_norm = torch.tensor(
        (pts["value"].values/scale7[sc_idx]).astype(np.float32),
        device=device
    ).unsqueeze(-1)  # (n_pts, 1)
    w = 80.0 if varname in PRIMARY else 30.0
    data_list.append((tau_pts, val_norm, comp, w))

# Random collocation points (resampled each step)
N_COL = 300          # reduced for CPU — 300 sufficient for SDE residual

net = SDEPINN(n_kkl=N_KKL).to(device)

# Kinetic parameters — initialise at Level 5 inferred values
log_theta = {
    "rhoN_max": nn.Parameter(torch.log(torch.tensor(LEVEL5_INFERRED["rhoN_max"], device=device))),
    "mu_max":   nn.Parameter(torch.log(torch.tensor(LEVEL5_INFERRED["mu_max"],   device=device))),
    "r3":       nn.Parameter(torch.log(torch.tensor(LEVEL5_INFERRED["r3"],       device=device))),
    "KH":       nn.Parameter(torch.log(torch.tensor(LEVEL5_INFERRED["KH"],       device=device))),
    "r1":       nn.Parameter(torch.log(torch.tensor(LEVEL5_INFERRED["r1"],       device=device))),
}

# Diffusion coefficients — initialise small
SIGMA_INIT = 0.01  # start at prior mode exp(PRIOR_MU)
log_sigma = {
    name: nn.Parameter(torch.log(torch.tensor(SIGMA_INIT, device=device)))
    for name in ["X","N","qN","A","S","L","H"]
}

def sample_collocation(n):
    """Sample n collocation points uniformly in [0,1]."""
    return torch.rand(n, device=device)

def sample_noise(m):
    """Sample m noise vectors from N(0,I_n)."""
    return torch.randn(m, N_KKL, device=device)

def sample_noise_ic(m):
    """Sample m noise vectors for IC loss."""
    return torch.randn(m, N_KKL, device=device)

def get_param_values():
    vals = {k: float(torch.exp(v).item()) for k,v in log_theta.items()}
    sigs = {k: float(torch.exp(v).item()) for k,v in log_sigma.items()}
    return vals, sigs

def print_params(step):
    vals, sigs = get_param_values()
    print(f"  step {step:>5d} | "
          f"rhoN={vals['rhoN_max']:.3f}(lit={p['rhoN_max']:.3f}) "
          f"mu={vals['mu_max']:.4f}(lit={p['mu_max']:.4f}) "
          f"r3={vals['r3']:.4f}(lit={p['r3']:.4f}) "
          f"KH={vals['KH']:.3f}(lit={p['KH']:.3f}) "
          f"r1={vals['r1']:.5f}(lit={p['r1']:.5f})")
    print(f"           | "
          f"σX={sigs['X']:.2e} σN={sigs['N']:.2e} σqN={sigs['qN']:.2e} "
          f"σA={sigs['A']:.2e} σS={sigs['S']:.2e} σL={sigs['L']:.2e} "
          f"σH={sigs['H']:.2e}")

# History for tracking
history = {"step":[], "loss":[], "loss_ic":[], "loss_anch":[], 
           "loss_data":[], "loss_phys":[], "loss_reg":[],
           "theta_vals":[], "sigma_vals":[]}

print("\n" + "="*65)
print("PHASE 0 — Warm start (zeta=0, freeze theta/sigma, ICs+anchors)")
print("="*65)

# Only network weights update in Phase 0
opt_p0 = torch.optim.Adam(net.parameters(), lr=1e-3)

W_IC_P0   = 100.0
W_ANCH_P0 = 10.0
N_STEPS_P0 = 6000

for step in range(N_STEPS_P0):
    opt_p0.zero_grad()

    # IC loss — sample noise to enforce IC holds for any zeta
    zeta_ic = sample_noise_ic(16)
    l_ic = loss_ic(net, zeta_ic)

    # Anchor loss — at zeta=0
    l_anch = loss_anchor(net, tau_anchor, y_anchor)

    loss = W_IC_P0*l_ic + W_ANCH_P0*l_anch
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt_p0.step()

    if step % 1000 == 0:
        print(f"  step {step:>5d} | loss={loss.item():.3e} "
              f"ic={l_ic.item():.3e} anch={l_anch.item():.3e}")

warmup_state = copy.deepcopy(net.state_dict())
print("Phase 0 complete — warm start saved")

print("\n" + "="*65)
print("PHASE 1 — Learn sigma only (theta fixed at Level5, activate SDE)")
print("="*65)

net.load_state_dict(warmup_state)

# Freeze theta, train sigma + net weights
sigma_params = list(log_sigma.values())
opt_p1 = torch.optim.Adam(list(net.parameters()) + sigma_params, lr=5e-4)

W_IC_P1   = 50.0
W_ANCH_P1 = 5.0
W_DATA_P1 = 3.0
W_PHYS_P1 = 5.0
W_REG_P1  = 0.1   # weight on log-normal prior
W_KH_P1   = 1.0   # KH prior weight Phase 1
N_STEPS_P1 = 8000

sigma_history_p1 = []

for step in range(N_STEPS_P1):
    opt_p1.zero_grad()

    tau_col  = sample_collocation(N_COL)
    zeta_phys = sample_noise(N_NOISE_SAMPLES)
    zeta_ic   = sample_noise_ic(16)

    l_ic   = loss_ic(net, zeta_ic)
    l_anch = loss_anchor(net, tau_anchor, y_anchor)
    l_data = loss_data(net, data_list)
    l_phys = loss_sde_residual(net, tau_col, zeta_phys, log_theta, log_sigma)
    l_reg  = loss_sigma_prior(log_sigma)
    l_kh   = loss_kh_prior(log_theta)

    loss = (W_IC_P1*l_ic + W_ANCH_P1*l_anch + W_DATA_P1*l_data
            + W_PHYS_P1*l_phys + W_REG_P1*l_reg + W_KH_P1*l_kh)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(net.parameters()) + sigma_params, 1.0)
    opt_p1.step()

    if step % 500 == 0:
        _, sigs = get_param_values()
        print(f"  step {step:>5d} | loss={loss.item():.3e} "
              f"phys={l_phys.item():.3e} data={l_data.item():.3e} "
              f"reg={l_reg.item():.3e}")
        print(f"           | "
              f"σX={sigs['X']:.3e} σN={sigs['N']:.3e} "
              f"σS={sigs['S']:.3e} σL={sigs['L']:.3e} σH={sigs['H']:.3e}")
        sigma_history_p1.append((step, {k:float(torch.exp(v).item()) 
                                         for k,v in log_sigma.items()}))

phase1_state = copy.deepcopy(net.state_dict())
phase1_sigma = {k: v.clone().detach() for k,v in log_sigma.items()}
print("Phase 1 complete")

print("\n" + "="*65)
print("PHASE 2 — Joint theta + sigma + network weights")
print("="*65)

net.load_state_dict(phase1_state)
for k,v in log_sigma.items():
    log_sigma[k] = nn.Parameter(phase1_sigma[k].clone().requires_grad_(True))

theta_params = list(log_theta.values())
sigma_params = list(log_sigma.values())

opt_p2 = torch.optim.Adam(
    list(net.parameters()) + theta_params + sigma_params, lr=2e-4
)

W_IC_P2   = 50.0
W_ANCH_P2 = 3.0
W_DATA_P2 = 5.0
W_PHYS_P2 = 3.0
W_REG_P2  = 0.1   # weight on log-normal prior
W_KH_P2   = 2.0   # weight on KH prior — tight anchor
N_STEPS_P2 = 8000

theta_sigma_history = []

for step in range(N_STEPS_P2):
    opt_p2.zero_grad()

    tau_col   = sample_collocation(N_COL)
    zeta_phys = sample_noise(N_NOISE_SAMPLES)
    zeta_ic   = sample_noise_ic(16)

    l_ic   = loss_ic(net, zeta_ic)
    l_anch = loss_anchor(net, tau_anchor, y_anchor)
    l_data = loss_data(net, data_list)
    l_phys = loss_sde_residual(net, tau_col, zeta_phys, log_theta, log_sigma)
    l_reg  = loss_sigma_prior(log_sigma)
    l_kh   = loss_kh_prior(log_theta)

    loss = (W_IC_P2*l_ic + W_ANCH_P2*l_anch + W_DATA_P2*l_data
            + W_PHYS_P2*l_phys + W_REG_P2*l_reg + W_KH_P2*l_kh)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(net.parameters()) + theta_params + sigma_params, 1.0
    )
    opt_p2.step()

    if step % 500 == 0:
        print_params(step)
        vals, sigs = get_param_values()
        theta_sigma_history.append((step, vals.copy(), sigs.copy()))

print("\nPhase 2 complete")

final_theta, final_sigma = get_param_values()
print("\n" + "="*65)
print("FINAL RESULTS")
print("-"*65)
print(f"{'Parameter':<12} {'Inferred':>12} {'Literature':>12} {'Δ%':>8}")
for name in ["rhoN_max","mu_max","r3","KH","r1"]:
    inf_v = final_theta[name]; lit_v = p[name]
    print(f"  {name:<10} {inf_v:>12.5f} {lit_v:>12.5f} {100*(inf_v-lit_v)/lit_v:>+7.1f}%")
print(f"\n{'State':<8} {'sigma':>12} {'h_i type':>20}")
h_types = {"X":"sqrt(X) CIR","N":"sqrt(N) CIR","qN":"qN-qN0",
           "A":"sqrt(A) CIR","S":"sqrt(S+eps)","L":"sqrt(L+eps)","H":"x* active"}
for name,h in h_types.items():
    print(f"  {name:<8} {final_sigma[name]:>12.5e}  {h}")
print("="*65)

print(f"\nGenerating {N_ENSEMBLE}-sample Monte Carlo ensemble...")
net.eval()

tau_dense_np = np.linspace(0.0, 1.0, 2001, dtype=np.float32)
tau_dense    = torch.tensor(tau_dense_np, device=device).unsqueeze(-1)  # (2001,1)
t_dense      = tau_dense_np * T_END

ensemble = np.zeros((N_ENSEMBLE, 2001, 7), dtype=np.float32)

with torch.no_grad():
    for m in range(N_ENSEMBLE):
        zeta_m = torch.randn(2001, N_KKL, device=device)
        y_pred = net(tau_dense, zeta_m).cpu().numpy()  # (2001, 7)
        ensemble[m] = y_pred * scale7[None,:]  # convert to physical units

# Deterministic mean trajectory (zeta=0)
with torch.no_grad():
    zeta_zero = torch.zeros(2001, N_KKL, device=device)
    y_mean_norm = net(tau_dense, zeta_zero).cpu().numpy()
    y_mean = y_mean_norm * scale7[None,:]

# Ensemble statistics
ens_mean = ensemble.mean(axis=0)   # (2001, 7)
ens_std  = ensemble.std(axis=0)    # (2001, 7)
ens_lo   = ens_mean - 2*ens_std
ens_hi   = ens_mean + 2*ens_std

print("Ensemble generated.")

STATE_NAMES = ["Biomass","Nitrogen","N quota","Acetate","Starch","Lipids","pH"]
YLABELS     = ["X (gC L⁻¹)","N (gN L⁻¹)","qN (gN gC⁻¹)","A (gC L⁻¹)",
               "S (gC L⁻¹)","L (gC L⁻¹)","H (–)"]
DIGIT_VARS  = ["Biomass","Nitrogen","N quota","Acetate","Starch","Lipids","pH"]

t_dense_np = tau_dense_np * T_END  # physical time array (2001,)
lsoda_lit  = sol_lit.sol(t_dense_np)[IDX7,:].T
lsoda_L5   = sol_L5.sol(t_dense_np)[IDX7,:].T

fig1, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()
fig1.suptitle(
    "SDE-PINN — Option C (State-Dependent Diffusion)\n"
    "Monte Carlo ensemble (200 realisations) | Blue band = mean ± 2σ",
    fontsize=11, fontweight="bold"
)

for i, (varname, ylabel) in enumerate(zip(STATE_NAMES, YLABELS)):
    ax = axes[i]
    # Uncertainty band
    ax.fill_between(t_dense_np, ens_lo[:,i], ens_hi[:,i],
                    alpha=0.25, color="#1f77b4", label="Mean ± 2σ")
    # Trajectories
    ax.plot(t_dense_np, lsoda_lit[:,i], "k--", lw=1.2, label="LSODA lit", alpha=0.6)
    ax.plot(t_dense_np, lsoda_L5[:,i],  "g-",  lw=1.0, label="LSODA L5",  alpha=0.6)
    ax.plot(t_dense_np, y_mean[:,i],    "b-",  lw=2.0, label="SDE mean (ζ=0)")
    ax.plot(t_dense_np, ens_mean[:,i],  "r-",  lw=1.5, ls="--", label="Ensemble mean", alpha=0.8)

    # Data points
    comp, sc_idx = DIGIT_MAP.get(varname, (i,i))
    pts = digit[digit["variable"]==varname].sort_values("time_h")
    if not pts.empty:
        ax.scatter(pts["time_h"], pts["value"], s=30, c="k", zorder=5, label="Data")

    ax.set_title(f"({chr(97+i)}) {varname}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (h)"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=6); ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlim(-3, 205)

axes[7].axis("off")
plt.tight_layout()
plt.savefig("fig_sde_bands.png", dpi=150, bbox_inches="tight")
print("Saved: fig_sde_bands.png")
plt.show()

fig2, axes2 = plt.subplots(2, 4, figsize=(18, 7))
axes2 = axes2.ravel()
fig2.suptitle(
    "SDE-PINN — Uncertainty Band Width (2σ envelope)\n"
    "Physically expected: widest during nitrogen transition, narrow at steady state",
    fontsize=10, fontweight="bold"
)
for i, varname in enumerate(STATE_NAMES):
    ax = axes2[i]
    band_width = 4 * ens_std[:,i]  # full 4σ = 2 × (2σ)
    ax.plot(t_dense_np, band_width, color=f"C{i}", lw=2)
    # Mark transition zone
    ax.axvspan(40, 100, alpha=0.1, color="red", label="N depletion zone")
    ax.set_title(f"({chr(97+i)}) {varname} band width", fontsize=9)
    ax.set_xlabel("Time (h)"); ax.set_ylabel(f"4σ width")
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, alpha=0.4)
axes2[7].axis("off")
plt.tight_layout()
plt.savefig("fig_sde_bandwidth.png", dpi=150, bbox_inches="tight")
print("Saved: fig_sde_bandwidth.png")
plt.show()

if theta_sigma_history:
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("SDE-PINN Phase 2 — Parameter + Sigma Convergence",
                  fontsize=10, fontweight="bold")

    steps = [h[0] for h in theta_sigma_history]

    # Theta
    ax = axes3[0]
    colors = ["#CC1111","#0066CC","#009900","#CC6600","#9900CC"]
    for ci, name in enumerate(["rhoN_max","mu_max","r3","KH","r1"]):
        vals = [h[1][name]/p[name] for h in theta_sigma_history]
        ax.plot(steps, vals, "-o", color=colors[ci], lw=2, ms=4,
                label=f"{name} (lit norm=1)")
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Literature")
    ax.set_title("Kinetic parameter convergence (normalised)")
    ax.set_xlabel("Training step"); ax.set_ylabel("Inferred/Literature")
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, alpha=0.4)

    # Sigma
    ax = axes3[1]
    sig_colors = [f"C{i}" for i in range(7)]
    for ci, name in enumerate(["X","N","qN","A","S","L","H"]):
        sigs = [h[2][name] for h in theta_sigma_history]
        ax.semilogy(steps, sigs, "-o", color=sig_colors[ci], lw=2, ms=4, label=f"σ_{name}")
    ax.set_title("Diffusion coefficient convergence (log scale)")
    ax.set_xlabel("Training step"); ax.set_ylabel("σ value")
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig("fig_sde_convergence.png", dpi=150, bbox_inches="tight")
    print("Saved: fig_sde_convergence.png")
    plt.show()

N_SPAGHETTI = 30
fig4, axes4 = plt.subplots(2, 4, figsize=(18, 9))
axes4 = axes4.ravel()
fig4.suptitle(
    f"SDE-PINN — Individual stochastic realisations ({N_SPAGHETTI} paths)\n"
    "Each green line = one noise realisation | Red = ensemble mean",
    fontsize=10, fontweight="bold"
)
for i, (varname, ylabel) in enumerate(zip(STATE_NAMES, YLABELS)):
    ax = axes4[i]
    for m in range(N_SPAGHETTI):
        ax.plot(t_dense_np, ensemble[m,:,i], color="#2ca02c", lw=0.4, alpha=0.4)
    ax.plot(t_dense_np, ens_mean[:,i], "r-", lw=2, label="Ensemble mean")
    ax.plot(t_dense_np, lsoda_lit[:,i], "k--", lw=1.2, alpha=0.6, label="LSODA lit")
    pts = digit[digit["variable"]==varname].sort_values("time_h")
    if not pts.empty:
        ax.scatter(pts["time_h"], pts["value"], s=30, c="k", zorder=5, label="Data")
    ax.set_title(f"({chr(97+i)}) {varname}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (h)"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=6); ax.grid(True, lw=0.3, alpha=0.4)
axes4[7].axis("off")
plt.tight_layout()
plt.savefig("fig_sde_spaghetti.png", dpi=150, bbox_inches="tight")
print("Saved: fig_sde_spaghetti.png")
plt.show()

torch.save({
    "net_state":    net.state_dict(),
    "log_theta":    {k: v.detach().cpu() for k,v in log_theta.items()},
    "log_sigma":    {k: v.detach().cpu() for k,v in log_sigma.items()},
    "final_theta":  final_theta,
    "final_sigma":  final_sigma,
    "ensemble":     ensemble,
    "ens_mean":     ens_mean,
    "ens_std":      ens_std,
    "t_dense":      t_dense_np,
    "scale7":       scale7,
}, "sde_pinn_optionC_results.pt")
print("Results saved: sde_pinn_optionC_results.pt")

print("\n" + "="*65)
print("ALL COMPLETE")
print("  fig_sde_bands.png       — headline uncertainty figure")
print("  fig_sde_bandwidth.png   — band width over time (key diagnostic)")
print("  fig_sde_convergence.png — theta + sigma convergence")
print("  fig_sde_spaghetti.png   — individual realisations")
print("="*65)