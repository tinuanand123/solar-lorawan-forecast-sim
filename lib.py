import numpy as np, ctypes, time, os
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
LAT=30.0; A_PANEL=1e-4; ETA_PV=0.17; ETA_MPPT=0.80
V_NOM=3.7; VCC=3.3
# --- node energy model (assumed, see paper Table) ---
SF=11; BW=125e3; CR=1; PAYLOAD=24; MACOH=13; NPRE=8
I_TX=0.120; I_RX=0.012; I_SLEEP=15e-6; E_SENSE=8e-3; T_RX=2*0.030
def toa(sf=SF,bw=BW,cr=CR,pl=PAYLOAD+MACOH,npre=NPRE,crc=1,ih=0,de=1):
    tsym=2**sf/bw
    tpre=(npre+4.25)*tsym
    nb=8+max(np.ceil((8*pl-4*sf+28+16*crc-20*ih)/(4*(sf-2*de)))*(cr+4),0)
    return tpre+nb*tsym
T_AIR=toa()
E_UP=I_TX*VCC*T_AIR+I_RX*VCC*T_RX+E_SENSE     # J per uplink cycle
S_HR=I_SLEEP*VCC*3600.0                        # J per hour sleep floor
RMIN,RMAX=1.0,30.0
ETA_C=0.95; ETA_D=0.95
FLOOR=0.10; RESUME=0.10; MARGIN=0.10; TGT=0.60; E0=0.60; WARM=14*24
# --- weather generator (assumed, synthetic) ---
KT_MEAN=np.array([0.85,0.55,0.25,0.10]); KT_SD=np.array([0.05,0.18,0.08,0.04])
PI={'winter':[.55,.30,.12,.03],'pre':[.62,.28,.08,.02],'monsoon':[.10,.25,.40,.25],'post':[.65,.25,.08,.02]}
RHO=0.55; PHI=0.70
def season(doy):
    m=int((doy-1)//30.42)+1
    if m in (12,1,2): return 'winter'
    if m in (3,4,5): return 'pre'
    if m in (6,7,8,9): return 'monsoon'
    return 'post'
def clear_ghi(doy,hmid):
    dec=np.radians(23.45*np.sin(2*np.pi*(284+doy)/365)); ha=np.radians(15*(hmid-12)); lat=np.radians(LAT)
    cz=np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha)
    czc=np.maximum(cz,0.02)
    return np.where(cz>0.02,1098.0*czc*np.exp(-0.059/czc),0.0)
def gen(years,seed,monsoon_shift=0.0):
    rng=np.random.default_rng(seed); nd=365*years
    st=np.zeros(nd,int); doy=np.arange(nd)%365+1
    seasons=[season(d) for d in range(1,366)]
    st[0]=rng.choice(4,p=PI['winter'])
    for d in range(1,nd):
        p=np.array(PI[seasons[(d)%365]])
        if seasons[d%365]=='monsoon' and monsoon_shift!=0:
            p=p.copy(); p[2:]*=(1+monsoon_shift); p[:2]*=(1-monsoon_shift*0.5); p/=p.sum()
        st[d]= st[d-1] if rng.random()<RHO else rng.choice(4,p=p)
    n=nd*24; hod=np.tile(np.arange(24),nd); dday=np.repeat(doy,24); sth=np.repeat(st,24)
    eps=rng.standard_normal(n); z=np.zeros(n)
    for i in range(1,n): z[i]=PHI*z[i-1]+np.sqrt(1-PHI**2)*eps[i]
    kt=np.clip(KT_MEAN[sth]+KT_SD[sth]*z,0.02,1.05)
    cs=clear_ghi(dday,hod+0.5)
    kt_n=np.where(cs>0,kt,np.nan)
    H=cs*np.where(cs>0,kt,0)*A_PANEL*ETA_PV*ETA_MPPT*3600.0
    return dict(kt=kt_n,cs=cs,H=H,hod=hod,doy=dday,state=sth,n=n)
FEATS=['ktl1','kt3','kt12','d1','d2','d3','sdoy','cdoy','shod_t','chod_t','shod_g','chod_g','csg','lead']
def base_feats(w):
    kt=w['kt']; n=w['n']; valid=~np.isnan(kt)
    idx=np.where(valid,np.arange(n),0); np.maximum.accumulate(idx,out=idx); ff=kt[idx]; ff=np.nan_to_num(ff,nan=0.5)
    c=np.cumsum(ff); 
    def mv(L): 
        o=np.empty(n); o[:]=np.nan; o[L:]=(c[L:]-c[:-L])/L; o[:L]=ff[:L]; return o
    kt3=mv(3); kt12=mv(12)
    k0=np.where(valid,kt,0.0); S=np.cumsum(k0); V=np.cumsum(valid.astype(float))
    def win(a,b):
        o=np.full(n,0.5); t=np.arange(n); ok=t>=b
        s=S[t[ok]-a]-S[t[ok]-b] if a>0 else S[t[ok]]-S[t[ok]-b]
        v=V[t[ok]-a]-V[t[ok]-b] if a>0 else V[t[ok]]-V[t[ok]-b]
        o[ok]=np.where(v>0,s/np.maximum(v,1),0.5); return o
    d1=win(0,24); d2=win(24,48); d3=win(48,72)
    return np.stack([ff,kt3,kt12,d1,d2,d3],1)
def build(w,T0=72,with_y=True):
    n=w['n']; B=base_feats(w); T=np.arange(T0,n-24); K=np.arange(1,25)
    ang=2*np.pi*w['doy']/365; hod=w['hod']
    TT,KK=np.meshgrid(T,K,indexing='ij'); G=TT+KK
    X=np.empty((len(T),24,14),np.float32)
    X[:,:,:6]=B[T][:,None,:]
    X[:,:,6]=np.sin(ang[T])[:,None]; X[:,:,7]=np.cos(ang[T])[:,None]
    X[:,:,8]=np.sin(2*np.pi*hod[T]/24)[:,None]; X[:,:,9]=np.cos(2*np.pi*hod[T]/24)[:,None]
    X[:,:,10]=np.sin(2*np.pi*hod[G]/24); X[:,:,11]=np.cos(2*np.pi*hod[G]/24)
    X[:,:,12]=w['cs'][G]/1000.0; X[:,:,13]=KK
    X=X.reshape(-1,14)
    y=w['kt'][G].reshape(-1) if with_y else None
    return T,X,y,G.reshape(-1)
def train_models(ws,drop=(),hgb_iter=100):
    Xs=[];ys=[]
    for w in ws:
        T,X,y,G=build(w); m=~np.isnan(y); Xs.append(X[m]); ys.append(y[m])
    X=np.concatenate(Xs); y=np.concatenate(ys)
    keep=[i for i,f in enumerate(FEATS) if f not in drop]
    t=time.time(); rg=Ridge(alpha=1.0).fit(X[:,keep],y); t_r=time.time()-t
    t=time.time(); hg=HistGradientBoostingRegressor(max_iter=hgb_iter,learning_rate=0.1,max_depth=6,random_state=0).fit(X[:,keep],y); t_h=time.time()-t
    return dict(ridge=rg,hgb=hg,keep=keep,t_ridge=t_r,t_hgb=t_h,ntrain=len(y))
def forecasts(w,models,T0=72):
    """returns dict name -> F array (n*24) of forecast energy (J) at decision index i for hour i+k (k=0..23) ; uses features at t=i-1"""
    n=w['n']; H=w['H']; cs=w['cs']; T,X,y,G=build(w,T0)
    out={}
    def pack(Ehat):  # Ehat shape (len(T),24) energy for target t+1..t+24, decision i=t+1
        F=np.zeros((n,24)); F[T+1]=Ehat; return F.reshape(-1)
    csG=cs[G].reshape(-1,24)*A_PANEL*ETA_PV*ETA_MPPT*3600.0
    for nm in ('ridge','hgb'):
        p=models[nm].predict(X[:,models['keep']]).reshape(-1,24); p=np.clip(p,0.0,1.05)
        out[nm]=(p,pack(p*csG))
    # persistence (same hour yesterday, energy), EWMA per slot (alpha=0.5), oracle
    Gm=G.reshape(-1,24)
    pers=H[Gm-24]
    ew=np.zeros(n); al=0.5
    for ph in range(24):
        idx=np.arange(ph,n,24); e=0.0
        for j in idx:
            e=al*e+(1-al)*H[j] if j>=24 else H[j]; ew[j]=e
    ewf=ew[Gm-24]
    orc=H[Gm]
    out['persist']=(None,pack(pers)); out['ewma']=(None,pack(ewf)); out['oracle']=(None,pack(orc))
    return out,T,Gm
def sim(lib,w,F,cap_mah,mode,rfix=0.0,eu_scale=1.0,hscale=1.0,trace=0,rmin=None):
    rmin=RMIN if rmin is None else rmin
    n=w['n']; cap=cap_mah*V_NOM*3.6
    H=np.ascontiguousarray(w['H']*hscale); 
    if F is None: F=np.zeros(1)
    F=np.ascontiguousarray(F*hscale)
    out=np.zeros(7); st=np.zeros(max(trace,1)); rt=np.zeros(max(trace,1))
    lib.simulate(n,72,WARM,H.ctypes.data_as(ctypes.c_void_p),F.ctypes.data_as(ctypes.c_void_p),ctypes.c_double(cap),mode,ctypes.c_double(rfix),
        ctypes.c_double(S_HR),ctypes.c_double(E_UP*eu_scale),ctypes.c_double(rmin),ctypes.c_double(RMAX),ctypes.c_double(ETA_C),ctypes.c_double(ETA_D),
        ctypes.c_double(FLOOR),ctypes.c_double(RESUME),ctypes.c_double(MARGIN),ctypes.c_double(TGT),ctypes.c_double(E0),
        out.ctypes.data_as(ctypes.c_void_p),st.ctypes.data_as(ctypes.c_void_p),rt.ctypes.data_as(ctypes.c_void_p),trace)
    return out,(st,rt)
def load():
    lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'core.so')); lib.simulate.restype=None; return lib
