import sys,pickle,time; sys.path.insert(0,'.')
from lib import *
from scipy.stats import wilcoxon
lib=load(); R=pickle.load(open('res1.pkl','rb')); res=R['res']; M=pickle.load(open('models.pkl','rb'))
TRAIN=[100,101,102,103]; TEST=list(range(200,210))
wtr=[gen(2,s) for s in TRAIN]; wte=[gen(2,s) for s in TEST]
FC=[forecasts(w,M)[0] for w in wte]
CTRL=['fixed','reactive','persist','ewma','ridge','hgb','oracle']
# ---- min capacity criterion: pooled avail>=0.999 and mean rate>=12/h
def agg(c,cn): r=res[(c,cn)]; return r[:,0].mean(), r[:,1].mean(), r[:,0].min()
mincap={}
for cn in CTRL:
    mincap[cn]=None
    for c in R['CAPS']:
        a,y,_=agg(c,cn)
        if a>=0.999 and y>=12.0: mincap[cn]=c; break
print('mincap',mincap)
# ---- paired stats at cap 5 mAh (tight)
st={}
for c in (2.5,5,10):
    for a,b in (('hgb','persist'),('hgb','ewma'),('hgb','reactive'),('ridge','ewma')):
        for k,nm in ((0,'avail'),(1,'rate')):
            x=res[(c,a)][:,k]; y=res[(c,b)][:,k]
            try: p=wilcoxon(x,y).pvalue
            except Exception: p=np.nan
            st[(c,a,b,nm)]=(x.mean()-y.mean(),p)
for k,v in st.items(): print(k,np.round(v,5))
# ---- permutation importance (HGB, kt RMSE)
w=wte[0]; T,X,y,G=build(w); m=~np.isnan(y); idx=np.where(m)[0]; rng=np.random.default_rng(0); idx=rng.choice(idx,120000,replace=False)
Xs=X[idx]; ys=y[idx]; keep=M['keep']
base=np.sqrt(np.mean((np.clip(M['hgb'].predict(Xs[:,keep]),0,1.05)-ys)**2)); imp={}
for j,f in enumerate(FEATS):
    Xp=Xs.copy(); Xp[:,j]=rng.permutation(Xp[:,j]); p=np.clip(M['hgb'].predict(Xp[:,keep]),0,1.05); imp[f]=np.sqrt(np.mean((p-ys)**2))-base
print('base rmse kt',base); print({k:round(v,4) for k,v in imp.items()})
# ---- inference timing (host CPU) 
Xt=X[:24]; t=time.time(); 
for _ in range(200): M['hgb'].predict(Xt)
t_h=(time.time()-t)/200
t=time.time()
for _ in range(200): M['ridge'].predict(Xt)
t_r=(time.time()-t)/200
print('infer ms per decision (24 leads) hgb',t_h*1e3,'ridge',t_r*1e3)
nodes=sum(p[0].nodes.shape[0] for p in M['hgb']._predictors)
print('hgb trees',len(M['hgb']._predictors),'nodes',nodes)
# ---- ablation (HGB, retrained) at cap 5 and forecast RMSE
abl={}
groups={'full':(),'-recent kt (ktl1,kt3,kt12)':('ktl1','kt3','kt12'),'-daily history (d1,d2,d3)':('d1','d2','d3'),'-season (doy)':('sdoy','cdoy'),'-decision-hour clock':('shod_t','chod_t')}
for nm,drop in groups.items():
    Mm=M if nm=='full' else train_models(wtr,drop=drop)
    rm=[];rows=[]
    for w in wte:
        out,T,Gm=forecasts(w,Mm); Htrue=w['H'][Gm]; day=w['cs'][Gm]>0
        F=out['hgb'][1].reshape(-1,24)[T+1]; rm.append(np.sqrt(np.mean((F-Htrue)[day]**2)))
        for c in (5,):
            o,_=sim(lib,w,out['hgb'][1],c,2); rows.append(o)
    rows=np.array(rows); abl[nm]=(np.mean(rm),rows[:,0].mean(),rows[:,1].mean())
    print('abl',nm,abl[nm])
# ---- noise/bias injection on oracle forecast at cap 5 and 10
inj={}
rng=np.random.default_rng(5)
for c in (5,10):
    for kind,vals in (('noise',[0,0.1,0.2,0.3,0.5]),('bias',[-0.3,-0.15,0,0.15,0.3])):
        for v in vals:
            rows=[]
            for w,out in zip(wte,FC):
                F=out['oracle'][1].copy()
                if kind=='noise': F=F*np.exp(rng.normal(0,v,F.shape)-0.5*v*v)
                else: F=F*(1+v)
                o,_=sim(lib,w,F,c,2); rows.append(o)
            rows=np.array(rows); inj[(c,kind,v)]=(rows[:,0].mean(),rows[:,1].mean(),rows[:,0].min())
            print('inj',c,kind,v,np.round(inj[(c,kind,v)],4))
# ---- panel-area and TX-energy sensitivity at cap 10 for fixed(tuned at base) / reactive / hgb
sens={}
rfx=R['rfixed']
for sc in (0.5,0.75,1.0,1.5,2.0):
    for cn in ('fixed','reactive','hgb'):
        rows=[]
        for w,out in zip(wte,FC):
            if cn=='fixed': o,_=sim(lib,w,None,10,0,rfix=rfx[10],hscale=sc)
            elif cn=='reactive': o,_=sim(lib,w,None,10,1,hscale=sc)
            else: o,_=sim(lib,w,out['hgb'][1],10,2,hscale=sc)
            rows.append(o)
        rows=np.array(rows); sens[('area',sc,cn)]=(rows[:,0].mean(),rows[:,1].mean())
for sc in (0.5,1.0,1.5):
    for cn in ('fixed','reactive','hgb'):
        rows=[]
        for w,out in zip(wte,FC):
            if cn=='fixed': o,_=sim(lib,w,None,10,0,rfix=rfx[10],eu_scale=sc)
            elif cn=='reactive': o,_=sim(lib,w,None,10,1,eu_scale=sc)
            else: o,_=sim(lib,w,out['hgb'][1],10,2,eu_scale=sc)
            rows.append(o)
        rows=np.array(rows); sens[('eu',sc,cn)]=(rows[:,0].mean(),rows[:,1].mean())
for k,v in sens.items(): print(k,np.round(v,4))
# ---- Scenario B: service floor rmin=6/h
scB={}
for c in (5,10,20,50,100,200):
    for cn in ('fixed','reactive','persist','ewma','hgb','oracle'):
        rows=[]
        for w,out in zip(wte,FC):
            if cn=='fixed': o,_=sim(lib,w,None,c,0,rfix=6.0,rmin=6.0)
            elif cn=='reactive': o,_=sim(lib,w,None,c,1,rmin=6.0)
            else: o,_=sim(lib,w,out[cn][1],c,2,rmin=6.0)
            rows.append(o)
        rows=np.array(rows); scB[(c,cn)]=(rows[:,0].mean(),rows[:,0].min(),rows[:,1].mean())
for k,v in scB.items(): print('B',k,np.round(v,4))
# ---- traces for figure: 21 days from realization 0, cap 10
w=wte[0]; out=FC[0]; tr={}
for cn,mode in (('fixed',0),('reactive',1),('hgb',2),('oracle',2)):
    o,(s_,r_)=sim(lib,w,None if mode<2 else out[cn][1],10,mode,rfix=rfx[10],trace=24*400)
    tr[cn]=(s_.copy(),r_.copy())
# energy utilisation, EFC
extra={}
for cn in CTRL:
    r=res[(10,cn)]; extra[cn]=dict(util=(r[:,4]/r[:,3]).mean(), efc_yr=(r[:,5]/(10*V_NOM*3.6)/ (r[:,6]/8760.0)).mean())
print(extra)
pickle.dump(dict(mincap=mincap,st=st,imp=imp,base=base,t_h=t_h,t_r=t_r,nodes=nodes,abl=abl,inj=inj,sens=sens,scB=scB,tr=tr,extra=extra),open('res2.pkl','wb'))
