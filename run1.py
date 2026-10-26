import sys,pickle,time; sys.path.insert(0,'.')
from lib import *
lib=load()
TRAIN=[100,101,102,103]; TEST=list(range(200,210))
t=time.time(); wtr=[gen(2,s) for s in TRAIN]; wte=[gen(2,s) for s in TEST]; print('gen',time.time()-t)
t=time.time(); M=train_models(wtr); print('train',time.time()-t,M['t_ridge'],M['t_hgb'],M['ntrain'])
pickle.dump(M,open('models.pkl','wb'))
CTRL=['fixed','reactive','persist','ewma','ridge','hgb','oracle']
FC=[]; 
t=time.time()
for w in wte:
    out,T,Gm=forecasts(w,M); FC.append((out,T,Gm))
print('fc',time.time()-t)
# forecast accuracy
acc={m:{g:[] for g in ('1-6','7-12','13-24','all')} for m in ('persist','ewma','ridge','hgb')}
accm={m:{g:[] for g in ('1-6','7-12','13-24','all')} for m in acc}
for w,(out,T,Gm) in zip(wte,FC):
    Htrue=w['H'][Gm]; day=w['cs'][Gm]>0
    for m in acc:
        F=out[m][1].reshape(-1,24)[T+1]  # decision index i=t+1
        e=(F-Htrue)
        for g,(a,b) in {'1-6':(0,6),'7-12':(6,12),'13-24':(12,24),'all':(0,24)}.items():
            mk=day[:,a:b]; ee=e[:,a:b][mk]
            acc[m][g].append(np.sqrt(np.mean(ee**2))); accm[m][g].append(np.mean(np.abs(ee)))
mean_day_E=np.mean([w['H'][w['cs']>0].mean() for w in wte])
print('mean daytime hourly E (J)',mean_day_E)
for m in acc: print(m,{g:(np.mean(v)*1e3).round(1) for g,v in acc[m].items()})
# fixed-rate design
CAPS=[2.5,5,7.5,10,15,20,30,50,75,100,150,200,300,500,750,1000]
rfixed={}
for c in CAPS:
    best=RMIN
    for r in range(int(RMAX),0,-1):
        ok=all(sim(lib,w,None,c,0,rfix=float(r))[0][0]>=0.999999 for w in wtr)
        if ok: best=float(r); break
    rfixed[c]=best
print('rfixed',rfixed)
res={}
t=time.time()
for c in CAPS:
    for cn in CTRL:
        rows=[]
        for w,(out,T,Gm) in zip(wte,FC):
            if cn=='fixed': o,_=sim(lib,w,None,c,0,rfix=rfixed[c])
            elif cn=='reactive': o,_=sim(lib,w,None,c,1)
            else: o,_=sim(lib,w,out[cn][1],c,2)
            rows.append(o.copy())
        res[(c,cn)]=np.array(rows)
print('sim',time.time()-t)
pickle.dump(dict(res=res,acc=acc,accm=accm,rfixed=rfixed,CAPS=CAPS,mean_day_E=mean_day_E),open('res1.pkl','wb'))
for c in (2.5,5,10,20,50,100,300):
    print('cap',c)
    for cn in CTRL:
        r=res[(c,cn)]; print('  %-9s avail %.4f (min %.4f) rate %.1f spill %.2f'%(cn,r[:,0].mean(),r[:,0].min(),r[:,1].mean(),r[:,2].mean()))
