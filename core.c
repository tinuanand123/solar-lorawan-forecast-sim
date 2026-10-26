#include <math.h>
#include <string.h>
/* Hourly energy-harvesting node simulator.
   mode 0: fixed rate rfix; mode 1: reactive SoC; mode 2: forecast-driven receding-horizon planner.
   H[i]: harvested energy (J) in hour i (after MPPT). F[i*24+k]: forecast of H[i+k] made at start of hour i. */
static int feasible(double E,double r,const double*f,double cap,double Efl,double Eres,double Etgt,
                    double s,double eu,double etac,double etad){
  double D=s+r*eu;
  for(int k=0;k<24;k++){
    double sur=f[k]-D;
    if(sur>=0){E+=etac*sur; if(E>cap)E=cap;} else E+=sur/etad;
    if(E<Eres) return 0;
  }
  return E>=Etgt;
}
void simulate(int n,int t0,int warm,const double*H,const double*F,double cap,int mode,double rfix,
  double s,double eu,double rmin,double rmax,double etac,double etad,
  double floor_frac,double resume_frac,double margin_frac,double tgt_frac,double E0frac,
  double*out,double*soc_tr,double*rate_tr,int ntr){
  double Efl=floor_frac*cap, Eres=Efl+margin_frac*cap, Etgt=Efl+tgt_frac*(cap-Efl);
  double E=Efl+E0frac*(cap-Efl); int on=1;
  double avail=0,up=0,spill=0,harv=0,load=0,thr=0,spw=0,ldw=0,thw=0; int cnt=0;
  for(int i=t0;i<n-24;i++){
    if(i==t0+warm){spw=spill;ldw=load;thw=thr;}
    double r=0;
    if(mode==0) r=rfix;
    else if(mode==1){ double sf=(E-Efl)/(cap-Efl); double x=(sf-0.2)/0.6; if(x<0)x=0; if(x>1)x=1; r=rmin+(rmax-rmin)*x; }
    else {
      const double*f=F+(long)i*24;
      if(feasible(E,rmax,f,cap,Efl,Eres,Etgt,s,eu,etac,etad)) r=rmax;
      else if(!feasible(E,rmin,f,cap,Efl,Eres,Etgt,s,eu,etac,etad)) r=rmin;
      else { double lo=rmin,hi=rmax; for(int it=0;it<14;it++){ double m=0.5*(lo+hi);
               if(feasible(E,m,f,cap,Efl,Eres,Etgt,s,eu,etac,etad)) lo=m; else hi=m; } r=lo; }
    }
    double h=H[i]; double fa=0, upl=0;
    if(!on){
      double Eu=E+etac*h; if(Eu>cap){spill+=(Eu-cap)/etac; Eu=cap;} E=Eu;
      if(E>=Efl+resume_frac*cap) on=1;
    } else {
      double D=s+r*eu; double sur=h-D;
      if(sur>=0){ double Eu=E+etac*sur; if(Eu>cap){spill+=(Eu-cap)/etac; Eu=cap;} E=Eu; fa=1; upl=r; load+=D; }
      else { double draw=-sur/etad;
        if(E-draw>=Efl){ thr+=draw; E-=draw; fa=1; upl=r; load+=D; }
        else { double av=h+(E-Efl)*etad; double f2=av/D; if(f2<0)f2=0; if(f2>1)f2=1;
               fa=f2; upl=f2*r; load+=f2*D; thr+=(E-Efl); E=Efl; on=0; }
      }
    }
    if(i>=t0+warm){ avail+=fa; up+=upl; harv+=h; cnt++; }
    int j=i-t0; if(soc_tr&&j<ntr){ soc_tr[j]=(E-Efl)/(cap-Efl); rate_tr[j]=on?r:0; }
  }
  out[0]=avail/cnt; out[1]=up/cnt; out[2]=harv>0?(spill-spw)/harv:0; out[3]=harv; out[4]=load-ldw; out[5]=thr-thw; out[6]=cnt;
}
