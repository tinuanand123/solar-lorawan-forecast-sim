import sys, pickle
sys.path.insert(0, '.')
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from lib import *
os.makedirs('f2', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 7.5, 'axes.linewidth': 0.6, 'figure.dpi': 200, 'savefig.dpi': 230, 'axes.titlesize': 8, 'legend.fontsize': 6.3})
R = pickle.load(open('res1.pkl', 'rb')); R2 = pickle.load(open('res2.pkl', 'rb')); G = pickle.load(open('res3.pkl', 'rb')); M = pickle.load(open('models.pkl', 'rb'))
res = R['res']; CAPS = R['CAPS']
NM = {'fixed': 'Fixed rate', 'reactive': 'Reactive SoC', 'persist': 'Planner + persistence', 'ewma': 'Planner + EWMA', 'ridge': 'Planner + ridge', 'hgb': 'Planner + HGB', 'oracle': 'Planner + oracle'}
ST = {'fixed': ('k', 's', '-'), 'reactive': ('#8c564b', 'v', '--'), 'persist': ('#1f77b4', 'o', '-'), 'ewma': ('#2ca02c', '^', '-'), 'ridge': ('#ff7f0e', 'D', '-'), 'hgb': ('#d62728', '*', '-'), 'oracle': ('0.45', 'x', ':')}
GRN = '#dff0d8'; BLU = '#d9e8f7'; YEL = '#fff3cd'; RED = '#f8d7da'; GRY = '#eeeeee'; PUR = '#e6dcf2'


def new(w, h):
    f = plt.figure(figsize=(w, h)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / w); ax.axis('off'); return f, ax


def box(ax, x, y, w, h, t, fc=BLU, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec='k', lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.15)


def arr(ax, p1, p2, lw=0.9, ls='-', cs='arc3'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=7, lw=lw, color='k', linestyle=ls, connectionstyle=cs))


def lab(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom', ha='right')


# ---- F1 architecture
f, ax = new(6.5, 2.75)
ax.text(50, 39.2, 'SOLAR LoRaWAN SENSING NODE: FROM PANEL TO USER', ha='center', fontsize=7.5, fontweight='bold')
top = [('INPUT\nsunlight on\n1 cm² PV cell', YEL), ('ENERGY PATH\nMPPT charger +\nLi-ion cell', GRN), ('SENSORS\nT, RH, pressure,\nair quality', BLU), ('DATA ACQUISITION\nMCU, ADC / I²C,\nRTC stamp', BLU), ('PREPROCESSING\nfilter and pack\n24-byte payload', BLU), ('COMMUNICATION\nLoRa radio\nSF11, 125 kHz', PUR)]
for i, (t, c) in enumerate(top):
    box(ax, 1 + i * 16.5, 25.5, 14.6, 10.5, t, c, fs=6.2)
    if i < 5: arr(ax, (15.9 + i * 16.5, 30.7), (17.1 + i * 16.5, 30.7))
bot = [(1, 22, 'AI / ML MODULE\n24-h harvest forecast\n(persistence, EWMA,\nridge, HGB)', RED), (28, 22, 'DECISION\nreceding-horizon planner:\nlargest feasible rate r', RED), (55, 20, 'OUTPUT\nuplink interval,\nsleep / wake schedule', BLU), (79, 20, 'GATEWAY → NETWORK\nSERVER → USER / GIS', PUR)]
for x, w, t, c in bot: box(ax, x, 5, w, 13.5, t, c, fs=6.2)
for a_, b_ in [(23.4, 27.6), (50.4, 54.6), (75.4, 78.6)]: arr(ax, (a_, 11.7), (b_, 11.7))
arr(ax, (20, 24.9), (12, 18.9)); ax.text(2, 22, 'SoC, V_bat, kt history', fontsize=5.8)
arr(ax, (65, 18.9), (65, 24.9), ls='--'); ax.text(66, 21.7, 'sets sensing rate', fontsize=5.8)
arr(ax, (89, 24.9), (89, 18.9)); ax.text(90, 21.7, 'uplink', fontsize=5.8)
f.savefig('figures/f1.png'); plt.close(f)

# ---- F2 methodology flowchart + control loop
f, ax = new(6.5, 3.15)
r1 = ['Problem and\ngap review', 'Synthetic\nweather\n(Markov + AR1)', 'Haurwitz GHI,\nPV harvest', 'Features\n(14 inputs)', 'Train 4 forecasters\n(4 × 2 years)']
r2 = ['Test on 10 unseen\n2-year realisations', 'Controllers:\nfixed, reactive,\nplanner + F', 'Metrics: availability,\nyield, spill, EFC', 'Ablation, bias,\narea, radio energy', 'CO₂e indicator,\nconclusions']
xs = [1 + i * 19.8 for i in range(5)]
for i, l in enumerate(r1):
    box(ax, xs[i], 36, 17.2, 11, l, BLU if i else GRN, fs=6.3)
    if i < 4: arr(ax, (xs[i] + 17.5, 41.5), (xs[i + 1] - 0.3, 41.5))
for i, l in enumerate(r2):
    box(ax, xs[4 - i], 21, 17.2, 11, l, RED if i == 4 else BLU, fs=6.3)
    if i < 4: arr(ax, (xs[4 - i] - 0.3, 26.5), (xs[3 - i] + 17.5, 26.5))
arr(ax, (xs[4] + 8.6, 35.6), (xs[4] + 8.6, 32.4))
ax.text(50, 50.3, 'METHODOLOGY (second row read right to left)', ha='center', fontsize=7.4, fontweight='bold')
ax.text(50, 14.5, 'Training seeds 100–103 and test seeds 200–209 are independent; the fixed-rate design also uses training weather only.', ha='center', fontsize=6.4, style='italic')
f.savefig('figures/f2.png'); plt.close(f)

# ---- F3 hourly control loop
f, ax = new(6.5, 2.0)
box(ax, 1, 14, 17, 12, 'History\nkt(t), daily means,\nday of year, hour', YEL, fs=6.2)
box(ax, 24, 14, 19, 12, 'Forecaster F\npersistence | EWMA\nridge | HGB (100 trees)', RED, fs=6.2)
box(ax, 49, 14, 22, 12, 'Planner (bisection)\nSoC(k) ≥ reserve, k ≤ 24\nSoC(24) ≥ target', RED, fs=6.2)
box(ax, 77, 14, 22, 12, 'Node action\nrate r(t) in 1–30 per h\nelse sleep', BLU, fs=6.2)
for a, b in [(18.4, 23.6), (43.4, 48.6), (71.4, 76.6)]: arr(ax, (a, 20), (b, 20))
box(ax, 49, 1, 22, 8, 'Battery: E(t+1) = E(t) + η(H − D)', GRN, fs=6.2)
box(ax, 77, 1, 22, 8, 'Actual harvest H(t) unknown', YEL, fs=6.2)
arr(ax, (88, 13.6), (88, 9.4)); arr(ax, (76.6, 5), (71.4, 5)); arr(ax, (60, 9.4), (60, 13.6)); ax.text(61, 11.3, 'SoC', fontsize=5.8)
arr(ax, (48.6, 5), (9.5, 5)); arr(ax, (9.5, 5), (9.5, 13.6)); ax.text(11, 7, 'history update (hourly)', fontsize=5.8)
f.savefig('figures/f3.png'); plt.close(f)

# ---- F4 traces
tr = R2['tr']; w = gen(2, 200); j0 = 24 * 172; L = 24 * 12; t = np.arange(L) / 24
fig, axs = plt.subplots(3, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.1, 1.1]})
Hh = w['H'][72 + j0:72 + j0 + L]; kt = w['kt'][72 + j0:72 + j0 + L]
axs[0].fill_between(t, Hh, step='mid', color='goldenrod', alpha=0.65); axs[0].set_ylabel('Harvest (J/h)')
a2 = axs[0].twinx(); a2.plot(t, kt, 'k.', ms=1.5); a2.set_ylim(0, 1.1); a2.set_ylabel('$k_t$')
for cn in ('fixed', 'reactive', 'hgb', 'oracle'):
    c, m, ls = ST[cn]; axs[1].plot(t, 100 * tr[cn][0][j0:j0 + L], color=c, ls=ls, lw=1.0, label=NM[cn]); axs[2].plot(t, tr[cn][1][j0:j0 + L], color=c, ls=ls, lw=1.0)
axs[1].set_ylabel('Usable SoC (%)'); axs[1].legend(ncol=2, loc='lower left', frameon=True); axs[2].set_ylabel('Uplinks per hour'); axs[2].set_xlabel('Day of excerpt (monsoon season, realisation 200)')
for a, s in zip(axs, 'abc'): a.grid(alpha=0.25, lw=0.4); lab(a, s)
fig.tight_layout(pad=0.4); fig.savefig('figures/f4.png'); plt.close(fig)

# ---- F5 forecast error + importance
errs = pickle.load(open('errs_lead.pkl', 'rb')) if os.path.exists('errs_lead.pkl') else None
wte = [gen(2, s) for s in range(200, 210)]
if errs is None:
    errs = {m: np.zeros((10, 24)) for m in ('persist', 'ewma', 'ridge', 'hgb')}
    for i, w_ in enumerate(wte):
        out, T, Gm = forecasts(w_, M); Ht = w_['H'][Gm]; day = w_['cs'][Gm] > 0
        for m in errs:
            F = out[m][1].reshape(-1, 24)[T + 1]; e = F - Ht
            for k in range(24): errs[m][i, k] = np.sqrt(np.mean(e[day[:, k], k] ** 2))
    pickle.dump(errs, open('errs_lead.pkl', 'wb'))
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.45), gridspec_kw={'width_ratios': [1.15, 1]})
ld = np.arange(1, 25)
for m in errs:
    c, mk, ls = ST[m]; mu = errs[m].mean(0); sd = errs[m].std(0, ddof=1)
    axs[0].plot(ld, mu, color=c, marker=mk, ms=3, lw=1.0, ls=ls, label=NM[m].replace('Planner + ', '').capitalize() if m not in ('ewma', 'hgb') else NM[m].replace('Planner + ', '')); axs[0].fill_between(ld, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
axs[0].set_xlabel('Forecast lead (h)'); axs[0].set_ylabel('RMSE of hourly harvest (J)'); axs[0].legend(loc='lower right'); axs[0].grid(alpha=0.25, lw=0.4)
imp = R2['imp']; lb = {'ktl1': 'last $k_t$', 'kt3': '$k_t$ 3-h mean', 'kt12': '$k_t$ 12-h mean', 'd1': 'daily mean d−1', 'd2': 'daily mean d−2', 'd3': 'daily mean d−3', 'sdoy': 'sin(doy)', 'cdoy': 'cos(doy)', 'shod_t': 'sin(hour now)', 'chod_t': 'cos(hour now)', 'shod_g': 'sin(hour tgt)', 'chod_g': 'cos(hour tgt)', 'csg': 'clear-sky GHI', 'lead': 'lead'}
ks = sorted(imp, key=lambda k: imp[k])[-9:]
axs[1].barh([lb[k] for k in ks], [imp[k] for k in ks], color='#d62728', edgecolor='k', lw=0.4); axs[1].set_xlabel('Rise in $k_t$ RMSE when shuffled'); axs[1].tick_params(axis='y', labelsize=6.3); axs[1].grid(alpha=0.25, lw=0.4, axis='x')
lab(axs[0], 'a'); lab(axs[1], 'b'); fig.tight_layout(pad=0.4); fig.savefig('figures/f5.png'); plt.close(fig)

# ---- F6 availability / yield / min capacity / carbon
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.7))
order = ['fixed', 'reactive', 'persist', 'ewma', 'ridge', 'hgb', 'oracle']
for cn in order:
    c, mk, ls = ST[cn]
    un = [max(100 * (1 - res[(cc, cn)][:, 0].mean()), 1e-3) for cc in CAPS]
    ym = np.array([res[(cc, cn)][:, 1].mean() for cc in CAPS]); ys = np.array([res[(cc, cn)][:, 1].std(ddof=1) for cc in CAPS])
    axs[0, 0].plot(CAPS, un, color=c, marker=mk, ms=3, lw=0.9, ls=ls, label=NM[cn]); axs[0, 1].plot(CAPS, ym, color=c, marker=mk, ms=3, lw=0.9, ls=ls); axs[0, 1].fill_between(CAPS, ym - ys, ym + ys, color=c, alpha=0.13, lw=0)
axs[0, 0].set_xscale('log'); axs[0, 0].set_yscale('log'); axs[0, 0].axhline(0.1, color='r', lw=0.6, ls=':'); axs[0, 0].text(25, 0.13, '0.1 % limit', fontsize=6, color='r'); axs[0, 0].set_ylabel('Unavailability (%), floor 0.001'); axs[0, 0].legend(loc='upper right', fontsize=5.6)
axs[0, 1].set_xscale('log'); axs[0, 1].axhline(12, color='r', lw=0.6, ls=':'); axs[0, 1].text(2.6, 12.4, '12 per hour target', fontsize=6, color='r'); axs[0, 1].set_ylabel('Mean uplinks per hour (± s.d.)')
for a in axs[0]: a.set_xlabel('Battery capacity (mAh)'); a.grid(alpha=0.25, lw=0.4, which='both')
mc = R2['mincap']; caps = [mc[o] for o in order]; x = np.arange(7)
names = ['Fixed', 'Reactive', 'Pers.', 'EWMA', 'Ridge', 'HGB', 'Oracle']
axs[1, 0].bar(x, caps, color=['0.3', '#8c564b', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '0.6'], edgecolor='k', lw=0.5); axs[1, 0].set_yscale('log'); axs[1, 0].set_xticks(x); axs[1, 0].set_xticklabels(names, rotation=35, ha='right')
for xi, cv in zip(x, caps): axs[1, 0].text(xi, cv * 1.12, f'{cv:g}', ha='center', fontsize=6.5)
axs[1, 0].set_ylabel('Minimum cell (mAh)'); axs[1, 0].set_ylim(4, 1100)
lo = [c * V_NOM / 1e6 * 61 * 1e3 for c in caps]; hi = [c * V_NOM / 1e6 * 106 * 1e3 for c in caps]
axs[1, 1].bar(x, hi, color='lightgray', edgecolor='k', lw=0.5, label='106 kg/kWh'); axs[1, 1].bar(x, lo, color='#2ca02c', edgecolor='k', lw=0.5, label='61 kg/kWh'); axs[1, 1].set_yscale('log'); axs[1, 1].set_xticks(x); axs[1, 1].set_xticklabels(names, rotation=35, ha='right'); axs[1, 1].set_ylabel('Cell-production CO$_2$e (g)'); axs[1, 1].legend(loc='upper right')
for a in axs[1]: a.grid(alpha=0.25, lw=0.4, axis='y', which='both')
for a, s in zip(axs.flat, 'abcd'): lab(a, s)
fig.tight_layout(pad=0.5); fig.savefig('figures/f6.png'); plt.close(fig)

# ---- F7 where does forecasting help: heatmaps + bias
CP = [5, 10, 20, 30, 50, 100]; EU = [0.5, 0.75, 1.0, 1.5, 2.0]
inj = R2['inj']
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.45), gridspec_kw={'width_ratios': [0.9, 1.05, 1.05]})
for c, mk in ((5, 'o'), (10, 's')):
    axs[0].plot([-30, -15, 0, 15, 30], [inj[(c, 'bias', v)][1] for v in (-0.3, -0.15, 0, 0.15, 0.3)], marker=mk, ms=3, lw=0.9, label=f'{c} mAh')
axs[0].set_xlabel('Bias of oracle forecast (%)'); axs[0].set_ylabel('Mean uplinks per hour'); axs[0].legend(); axs[0].grid(alpha=0.25, lw=0.4)
dy = np.array([[G[('eu', c, v, 'hgb')][1] - G[('eu', c, v, 'reactive')][1] for v in EU] for c in CP])
da = np.array([[100 * (G[('eu', c, v, 'hgb')][0] - G[('eu', c, v, 'reactive')][0]) for v in EU] for c in CP])
for ax_, A, ttl, lim in ((axs[1], dy, 'Δ uplinks/h (HGB − reactive)', 5), (axs[2], da, 'Δ availability, pp (HGB − reactive)', 10)):
    im = ax_.imshow(A, origin='lower', cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto'); ax_.set_xticks(range(5)); ax_.set_xticklabels(EU); ax_.set_yticks(range(len(CP))); ax_.set_yticklabels(CP)
    ax_.set_xlabel('Energy-per-uplink scale'); ax_.set_title(ttl, fontsize=6.8)
    for i in range(len(CP)):
        for j in range(5): ax_.text(j, i, f'{A[i, j]:.1f}' if abs(A[i, j]) >= 0.05 else '0', ha='center', va='center', fontsize=5.6, color='k')
    fig.colorbar(im, ax=ax_, fraction=0.05, pad=0.02).ax.tick_params(labelsize=5.5)
axs[1].set_ylabel('Capacity (mAh)')
for a, s in zip(axs, 'abc'): lab(a, s)
fig.tight_layout(pad=0.4); fig.savefig('figures/f7.png'); plt.close(fig)

# ---- F8 stress scenario
sc = R2['scB']; cb = [5, 10, 20, 50, 100, 200]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))
for cn in ('fixed', 'reactive', 'persist', 'hgb', 'oracle'):
    c, mk, ls = ST[cn]
    axs[0].plot(cb, [max(100 * (1 - sc[(cc, cn)][0]), 0.01) for cc in cb], color=c, marker=mk, ms=3, lw=0.9, ls=ls, label=NM[cn]); axs[1].plot(cb, [sc[(cc, cn)][2] for cc in cb], color=c, marker=mk, ms=3, lw=0.9, ls=ls)
axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_ylabel('Unavailability (%), floor 0.01'); axs[0].legend(loc='lower left'); axs[1].set_xscale('log'); axs[1].set_ylabel('Mean uplinks per hour')
for a, s in zip(axs, 'ab'): a.set_xlabel('Battery capacity (mAh)'); a.grid(alpha=0.25, lw=0.4, which='both'); lab(a, s)
fig.tight_layout(pad=0.4); fig.savefig('figures/f8.png'); plt.close(fig)
print('ok')
