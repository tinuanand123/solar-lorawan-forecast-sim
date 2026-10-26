import sys, pickle, time
sys.path.insert(0, '.')
from lib import *
lib = load(); M = pickle.load(open('models.pkl', 'rb'))
TEST = list(range(200, 210)); wte = [gen(2, s) for s in TEST]
FC = [forecasts(w, M)[0] for w in wte]
CAPS = [5, 10, 20, 30, 50, 100]; EUS = [0.5, 0.75, 1.0, 1.5, 2.0]; AREAS = [0.5, 0.75, 1.0, 1.5, 2.0]
G = {}
t = time.time()
for kind, vals in (('eu', EUS), ('area', AREAS)):
    for c in CAPS:
        for v in vals:
            for cn in ('reactive', 'hgb', 'oracle'):
                rows = []
                for w, out in zip(wte, FC):
                    kw = {'eu_scale': v} if kind == 'eu' else {'hscale': v}
                    o, _ = sim(lib, w, None if cn == 'reactive' else out[cn][1], c, 1 if cn == 'reactive' else 2, **kw)
                    rows.append(o[:2])
                rows = np.array(rows)
                G[(kind, c, v, cn)] = (rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 1].std(ddof=1))
print('grid', time.time() - t)
# per-realisation mean +- sd for the baseline sweep (for error bands)
R = pickle.load(open('res1.pkl', 'rb')); res = R['res']
pickle.dump(G, open('res3.pkl', 'wb'))
for c in CAPS:
    print(c, [round(G[('eu', c, v, 'hgb')][1] - G[('eu', c, v, 'reactive')][1], 2) for v in EUS], [round(100 * (G[('eu', c, v, 'reactive')][0] - G[('eu', c, v, 'hgb')][0]), 3) for v in EUS])
