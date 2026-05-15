"""Compile optimal per-SN strategy from experiment results."""
data = {
    'SN13': (0.0909, 0.0801, 92.14, 62.68, 20.14, 16.26),
    'SN14': (0.0876, 0.0865, 41.37, 37.57, 16.70, 16.20),
    'SN15': (0.0848, 0.0712, 98.28, 81.79, 14.24, 12.46),
    'SN16': (0.0992, 0.0988, 44.25, 45.14, 16.74, 16.29),
    'SN17': (0.0908, 0.0902, 41.44, 38.36, 16.80, 16.22),
    'SN18': (0.0998, 0.0984, 50.87, 49.13, 16.71, 16.25),
    'SN19': (0.0982, 0.0970, 45.81, 44.52, 16.64, 16.21),
    'SN20': (0.0871, 0.0866, 42.09, 39.37, 16.82, 16.27),
    'SN21': (0.1033, 0.1017, 51.88, 50.06, 16.77, 16.24),
    'SN22': (0.0904, 0.0899, 34.28, 34.12, 16.71, 16.24),
    'SN23': (0.1447, 0.1336, 144.36, 127.21, 17.03, 16.37),
    'SN24': (0.0903, 0.0901, 36.76, 37.32, 16.65, 16.20),
}

THRESH = 0.5

hdr = f" {'SN':>5s}  {'Thr Base':>9s}  {'Thr FT':>9s}  {'dThr':>7s}  {'dMFR':>7s}  {'dIsp':>7s}  {'Strategy':>22s}  {'Eff dThr':>9s}"
print(hdr)
print('-' * 95)

adapter_sns = []
global_sns = []
all_eff = []

for sn, (tb, tf, mb, mf, ib_global, ib_ft) in sorted(data.items()):
    dt = (tb - tf) / tb * 100
    dm = (mb - mf) / mb * 100
    di = (ib_global - ib_ft) / ib_global * 100

    if dt < THRESH:
        strategy = 'Global model (skip)'
        eff = 0.0
        global_sns.append(sn)
    else:
        strategy = 'Adapter + Isp Loss'
        eff = dt
        adapter_sns.append(sn)
    all_eff.append(eff)

    sn_num = int(sn.replace('SN', ''))
    print(f' SN{sn_num:2d}  {tb:7.4f} N  {tf:7.4f} N  {dt:+6.1f}%  {dm:+6.1f}%  {di:+6.1f}%  {strategy:22s}  {eff:+9.1f}%')

print('-' * 95)
avg = sum(all_eff) / len(all_eff)
print(f'Adapter SNs ({len(adapter_sns)}): {adapter_sns}')
print(f'Global SNs  ({len(global_sns)}): {global_sns}')
print(f'Overall avg effective thrust improvement: {avg:+.1f}%')
print(f'Catastrophic forgetting: ZERO (base model frozen, adapters per-SN)')
print()
print('Config: adapter(64/32), triple_loss(w_isp=0.05), lr=2e-4, epochs=10')
