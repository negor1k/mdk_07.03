def mean(lst):
    total = 0
    for v in lst:
        total += v
    return total / len(lst)

def minmax(lst):
    lo, hi = lst[0], lst[0]
    for v in lst:
        if v < lo: lo = v
        if v > hi: hi = v
    return lo,hi

print('Sr znach' ,mean([3, 1, 4]), ',' , 'min max',minmax([3, 1, 4]))

with open('ml-start/data/titanic.csv', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]
print('Zagolovki',header)
print(len(rows), 'strok')
print('1st stroka' , rows[0])

cols = {h: [] for h in header}
for r in rows:
    for h,v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)
print(cols['age'][:10])

for h, vals in cols.items():
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')