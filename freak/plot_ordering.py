import pylab
import sys
import re
import numpy as np
import pylab

idxre = re.compile('.*Ordered/Order([0-9]+)K.*Ordered/Order([0-9]+)K')

vals = []
for l in sys.stdin:
    if l.startswith('PAIR'):
        m = idxre.search(l)
        idx0 = int(m.group(1))
        idx1 = int(m.group(2))
    if l.startswith('MAD FORWARD'):
        v = float(l.split(' ')[-1])
        vals.append((idx0, idx1, v))
    if l.startswith('MAD REVERSE'):
        v = float(l.split(' ')[-1])
        vals.append((idx1, idx0, v))
    print "foo"

imax = max(v[0] for v in vals)
jmax = max(v[1] for v in vals)
vmax = max(v[2] for v in vals)
V = vmax * np.ones((imax + 1, jmax + 1))
for k in range(max(imax, jmax)):
    V[k, k] = 0
for i, j, v in vals:
    V[i, j] = v
pylab.matshow(V)
pylab.colorbar()
pylab.show()
