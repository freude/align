import h5py

def lerp(a, b, t):
    return (b - a) * t + a

class Warp(object):
    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, 'r')
        xk, yk = f.keys()
        if xk.endswith('y'):
            xk, yk = yk, xk
        self.xwarp = f[xk]
        self.ywarp = f[yk]
        self.shape = self.xwarp.shape

    def __call__(self, x, y):
        x *= (self.shape[0] - 1)
        y *= (self.shape[1] - 1)
        ix = int(x)
        iy = int(y)
        tx = 1.0 - (x - ix)
        ty = 1.0 - (y - iy)
        if ix < self.xwarp.shape[0] - 1:
            xsl = slice(ix, ix+2)
        else:
            xsl = slice(ix, ix+1)
        if iy < self.ywarp.shape[1] - 1:
            ysl = slice(iy, iy + 2)
        else:
            ysl = slice(iy, iy + 1)
        xsq = self.xwarp[xsl, ysl]
        ysq = self.ywarp[xsl, ysl]
        xsq /= (self.shape[0] - 1)
        ysq /= (self.shape[1] - 1)
        return (lerp(lerp(xsq[0, 0], xsq[-1, 0], tx),
                     lerp(xsq[0, -1], xsq[-1, -1], tx),
                     ty),
                lerp(lerp(ysq[0, 0], ysq[-1, 0], tx),
                     lerp(ysq[0, -1], ysq[-1, -1], tx),
                     ty))
