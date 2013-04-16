import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import sparsesvd
import sys

def newton_step(springs, i_pos, j_pos):
    # springs is n columns of idx0, idx1, k, rest_length
    N = i_pos.size
    idx0 = springs[0, :].astype(int)
    idx1 = springs[1, :].astype(int)
    ks = springs[2, :]
    rl = springs[3, :]
    di = i_pos[idx1] - i_pos[idx0]
    dj = j_pos[idx1] - j_pos[idx0]
    disq = di ** 2
    djsq = dj ** 2
    distssq = disq + djsq
    dists = np.sqrt(distssq)
    distscube = distssq * dists
    dists[dists == 0] = 1
    distscube[distscube == 0] = 1
    fscale = ks * (1 - rl / dists)
    sfi = fscale * di
    sfj = fscale * dj
    # distribute to points
    fi1 = np.bincount(idx0, sfi, N)
    fj1 = np.bincount(idx0, sfj, N)
    fi2 = np.bincount(idx1, sfi, N)
    fj2 = np.bincount(idx1, sfj, N)
    Fi = fi1 - fi2;
    Fj = fj1 - fj2;

    # Jacobian
    J00 = ks * (1 - rl * djsq / distscube)
    J01 = ks * (- (rl * (- di * dj) / distscube))
    J10 = J01
    J11 = ks * (1 - rl * disq / distscube)
    Jii = sparse.coo_matrix((J00, (idx0, idx1)), (N, N)).tocsc() + \
        sparse.coo_matrix((J00, (idx1, idx0)), (N, N)).tocsc() + \
        sparse.coo_matrix((-J00, (idx0, idx0)), (N, N)).tocsc() + \
        sparse.coo_matrix((-J00, (idx1, idx1)), (N, N)).tocsc()
    Jij = sparse.coo_matrix((J01, (idx0, idx1)), (N, N)).tocsc() + \
        sparse.coo_matrix((J01, (idx1, idx0)), (N, N)).tocsc() + \
        sparse.coo_matrix((-J01, (idx0, idx0)), (N, N)).tocsc() + \
        sparse.coo_matrix((-J01, (idx1, idx1)), (N, N)).tocsc()
    Jji = Jij
    Jjj = sparse.coo_matrix((J11, (idx0, idx1)), (N, N)).tocsc() + \
        sparse.coo_matrix((J11, (idx1, idx0)), (N, N)).tocsc() +\
        sparse.coo_matrix((-J11, (idx0, idx0)), (N, N)).tocsc() + \
        sparse.coo_matrix((-J11, (idx1, idx1)), (N, N)).tocsc()

    F = np.vstack((Fi.reshape(-1, 1), Fj.reshape(-1, 1)))
    print "Force max", abs(F).max()
    J = sparse.vstack((sparse.hstack((Jii, Jij)),
                       sparse.hstack((Jji, Jjj)))).tocsc()
    # dij = linalg.spsolve(J, -F)
    P = sparse.diags(1.0 / J[range(J.shape[0]), range(J.shape[1])].A, [0])
    print "CG"
    def foo(x):
        print ".",
        sys.stdout.flush()
    dp, info = linalg.cg(J, -F, P * -F, tol=1e-2, M=P, callback=foo)
    print ""
    assert info == 0
    dp *= 0.5
    return i_pos + dp[:i_pos.size], j_pos + dp[i_pos.size:], abs(F).max()

