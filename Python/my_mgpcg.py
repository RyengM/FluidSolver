import taichi as ti
import numpy as np
import math
ti.init(arch=ti.gpu)

@ti.data_oriented
class MultigridPCGPoissonSolver:
    def __init__(self, marker, nx, ny, n_mg_levels = 4):
        shape = (nx, ny)
        self.nx, self.ny = shape
        print(f'nx, ny = {nx}, {ny}')

        self.dim = 2
        self.max_iters = 300
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 30
        self.use_multigrid = True

        def _res(l): return (nx // (2**l), ny // (2**l))

        self.r = [ti.field(float, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(float, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # M^-1 r
        self.d = [ti.field(float, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # temp
        self.f = [marker] + [ti.field(float, shape=_res(_))
                             for _ in range(self.n_mg_levels - 1)]  # marker
        self.L = [ti.Vector.field(6, float, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # -L operator

        self.x = ti.field(float, shape=shape)  # solution
        self.p = ti.field(float, shape=shape)  # conjugate gradient
        self.Ap = ti.field(float, shape=shape)  # matrix-vector product
        self.alpha = ti.field(float, shape=())  # step size
        self.beta = ti.field(float, shape=())  # step size
        self.sum = ti.field(float, shape=())  # storage for reductions

    @ti.func
    def is_fluid(self, f, i, j, nx, ny):
        return i >= 0 and i < nx and j >= 0 and j < ny and 1 == f[i, j]

    @ti.func
    def is_solid(self, f, i, j, nx, ny):
        return i < 0 or i >= nx or j < 0 or j >= ny or 2 == f[i, j]

    @ti.func
    def is_air(self, f, i, j, nx, ny):
        return i >= 0 and i < nx and j >= 0 and j < ny and 0 == f[i, j]

    @ti.func
    def neighbor_sum(self, L, x, f, i, j, nx, ny):
        ret = x[(i - 1 + nx) % nx, j] * L[i, j][2]
        ret += x[(i + 1 + nx) % nx, j] * L[i, j][3]
        ret += x[i, (j - 1 + ny) % ny] * L[i, j][4]
        ret += x[i, (j + 1 + ny) % ny] * L[i, j][5]
        return ret

    # -L matrix : 0-diagonal, 1-diagonal inverse, 2...-off diagonals
    @ti.kernel
    def init_L(self, l: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.L[l]:
            if 1 == self.f[l][i, j]:
                s = 4.0
                s -= float(self.is_solid(self.f[l], i - 1, j, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i + 1, j, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i, j - 1, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i, j + 1, _nx, _ny))
                self.L[l][i, j][0] = s
                self.L[l][i, j][1] = 1.0 / s
            self.L[l][i, j][2] = float(
                self.is_fluid(self.f[l], i - 1, j, _nx, _ny))
            self.L[l][i, j][3] = float(
                self.is_fluid(self.f[l], i + 1, j, _nx, _ny))
            self.L[l][i, j][4] = float(
                self.is_fluid(self.f[l], i, j - 1, _nx, _ny))
            self.L[l][i, j][5] = float(
                self.is_fluid(self.f[l], i, j + 1, _nx, _ny))

    def solve(self, x, rhs):
        tol = 1e-12

        self.r[0].copy_from(rhs)
        self.x.fill(0.0)

        self.Ap.fill(0.0)
        self.p.fill(0.0)

        for l in range(1, self.n_mg_levels):
            self.downsample_f(self.f[l - 1], self.f[l],
                              self.nx // (2**l), self.ny // (2**l))
        for l in range(self.n_mg_levels):
            self.L[l].fill(0.0)
            self.init_L(l)

        self.sum[None] = 0.0
        self.reduction(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        print(f"init rtr = {initial_rTr}")

        if initial_rTr < tol:
            print(f"converged: init rtr = {initial_rTr}")
        else:
            # r = b - Ax = b    since x = 0
            # p = r = r + 0 p
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            self.update_p()

            self.sum[None] = 0.0
            self.reduction(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iter = 0
            for i in range(self.max_iters):
                # alpha = rTr / pTAp
                self.apply_L(0, self.p, self.Ap)

                self.sum[None] = 0.0
                self.reduction(self.p, self.Ap)
                pAp = self.sum[None]

                self.alpha[None] = old_zTr / pAp

                # x = x + alpha p
                # r = r - alpha Ap
                self.update_x_and_r()

                # check for convergence
                self.sum[None] = 0.0
                self.reduction(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < initial_rTr * tol:
                    break

                # z = M^-1 r
                if self.use_multigrid:
                    self.apply_preconditioner()
                else:
                    self.z[0].copy_from(self.r[0])

                # beta = new_rTr / old_rTr
                self.sum[None] = 0.0
                self.reduction(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                self.beta[None] = new_zTr / old_zTr

                # p = z + beta p
                self.update_p()
                old_zTr = new_zTr

                iter = i
            print(f'converged to {rTr} in {iter} iters')

        x.copy_from(self.x)

    @ti.kernel
    def apply_L(self, l: ti.template(), x: ti.template(), Ax: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in Ax:
            if 1 == self.f[l][i, j]:
                r = x[i, j] * self.L[l][i, j][0]
                r -= self.neighbor_sum(self.L[l], x,
                                       self.f[l], i, j, _nx, _ny)
                Ax[i, j] = r

    @ti.kernel
    def reduction(self, p: ti.template(), q: ti.template()):
        for I in ti.grouped(p):
            if 1 == self.f[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x_and_r(self):
        a = float(self.alpha[None])
        for I in ti.grouped(self.p):
            if 1 == self.f[0][I]:
                self.x[I] += a * self.p[I]
                self.r[0][I] -= a * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if 1 == self.f[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    # ------------------ multigrid ---------------
    @ti.kernel
    def downsample_f(self, f_fine: ti.template(), f_coarse: ti.template(),
                     nx: ti.template(), ny: ti.template()):
        for i, j in f_coarse:
            i2 = i * 2
            j2 = j * 2

            if 0 == f_fine[i2, j2] or 0 == f_fine[i2 + 1, j2] or \
               0 == f_fine[i2, j2 + 1] or 0 == f_fine[i2 + 1, j2 + 1]:
                f_coarse[i, j] = 0
            else:
                if 1 == f_fine[i2, j2] or 1 == f_fine[i2 + 1, j2] or \
                   1 == f_fine[i2 + 1, j2] or 1 == f_fine[i2 + 1, j2 + 1]:
                    f_coarse[i, j] = 1
                else:
                    f_coarse[i, j] = 2

    @ti.kernel
    def restrict(self, l: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.r[l]:
            if 1 == self.f[l][i, j]:
                Az = self.L[l][i, j][0] * self.z[l][i, j]
                Az -= self.neighbor_sum(self.L[l],
                                        self.z[l], self.f[l], i, j, _nx, _ny)
                res = self.r[l][i, j] - Az
                self.r[l + 1][i // 2, j // 2] += res

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    # Gause-Seidel
    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.r[l]:
            if 1 == self.f[l][i, j] and (i + j) & 1 == phase:
                self.z[l][i, j] = (self.r[l][i, j]
                                   + self.neighbor_sum(self.L[l], self.z[l], self.f[l], i, j, _nx, _ny)
                                   ) * self.L[l][i, j][1]

    def apply_preconditioner(self):

        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.d[l].fill(0.0)
            self.restrict(l)

        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

nx = 128
marker = ti.field(int, shape=(nx, nx))
class MGPCG_Example(MultigridPCGPoissonSolver):
    def __init__(self):
        super().__init__(marker, 128, 128, n_mg_levels=4)
        self.rhs = ti.field(float, shape=(self.nx, self.nx))

    @ti.kernel
    def init(self):
        for i, j in self.rhs:
            self.rhs[i, j] = 5.0 * ti.cos(5 * 3.1415926 * (i * self.nx + j) / self.nx)
            self.f[0][i, j] = 1
        for i in range(self.nx):
            self.f[0][self.nx - 1, i] = 0
            self.f[0][i, 0] = 0
            self.f[0][i, self.nx - 1] = 0
            self.f[0][0, i] = 0

    def run(self):
        self.init()
        self.solve(self.x, self.rhs)

if __name__ == '__main__':
    solver = MGPCG_Example()
    solver.run()
