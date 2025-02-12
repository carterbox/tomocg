"""Module for tomography."""

import cupy as cp
import numpy as np
from tomocg.radonusfft import radonusfft


class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of slice partitions to process together
        simultaneously.
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data)
        self.nz = nz
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u.data.ptr)        
        return res

    def adj_tomo(self, data):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data.data.ptr)
        return res

    def line_search(self, minf, gamma, Ru, Rd):
        """Line search for the step sizes gamma"""
        while(minf(Ru)-minf(Ru+gamma*Rd) < 0):
            gamma *= 0.5
        return gamma
    
    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            data_gpu = cp.array(data[:, ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    # Conjugate gradients tomography (for 1 slice partition)
    def cg_tomo(self, xi0, u, titer):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u)
            grad = self.adj_tomo(Ru-xi0) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (np.mod(i, 1) == -1):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u
  
    # Conjugate gradients tomography (by slices partitions)
    def cg_tomo_batch(self, xi0, init, titer):
        """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
        u = init.copy()

        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            u_gpu = cp.array(u[ids])
            xi0_gpu = cp.array(xi0[:, ids])
            # reconstruct
            u_gpu = self.cg_tomo(xi0_gpu, u_gpu, titer)
            u[ids] = u_gpu.get()
        return u

    # Conjugate gradients tomography (for all slices)
    def cg_tomo_batch2(self, xi0, u, titer):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo_batch(u)
            grad = self.adj_tomo_batch(Ru-xi0) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo_batch(d)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (np.mod(i, 1) == -1):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u
