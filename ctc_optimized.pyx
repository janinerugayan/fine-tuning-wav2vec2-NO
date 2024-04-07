from libc cimport math
import cython
import numpy as np
cimport numpy as np
np.seterr(divide='raise',invalid='raise')

ctypedef np.float64_t f_t

# Turn off bounds checking, negative indexing
@cython.boundscheck(False)
@cython.wraparound(False)
def forward_pass(double[::1,:] params,
                 int[::1] seq,
                 unsigned int blank=31):
    cdef unsigned int seqLen = seq.shape[0]
    cdef unsigned int L = 2*seqLen + 1
    cdef unsigned int T = params.shape[1]

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double)
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double)

    cdef unsigned int start, end
    cdef unsigned int t, s, l
    cdef double c, llForward, llBackward

    # Initialize alphas and forward pass
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    c = alphas[0,0] + alphas[1,0]
    alphas[0,0] = alphas[0,0] / c
    alphas[1,0] = alphas[1,0] / c
    llForward = math.log(c)
    for t in range(1,T):
        start = 2*(T-t)
        if L <= start:
            start = 0
        else:
            start = L-start
        end = min(2*t+2,L)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
            # same label twice
            elif s == 1 or seq[l] == seq[l-1]:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                                * params[seq[l],t]

        # normalize at current time (prevent underflow)
        c = 0.0
        for s in range(start,end):
            c += alphas[s,t]
        for s in range(start,end):
            alphas[s,t] = alphas[s,t] / c
        llForward += math.log(c)

    # Initialize betas and backwards pass
    betas[L-1,T-1] = params[blank,T-1]
    betas[L-2,T-1] = params[seq[seqLen-1],T-1]
    c = betas[L-1,T-1] + betas[L-2,T-1]
    betas[L-1,T-1] = betas[L-1,T-1] / c
    betas[L-2,T-1] = betas[L-2,T-1] / c
    llBackward = math.log(c)
    for t in range(T-1,0,-1):
        t = t-1
        start = 2*(T-t)
        if L <= start:
            start = 0
        else:
            start = L-start
        end = min(2*t+2,L)
        for s in range(end,0,-1):
            s = s-1
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s == L-1:
                    betas[s,t] = betas[s,t+1] * params[blank,t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
            # same label twice
            elif s == L-2 or seq[l] == seq[l+1]:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
            else:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                                * params[seq[l],t]

        c = 0.0
        for s in range(start,end):
            c += betas[s,t]
        for s in range(start,end):
            betas[s,t] = betas[s,t] / c
        llBackward += math.log(c)

    return -llForward, llBackward, alphas, betas