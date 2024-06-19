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

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double, order='F')

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

    return -llForward, llBackward, np.asarray(alphas), np.asarray(betas)


def forward_pass_with_ASD(double[::1,:] params,
                          int[::1] seq,
                          double[::1] cosdist_for_ctc,
                          double lambda_asd,
                          unsigned int blank=31):

    cdef unsigned int seqLen = seq.shape[0]
    cdef unsigned int L = 2*seqLen + 1
    cdef unsigned int T = params.shape[1]

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double, order='F')

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
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t] / (1 + cosdist_for_ctc[l])
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t] / (1 + cosdist_for_ctc[l])

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
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t] / (1 + cosdist_for_ctc[l])
            else:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t] / (1 + cosdist_for_ctc[l])

        c = 0.0
        for s in range(start,end):
            c += betas[s,t]
        for s in range(start,end):
            betas[s,t] = betas[s,t] / c
        llBackward += math.log(c)

    return -llForward, llBackward, np.asarray(alphas), np.asarray(betas)


def forward_pass_with_ASD_nbest(double[::1,:] params,
                                int[::1] seq,
                                int[::1] nbest_seq,
                                double lambda_asd,
                                unsigned int blank=31):

    cdef unsigned int seqLen = seq.shape[0]
    cdef unsigned int nbest_seqLen = nbest_seq.shape[0]
    cdef unsigned int L = 2*seqLen + 1
    cdef unsigned int nbest_L = 2*nbest_seqLen + 1
    cdef unsigned int T = params.shape[1]

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double, order='F')

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

    # alphas calculation with lambda weighted nbest
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
                    alphas[s,t] = alphas[s,t-1] * params[blank,t] * (1 - lambda_asd)
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t] * (1 - lambda_asd)
            # same label twice
            elif s == 1 or seq[l] == seq[l-1]:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t] * (1 - lambda_asd)
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t] * (1 - lambda_asd)

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

    # ORIGINAL betas calculation with lambda
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
                    betas[s,t] = betas[s,t+1] * params[blank,t] * (1 - lambda_asd)
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t] * (1 - lambda_asd)
            # same label twice
            elif s == L-2 or seq[l] == seq[l+1]:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t] * (1 - lambda_asd)
            else:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t] * (1 - lambda_asd)

        c = 0.0
        for s in range(start,end):
            c += betas[s,t]
        for s in range(start,end):
            betas[s,t] = betas[s,t] / c
        llBackward += math.log(c)

    return -llForward, llBackward, np.asarray(alphas), np.asarray(betas)


def backward_pass(double[::1,:] params,
                 int[::1] seq,
                 double[::1,:] alphas,
                 double[::1,:] betas,
                 unsigned int input_length,
                 unsigned int blank=31):

    cdef unsigned int seqLen = seq.shape[0]
    cdef unsigned int numphones = params.shape[0] # Number of labels
    cdef unsigned int L = 2*seqLen + 1
    cdef unsigned int T = params.shape[1]

    cdef double[::1,:] ab = np.empty((L,T), dtype=np.double, order='F')
    cdef np.ndarray[f_t, ndim=2] grad = np.zeros((numphones,T),
                            dtype=np.double, order='F')
    cdef double[::1,:] grad_v = grad
    cdef double[::1] absum = np.empty(T, dtype=np.double)

    cdef unsigned int t, s
    cdef double tmp

    # Compute gradient with respect to unnormalized input parameters
    for t in range(T):
        for s in range(L):
            ab[s,t] = alphas[s,t]*betas[s,t]
    for s in range(L):
        # blank
        if s%2 == 0:
            for t in range(T):
                grad_v[blank,t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/params[blank,t]
        else:
            for t in range(T):
                grad_v[seq[int((s-1)/2)],t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/(params[seq[int((s-1)/2)],t])

    for t in range(T):
        absum[t] = 0
        for s in range(L):
            absum[t] += ab[s,t]

    for t in range(T):
        for s in range(numphones):
            tmp = (params[s,t]*absum[t])
            if tmp > 0:
                grad_v[s,t] = params[s,t] - grad_v[s,t] / tmp
            else:
                grad_v[s,t] = params[s,t]

    # zero the gradients that are out of context
    for i in range(T):
        if i >= input_length:
            grad_v[:,i] = 0.0

    return grad


def backward_pass_with_ASD(double[::1,:] params,
                 int[::1] seq,
                 double[::1,:] alphas,
                 double[::1,:] betas,
                 double[::1] cosdist_for_ctc,
                 double lambda_asd,
                 unsigned int input_length,
                 unsigned int blank=31):

    cdef unsigned int seqLen = seq.shape[0]
    cdef unsigned int numphones = params.shape[0] # Number of labels
    cdef unsigned int L = 2*seqLen + 1
    cdef unsigned int T = params.shape[1]

    cdef double[::1,:] ab = np.empty((L,T), dtype=np.double, order='F')
    cdef np.ndarray[f_t, ndim=2] grad = np.zeros((numphones,T),
                            dtype=np.double, order='F')
    cdef double[::1,:] grad_v = grad
    cdef double[::1] absum = np.empty(T, dtype=np.double)

    cdef unsigned int t, s, l
    cdef double tmp

    # Compute gradient with respect to unnormalized input parameters
    for t in range(T):
        for s in range(L):
            ab[s,t] = alphas[s,t]*betas[s,t]
    for s in range(L):
        # blank
        if s%2 == 0:
            for t in range(T):
                grad_v[blank,t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/params[blank,t]
        else:
            for t in range(T):
                grad_v[seq[int((s-1)/2)],t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/(params[seq[int((s-1)/2)],t])

    for t in range(T):
        absum[t] = 0
        for s in range(L):
            absum[t] += ab[s,t]

    for t in range(T):
        for s in range(numphones):
            tmp = (params[s,t]*absum[t])
            if tmp > 0:
                grad_v[s,t] = params[s,t] - grad_v[s,t] / tmp
            else:
                grad_v[s,t] = params[s,t]

    # weighting the gradients according to cosdist
    for t in range(T):
        for l in range(seqLen):
            grad_v[seq[l],t] = grad_v[seq[l],t] * (1 + (cosdist_for_ctc[l] * lambda_asd))

    # zero the gradients that are out of context
    for i in range(T):
        if i >= input_length:
            grad_v[:,i] = 0.0

    return grad