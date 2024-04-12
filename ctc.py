import torch


def forward_pass(params, seq, device, blank=31):
    params = logits.transpose(1,0)
    seqLen = seq.shape[0]  # length of label sequence
    L = 2*seqLen + 1  # length of the label sequence with blanks
    T = params.shape[1]  # length of utterance (time)

    alphas = torch.zeros((L,T), device=device).double()
    betas = torch.zeros((L,T), device=device).double()

    # convert logits to log probs
    params = params - (torch.max(params, dim=0)[0])
    params = torch.exp(params)
    params = params / torch.sum(params, dim=0)
    # log probs calculated here not the same with log probs input to pytorch CTC function
    # but they both produce the same CTC loss

    # initialize alphas and forward pass
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    c = torch.sum(alphas[:,0])
    alphas[:,0] = alphas[:,0] / c
    llForward = torch.log(c)

    for t in range(1,T):
        start = max(0,L-2*(T-t))
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
                # alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
            else:
                # alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t]

        # normalize at current time (prevent underflow)
        c = torch.sum(alphas[start:end,t])
        alphas[start:end,t] = alphas[start:end,t] / c
        llForward = llForward + torch.log(c)

    # initialize betas and backwards pass
    betas[-1,-1] = params[blank,-1]
    betas[-2,-1] = params[seq[-1],-1]
    c = torch.sum(betas[:,-1])
    betas[:,-1] = betas[:,-1] / c
    llBackward = torch.log(c)

    for t in range(T-2,-1,-1):
        start = max(0,L-2*(T-t))
        end = min(2*t+2,L)
        for s in range(end-1,-1,-1):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s == L-1:
                    betas[s,t] = betas[s,t+1] * params[blank,t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
            # same label twice
            elif s == L-2 or seq[l] == seq[l+1]:
                # betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
            else:
                # betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t] * (1 + cosdist_for_ctc[l])
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t]

        # normalize at current time
        c = torch.sum(betas[start:end,t])
        betas[start:end,t] = betas[start:end,t] / c
        llBackward = llBackward + torch.log(c)

    return -llForward, llBackward, alphas, betas, params


def backward_pass(params, seq, alphas, betas, device, blank=31):
    seqLen = seq.shape[0]  # length of label sequence
    L = 2*seqLen + 1  # length of the label sequence with blanks
    T = params.shape[1]  # length of utterance (time)

    grad = torch.zeros(params.shape, device=device)
    ab = alphas*betas

    for s in range(L):
        # blank
        if s%2 == 0:
            for t in range(T):
                grad[blank,t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/params[blank,t]
        else:
            for t in range(T):
                k = int((s-1)/2)
                grad[seq[k],t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/(params[seq[k],t])

    absum = torch.sum(ab, axis=0)

    grad = params - grad / (params * absum)

    # zero the gradients that are out of context
    print("grad shape:", grad.shape, "input length", input_length)
    for i in range(grad.shape[1]):
        if i >= input_length:
            grad[:,i] = 0.0

    return grad