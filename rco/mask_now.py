import torch, sys
P="/work/glm53"
L, E = 42, 288
R = torch.load(f"{P}/calib53/ref_top20.pt", map_location="cpu", weights_only=False)
freq = R["freq"].sum(0).float().reshape(L, E)
del R
share = freq / freq.sum(1, keepdim=True)
for run, keep, nu in [("k72",72,0),("k96",96,0),("k144",144,0),("k72nu",72,1),("k96nu",96,1),("k144nu",144,1)]:
    c = torch.load(f"{P}/rco_runs/{run}/ckpt.pt", map_location="cpu", weights_only=False)
    p = torch.softmax(c["alpha"], 1)[:, 0].reshape(L, E)  # prob of keep
    if nu:
        k = torch.zeros_like(p, dtype=torch.bool).flatten(); k[p.flatten().topk(keep*L).indices] = True; k = k.reshape(L, E)
    else:
        k = torch.zeros_like(p, dtype=torch.bool)
        for l in range(L): k[l, p[l].topk(keep).indices] = True
    fk = torch.zeros_like(k)
    for l in range(L): fk[l, freq[l].topk(keep).indices] = True
    sure_keep = (p > 0.9).sum().item(); sure_cut = (p < 0.1).sum().item()
    ov = ((k & fk).sum(1).float() / keep)
    cov = (share * k).sum(1)  # fraction of routing traffic going to kept experts
    fcov = (share * fk).sum(1)
    per = k.sum(1)
    print(f"{run:7s} step {c['step']}: surely kept {sure_keep}, surely pruned {sure_cut}, undecided {L*E-sure_keep-sure_cut}; "
          f"overlap with frequency {ov.mean()*100:.0f}% (min layer {ov.min()*100:.0f}%); "
          f"traffic covered RCO {cov.mean()*100:.0f}% vs frequency {fcov.mean()*100:.0f}%"
          + (f"; experts/layer min {per.min().item()} max {per.max().item()}, first 6 layers {per[:6].tolist()}, last 6 {per[-6:].tolist()}" if nu else ""))
