"""Frequency-baseline masks (the same rule as prune.frequency_uniform_mask): per sparse layer keep the `keep`
experts the router chose most often on the 512 search windows. Saved as bool (n_sparse, n_exp), True = pruned."""
import os
import torch
P = "/work/glm53"
R = torch.load(f"{P}/calib53/ref_top20.pt", map_location="cpu", weights_only=False)
freq = R["freq"].sum(0).float()
L, E = freq.shape
os.makedirs(f"{P}/rco_runs/freq", exist_ok=True)
for keep in (72, 96, 144):
    m = torch.zeros(L, E, dtype=torch.bool)
    order = freq.argsort(dim=1)
    for l in range(L):
        m[l, order[l, :E - keep]] = True
    assert ((~m).sum(1) == keep).all()
    torch.save(m, f"{P}/rco_runs/freq/mask_k{keep}.pt")
    print(keep, "kept/layer", keep, "pruned total", m.sum().item())
