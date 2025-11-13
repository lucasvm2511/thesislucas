# NasSearchSpace

Repo containing different search spaces for NAS:
- Once-For-All (OFA) MobilenetV3
- OFA Resnet50
- NASBench201
- DARTS

Pretrained weights of OFA can now be downloaded from torch!! Place the weights file in ofa/supernets.
```
import torch
super_net_name = "ofa_supernet_mbv3_w10" 
# other options: 
#    ofa_supernet_resnet50 / 
#    ofa_supernet_mbv3_w12 / 
#    ofa_supernet_proxyless

super_net = torch.hub.load('mit-han-lab/once-for-all', super_net_name, pretrained=True).eval()
```

# References

[1] ONCE-FOR-ALL: TRAIN ONE NETWORK AND SPECIALIZE IT FOR EFFICIENT DEPLOYMENT, ICLR 2019 (https://openreview.net/pdf?id=HylxE1HKwS)

[2]: NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020 (https://openreview.net/forum?id=HJxyZkBKDr)

[3] DARTS: Differentiable Architecture Search (https://arxiv.org/abs/1806.09055)
