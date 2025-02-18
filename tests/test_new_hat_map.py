import numpy as np
import torch
import time

def old_hat_map(s):
    """
    Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
    In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
    """
    device = s.device
    if len(s.shape) > 1:  # Vectorized implementation
        s = s.cpu()
        return torch.from_numpy(np.array([[ np.zeros(s.shape[0]), -s[:,2],  s[:,1]],
                                          [ s[:,2],     np.zeros(s.shape[0]), -s[:,0]],
                                          [-s[:,1],  s[:,0],     np.zeros(s.shape[0])]])).to(device)


def new_hat_map(s):
    s = s.unsqueeze(-1)
    device = s.device
    hat = torch.cat([torch.zeros(s.shape[0], 1, device=device), -s[:, 2], s[:,1],
                     s[:,2], torch.zeros(s.shape[0], 1, device=device), -s[:,0],
                     -s[:,1], s[:,0], torch.zeros(s.shape[0], 1, device=device)], dim=0).view(3, 3, s.shape[0]).float()
    return hat

N = int(4e5)
random_vectors = torch.rand(N, 3).to(torch.device("cuda"))

t = time.time()
gt_hat_map = old_hat_map(random_vectors)
print(f"time to compute old hat map: {time.time() - t}")
t = time.time()
test_hat_map = new_hat_map(random_vectors)
print(f"time to compute new hat map: {time.time() - t}")

assert torch.equal(test_hat_map, gt_hat_map)