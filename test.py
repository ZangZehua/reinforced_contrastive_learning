import torch
from torch.distributions import MultivariateNormal

action_list = torch.Tensor([])
a = torch.Tensor([1, 2, 3])
b = torch.Tensor([[0.36, 0, 0],
                  [0, 0.36, 0],
                  [0, 0, 0.36]])
c = torch.clone(b)
c -= 1
print(c, b)
# dist = MultivariateNormal(a, b)
# for _ in range(1000):
#     # print(dist)
#     action = dist.sample().unsqueeze(0)
#     action_list = torch.cat((action_list, action), dim=0)
#     # print(action)
# print(torch.mean(action_list, dim=0))
