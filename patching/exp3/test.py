import torch

errors = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/true_label_error.pt')
timesteps = timesteps = [11,31,51,71,91,111,131,151,171,191,211,231,251,271,291,311,331,351,371,391,411,431,451,471,491,511,531,551,571,591,611,631]

print(timesteps)
print(len(timesteps))