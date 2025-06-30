import torch
print(torch.__version__)
print(torch.cuda.is_available())

position = torch.arange(0, 5, dtype=torch.float)

print(position)

position = torch.arange(0, 5, dtype=torch.float).unsqueeze(0)
print(position)

position = torch.arange(0, 5, dtype=torch.float).unsqueeze(1)

print( position)