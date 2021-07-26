import torch
import torch.nn as nn

output = torch.Tensor([[0.8982, 0.805, 0.6393, 0.9983]])
target = torch.LongTensor([1])

print(output.shape)
print(target.shape)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(loss) # tensor(2.1438)