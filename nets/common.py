import torch
from torch import nn

class MLP(nn.Module):
	def __init__(self, hiddens, activation=torch.relu) -> None:
		super().__init__()
		self.activation = activation
		self.linears = nn.ModuleList()
		for i in range(1, len(hiddens)):
			self.linears.append(nn.Linear(hiddens[i-1], hiddens[i]))
		
	def forward(self, x):
		zs = [x]
		for i, linear in enumerate(self.linears):
			z = linear(zs[i])
			if i != len(self.linears) - 1:
				a = self.activation(z)
				zs.append(a)
			else:
				zs.append(z)
		
		return zs[-1]
		
		
