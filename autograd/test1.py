import torch
import random

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# func
def obj_func(x):
    return 3*x*x + 2*x + 5

data = []
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
c = torch.randn(1, requires_grad=True, dtype=torch.float)
for _ in range(1000):
    x = random.random()
    y = obj_func(x)
    data.append((torch.tensor(x, requires_grad=False), torch.tensor(y, requires_grad=False)))
print([a,b,c])

optimizer = torch.optim.Adam([a,b,c], lr=1e-4)
for _ in range(500):
    for item in data:
        loss = torch.abs(a*item[0]*item[0] + b*item[0] + c - item[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss.item())
    print([a.item(),b.item(),c.item()])
