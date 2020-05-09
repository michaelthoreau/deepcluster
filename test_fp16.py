import time

import torch
from torch.autograd import Variable
import torchvision.models as models

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


print(torch.cuda.get_device_capability(0)[0])



net = models.resnet152().cuda()
inp = torch.randn(32, 3, 224, 224).cuda()

for i in range(5):
    net.zero_grad()
    out = net.forward(Variable(inp, requires_grad=True))
    loss = out.sum()
    loss.backward()

torch.cuda.synchronize()
start=time.time()
for i in range(100):
    net.zero_grad()
    out = net.forward(Variable(inp, requires_grad=True))
    loss = out.sum()
    loss.backward()
torch.cuda.synchronize()
end=time.time()

print("FP32 Iterations per second: ", 100/(end-start))

net = models.resnet152().cuda().half()
inp = torch.randn(32, 3, 224, 224).cuda().half()

torch.cuda.synchronize()
start=time.time()
for i in range(100):
    net.zero_grad()
    out = net.forward(Variable(inp, requires_grad=True))
    loss = out.sum()
    loss.backward()
torch.cuda.synchronize()
end=time.time()

print("FP16 Iterations per second: ", 100/(end-start))