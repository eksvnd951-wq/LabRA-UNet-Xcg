
import torch, torch.nn as nn

class CB(nn.Module):
    def __init__(self,i,o): super().__init__(); self.c=nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self,x): return self.c(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g); x1 = self.W_x(x); psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class AttentionUNet(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
        self.e1=CB(3,64); self.e2=CB(64,128); self.e3=CB(128,256); self.e4=CB(256,512); self.b=CB(512,1024); self.mp=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(1024,512,2,2); self.att4=AttentionBlock(512,512,256); self.d4=CB(1024,512)
        self.u3=nn.ConvTranspose2d(512,256,2,2); self.att3=AttentionBlock(256,256,128); self.d3=CB(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,2); self.att2=AttentionBlock(128,128,64); self.d2=CB(256,128)
        self.u1=nn.ConvTranspose2d(128,64,2,2); self.att1=AttentionBlock(64,64,32); self.d1=CB(128,64)
        self.f=nn.Conv2d(64,n_cls,1)
    def forward(self,x):
        x1=self.e1(x); x2=self.e2(self.mp(x1)); x3=self.e3(self.mp(x2)); x4=self.e4(self.mp(x3)); b=self.b(self.mp(x4))
        d4=self.d4(torch.cat((self.u4(b),x4),1)); d3=self.d3(torch.cat((self.u3(d4),x3),1))
        d2=self.d2(torch.cat((self.u2(d3),x2),1)); d1=self.d1(torch.cat((self.u1(d2),x1),1))
        return self.f(d1)
