# %%
import torch
import math
class Camera:
    def __init__(self,fov_x=torch.pi/2,fov_y=torch.pi/2,resolution=(112,112)) -> None:
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.res_x = resolution[0]
        self.res_y = resolution[1]

        self.cam = torch.zeros(self.res_x,self.res_y,3)
        X = self.fov_x/2*(torch.arange(self.res_x)/(self.res_x-1)-0.5)
        Y = self.fov_y/2*(torch.arange(self.res_y)/(self.res_y-1)-0.5)
        X = X.unsqueeze(1)
        Y = -Y.unsqueeze(0)

        self.cam[:,:,0] = -torch.sin(X)*torch.cos(Y)
        self.cam[:,:,1] = torch.sin(Y)
        self.cam[:,:,2] = torch.cos(X)*torch.cos(Y)

    def __call__(self):
        return self.cam


class Camera_v2:
    def __init__(self,fx=1931.371337890625,fy=1931.371337890625,width=1600,height=1600) -> None:
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height
        self.fov_x = math.atan(self.height/self.fx)
        self.fov_y = math.atan(self.width/self.fy)
        
    def rays(self,nx,ny,M=None):
        self.cam = torch.zeros(nx,ny,3)
        X = self.fov_x/2*((torch.arange(nx)+0.5)/nx-0.5)
        Y = self.fov_y/2*((torch.arange(ny)+0.5)/ny-0.5)
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(0)

        self.cam[...,0] = torch.sin(Y)
        self.cam[...,1] = -torch.cos(Y)*torch.sin(X)
        self.cam[...,2] = -torch.cos(X)*torch.cos(Y)

        if not M is None:
            self.cam = self.M[...,:3,:3]@self.cam

        return self.cam

def relative_M(M_init,M):
    relative_M = torch.zeros(M_init.shape)
    relative_M[...,:3,:3] = torch.linalg.inv(M_init[...,:3,:3])@M[...,:3,:3]
    relative_M[...,:3,3] = M[...,:3,3]-M_init[...,:3,3]
    relative_M[...,3,3] = 1

    return relative_M
    
        
        
# %%
if __name__ == "__main__":
    M_init = torch.tensor([[
                -0.8362302780151367,
                -0.25379472970962524,
                0.486114501953125,
                0.0
            ],
            [
                0.548378586769104,
                -0.38701510429382324,
                0.7412829399108887,
                0.0
            ],
            [
                -1.4901161193847656e-08,
                0.8864579200744629,
                0.4628092646598816,
                0.0
            ],
            [
                -0.0033607978839427233,
                -0.07742857933044434,
                -1.2977442741394043,
                1.0
            ]]).T
    M = torch.tensor([[
                0.21511751413345337,
                0.9670152068138123,
                -0.13640376925468445,
                0.0
            ],
            [
                -0.9765881896018982,
                0.2130088210105896,
                -0.030046284198760986,
                0.0
            ],
            [
                0.0,
                0.13967378437519073,
                0.9901975989341736,
                0.0
            ],
            [
                -0.018085263669490814,
                0.013038650155067444,
                -1.2998102903366089,
                1.0
            ]]).T
    M_relative = relative_M(M_init,M_init)
    print(M_relative)
    C=Camera_v2()
    print(C.rays(32,32).shape)


# %%
