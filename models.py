#%%
import torch
import torch.nn as nn
import numpy as np
import torchvision
from camera import Camera
import utils 

def mask_gen(dimension):
    return torch.triu(torch.full((dimension, dimension), float('-inf')), diagonal=1)
"""
class RayTransformer(nn.Module):
    def __init__(self,nb_object=20,d_encoder=64,d_object=32,channel_output=16,num_transformer_layer=1,resolution_in=(112,112),resolution_out=(56,56)):
        super().__init__()
        self.d_encoder = d_encoder
        self.d_object = d_object
        self.channel_output=channel_output
        self.nb_object=nb_object
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out

        
        self.imageEncoder = ImageEncoder(last_channel=d_encoder,resolution=resolution_in)

        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.init_token = nn.Parameter(torch.randn(d_encoder))

        self.objectDisentenglement = nn.Linear(d_encoder,d_object+9)

        C = Camera(resolution=resolution_out)
        self.rays = C().reshape(-1,3)

        self.ray_to_object = nn.Linear(6,d_object)

        #self.form_decoder = nn.Linear(d_object,channel_output+2)
        self.form_decoder = Residual_fc(d_object,d_object,nb_residual_blocks=2,hidden_state=d_object)
        self.ray_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_object, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.linear_pond = nn.Linear(d_object,1)

        self.conv_1 = nn.Conv2d(d_object,d_object,1)
        self.conv_2 = nn.Conv2d(d_object,channel_output,3,padding=1)
        self.conv_upscale_1 = nn.ConvTranspose2d(channel_output,channel_output,3,stride=2,output_padding=1,padding=1)
        self.conv_upscale_2 = nn.ConvTranspose2d(channel_output,3,3,stride=2,output_padding=1,padding=1)

        

        

    def conv_output(self,x):
        render = self.conv_1(x).relu()
        render = self.conv_2(render).relu()
        render = self.conv_upscale_1(render)
        render = self.conv_upscale_2(render)
        return render
    
    def decoder(self,source,gen=20):
        mask = mask_gen(20).to(source.device)
        tokens = self.init_token.unsqueeze(0).unsqueeze(0).repeat(source.shape[0],1,1)

        for i in range(gen):
            out = self.TransformerDecoder(tokens,source,tgt_mask=mask[:i+1,:i+1])
            tokens = torch.concat([tokens,out[:,-1:,:]],dim=-2)
        
        return out
    
    def objects_render(self,form_decoding):
        b,r,o,d = form_decoding.shape
        pond = self.linear_pond(self.ray_transformer(form_decoding.reshape(-1,o,d))).squeeze(-1).softmax(-1).reshape(b,r,o)
        result = torch.einsum('bro,brod->brd',pond,form_decoding)
        result = result.reshape((b,)+self.resolution_out+(result.shape[-1],)).permute(0,3,1,2)

        return result,pond
    

    def objectEncoder(self,images):
        x = self.imageEncoder(images)
        b,w,h,d= x.shape
        x = x.reshape(b,-1,d)
        x = self.TransformerEncoder(x) #(b,e,32)
        
        x = self.decoder(x,self.nb_object)#(b,t,d)

        x_object,x_spatial = self.objectDisentenglement(x).split([self.d_object,9],dim=-1) #(b,t,d_o),(b,t,9)

        return x_object,x_spatial
    
    def objectDecoder(self,x_object,x_spatial,rays,prompt_var=False):
        b = x_object.shape[0]
        t = x_object.shape[1]
        x_object = x_object.unsqueeze(1).repeat(1,rays.shape[0],1,1) #(b,r,t,d_o)
        ray_result = utils.coordRay(x_spatial,rays.to(x_object.device)) # (b,r,t,6)

        form_decoding = self.form_decoder(x_object + self.ray_to_object(ray_result))
        
        result,pond = self.objects_render(form_decoding)
        
        render = self.conv_output(result)

        if not prompt_var:
            return render
        else:
            dic = {'render':render}
            dic['ponderation'] = pond.reshape((b,)+self.resolution_out+(t,))
            dic['segment_hard'] = dic['ponderation'].max(-1)[1]
            return dic

    def forward(self,x):
        x_object,x_spatial = self.objectEncoder(x)
        render = self.objectDecoder(x_object,x_spatial,self.rays)

        return render
    
    def output(self,x,object_pos=None,keep_object=None,color=None):
        x_object,x_spatial = self.objectEncoder(x)

        if not object_pos is None:
            if len(object_pos.shape)==1:
                object_pos = object_pos.unsqueeze(0).repeat(x_object.shape[1],1)
            R_cam = rotation_to_matrix(object_pos[...,:3]).to(x_spatial.device)
            x_cam = torch.concat([torch.zeros(x_object.shape[1],6),object_pos[...,3:]],dim=-1).to(x_spatial.device)
            x_spatial = torch.einsum('oij,bojd->boid',R_cam,(x_spatial+ x_cam).unfold(-1,3,3).permute(0,1,3,2)).permute(0,1,3,2).reshape(x_spatial.shape)
        
        if not keep_object is None:
            keep_object = keep_object.to(x_spatial.device)
            x_spatial = x_spatial.index_select(1,keep_object)
            x_object = x_object.index_select(1,keep_object)
            if not color is None:
                color = color.to(x_object.device)
                color = color.index_select(0,keep_object)
        
        dic = self.objectDecoder(x_object,x_spatial,self.rays,prompt_var=True)

        if color is None:
                color = torch.rand(x_object.shape[1],3)
        color = color.to(x_object.device)
        dic['segmentat_hard'] = dic['ponderation'].max(-1)
        dic['x_spatial'] = x_spatial
        dic['x_object'] = x_object

        return dic

"""

class RayTransformer(nn.Module):
    def __init__(self,nb_object=20,d_encoder=64,d_object=32,channel_output=16,num_transformer_layer=1,resolution_in=(112,112),resolution_out=(56,56)):
        super().__init__()
        self.d_encoder = d_encoder
        self.d_object = d_object
        self.channel_output=channel_output
        self.nb_object=nb_object
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out

        
        self.imageEncoder = ImageEncoder(last_channel=d_encoder,resolution=resolution_in)

        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.init_token = nn.Parameter(torch.randn(d_encoder))

        self.objectDisentenglement = nn.Linear(d_encoder,d_object+9)

        C = Camera(resolution=resolution_out)
        self.rays = C().reshape(-1,3)

        self.ray_to_object = nn.Linear(6,d_object)

        #self.form_decoder = nn.Linear(d_object,channel_output+2)
        self.form_decoder = Residual_fc(d_object,d_object+2,nb_residual_blocks=4,hidden_state=d_object)

        self.conv_1 = nn.Conv2d(d_object,d_object,1)
        self.conv_2 = nn.Conv2d(d_object,channel_output,3,padding=1)
        self.conv_upscale = nn.ConvTranspose2d(channel_output,3,3,stride=2,output_padding=1,padding=1)

        

        

    def conv_output(self,x):
        render = self.conv_1(x).relu()
        render = self.conv_2(render).relu()
        render = self.conv_upscale(render)
        return render
    
    def decoder(self,source,gen=20):
        mask = mask_gen(20).to(source.device)
        tokens = self.init_token.unsqueeze(0).unsqueeze(0).repeat(source.shape[0],1,1)

        for i in range(gen):
            out = self.TransformerDecoder(tokens,source,tgt_mask=mask[:i+1,:i+1])
            tokens = torch.concat([tokens,out[:,-1:,:]],dim=-2)
        
        return out
    
    def objects_render(self,form_decoding,depth,transmitance):
        b= form_decoding.shape[0]
        o = depth.shape[-1]
        depth_comp_matrix = torch.sigmoid(1000*(depth.unsqueeze(-1).repeat(1,1,1,o)-depth.unsqueeze(-2).repeat(1,1,o,1)))
        depth_comp_matrix = depth_comp_matrix*(torch.full((o,o),1)-torch.eye(o)).to(form_decoding.device)
        ponderation  = torch.exp(torch.einsum('brO,broO->bro',transmitance,depth_comp_matrix))*(1-torch.exp(transmitance))

        result = torch.einsum('bro,brod->brd',ponderation,form_decoding)
        result = result.reshape((b,)+self.resolution_out+(result.shape[-1],)).permute(0,3,1,2)

        return result,ponderation
    

    def objectEncoder(self,images):
        x = self.imageEncoder(images)
        b,w,h,d= x.shape
        x = x.reshape(b,-1,d)
        x = self.TransformerEncoder(x) #(b,e,32)
        
        x = self.decoder(x,self.nb_object)#(b,t,d)

        x_object,x_spatial = self.objectDisentenglement(x).split([self.d_object,9],dim=-1) #(b,t,d_o),(b,t,9)

        return x_object,x_spatial
    
    def objectDecoder(self,x_object,x_spatial,rays,prompt_var=False):
        b = x_object.shape[0]
        t = x_object.shape[1]
        x_object = x_object.unsqueeze(1).repeat(1,rays.shape[0],1,1) #(b,r,t,d_o)
        ray_result = utils.coordRay(x_spatial,rays.to(x_object.device)) # (b,r,t,6)

        form_decoding,relative_pos,transmitance = self.form_decoder(x_object + self.ray_to_object(ray_result)).split([self.d_object,1,1],dim=-1)

        transmitance = -transmitance.squeeze(-1)**2
        depth = relative_pos.squeeze(-1)+torch.einsum('bod,rd->bro',x_spatial[...,6:],rays.to(x_object.device))
        
        
        result,pond = self.objects_render(form_decoding,depth,transmitance)
        
        render = self.conv_output(result)

        if not prompt_var:
            return render
        else:
            dic = {'render':render}
            

            dic['ponderation']  = pond.reshape((b,)+self.resolution_out+(t,))
            dic['depth'] = depth.reshape((b,)+self.resolution_out+(t,))
            dic['transmitance'] = transmitance.reshape((b,)+self.resolution_out+(t,)).exp()

            return dic

    def forward(self,x):
        x_object,x_spatial = self.objectEncoder(x)
        render = self.objectDecoder(x_object,x_spatial,self.rays)

        return render
    
    def output(self,x,object_pos=None,keep_object=None,color=None):
        x_object,x_spatial = self.objectEncoder(x)

        if not object_pos is None:
            if len(object_pos.shape)==1:
                object_pos = object_pos.unsqueeze(0).repeat(x_object.shape[1],1)
            R_cam = rotation_to_matrix(object_pos[...,:3]).to(x_spatial.device)
            x_cam = torch.concat([torch.zeros(x_object.shape[1],6),object_pos[...,3:]],dim=-1).to(x_spatial.device)
            x_spatial = torch.einsum('oij,bojd->boid',R_cam,(x_spatial+ x_cam).unfold(-1,3,3).permute(0,1,3,2)).permute(0,1,3,2).reshape(x_spatial.shape)
        
        if not keep_object is None:
            keep_object = keep_object.to(x_spatial.device)
            x_spatial = x_spatial.index_select(1,keep_object)
            x_object = x_object.index_select(1,keep_object)
            if not color is None:
                color = color.to(x_object.device)
                color = color.index_select(0,keep_object)
        
        dic = self.objectDecoder(x_object,x_spatial,self.rays,prompt_var=True)

        if color is None:
                color = torch.rand(x_object.shape[1],3)
        color = color.to(x_object.device)

        dic['x_spatial'] = x_spatial
        dic['x_object'] = x_object
        dic['segment_soft'] = torch.einsum('oc,bwho->bcwh',color,dic['ponderation'])
        dic['ponderation_max'],dic['segment_hard'] = dic['ponderation'].max(-1)
        dic['depth_sum'] = (dic['ponderation']*dic['depth'] ).sum(-1)

        return dic


def rotation_to_matrix(rotation):
    R_x,R_y,R_z = torch.zeros(rotation.shape[0],3,3),torch.zeros(rotation.shape[0],3,3),torch.zeros(rotation.shape[0],3,3)

    R_x[:,0,0] = 1
    R_x[:,1,1] = rotation[...,0].cos()
    R_x[:,2,2] = rotation[...,0].cos()
    R_x[:,1,2] = -rotation[...,0].sin()
    R_x[:,2,1] = rotation[...,0].sin()

    R_y[:,0,0] = rotation[...,1].cos()
    R_y[:,1,1] = 1
    R_y[:,0,2] = rotation[...,1].sin()
    R_y[:,2,0] = -rotation[...,1].sin()
    R_y[:,2,2] = rotation[...,1].cos()

    R_z[:,0,0] = rotation[...,2].cos()
    R_z[:,0,1] = -rotation[...,2].sin()
    R_z[:,1,0] = rotation[...,2].sin()
    R_z[:,1,1] = rotation[...,2].cos()
    R_z[:,2,2] = 1

    R_cam = torch.einsum('oij,ojd->oid', torch.einsum('oij,ojd->oid',R_z,R_y),R_x)
    return R_cam
        
    
def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

    

class ImageEncoder(nn.Module):
    def __init__(self,patch_size=16,last_channel=64,resolution=(112,112)) -> None:
        super().__init__()
        self.last_channel = last_channel
        self.conv1 = nn.Conv2d(3,last_channel//4,3,bias=False)
        self.conv2 = nn.Conv2d(last_channel//4,last_channel//2,5,stride=2,bias=False)
        self.conv3 = nn.Conv2d(last_channel//2,last_channel,5,bias=False)
        self.grid = build_grid((resolution[0]//patch_size,resolution[1]//patch_size))
        self.softPosEmbbeding = nn.Linear(4,last_channel)
    def forward(self,x):

        x = x.unfold(2,16,16).unfold(3,16,16).permute(0,2,3,1,4,5)
        b,p_w,p_h,_,w,h = x.shape
        x= x.reshape(-1,3,w,h)
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).squeeze(-1).squeeze(-1)
        x = x.reshape(b,p_w,p_h,self.last_channel)

        x = x + self.softPosEmbbeding(self.grid.to(x.device))
        return x
    

class Residual_fc(nn.Module):
    def __init__(self,input_dim=64,output_dim=64,nb_residual_blocks=4,hidden_state=64,dropout_rate=0):
        super(Residual_fc,self).__init__()
        self.fc_input = nn.Linear(input_dim,hidden_state)

        self.residual_blocks = nn.ModuleList()
        for l in range(nb_residual_blocks):
            layer = []
            layer.append(nn.Linear(hidden_state,hidden_state))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout_rate))

            self.residual_blocks.append(nn.Sequential(*layer))

        self.fc_output = nn.Linear(hidden_state,output_dim)


    def forward(self,x):
        x = self.fc_input(x).relu()
        for block in self.residual_blocks:
            x = x+block(x)
        x = self.fc_output(x)
        return x