import torch
import torch.nn.functional as F
import torch.nn as nn
from Torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import config

class Embedding(nn.Module):
'''
input: original_image
output: patchfied pos enbedded tensor(z0 in paper)
        dim : [mini_batch_size, #patches +1, dim_model]
'''
    def __init__(self):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.input_channel = 3
        self.dim_model = config.dim_model
        self.embedding = nn.Linear(self.patch_size * self.patch_size \
                                   * self.input_channel, self.dim_model)
        self.class_weight = nn.Parameter(torch.zeros(1,1,self.dim_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (self.img_size//\
                                    self.patch_size)**2+1,self.dim_model))

        init.xavier_uniform_(self.class_weight)
        init.xavier_uniform_(self.pos_embedding)

    def forward(self, image):
        def patchify(image, patch_size, img_size):
        '''
        input
            - image : original image
            - patch_size : a length of a patch
            - img_size : target length after resize the original image

        output
            - patchified tensor, dim : [batch_size,#patch,(patch_size**2)*#channel]
        '''
            transform = Compose([Resize((img_size, img_size)), ToTensor()])
            x = transform(image)
            x = x.unsqueeze(0)
            patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',\
                                p1 = patch_size , p2 = patch_size)
            return patches

        patches = patchify(image, self.patch_size, self.img_size)
        embedded = self.embedding(patches)

        #to expand parameters (x_class and pos_embedding) mini batch size times
        x_class=repeat(self.class_weight,'() n d -> b n d',b =patches.size(0))
        pos_embedding = repeat(self.pos_embedding, '() n d -> b n d',\
                               b = patches.size(0))

        #z0 = [b * [x_class, x1p,x2p,...xnp]], shape :(n+1, dim_model)
        z = torch.cat([x_class, embedded], dm=1)
        z = z + pos_embedding #Positional embedding
        return z

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FFN(nn.Sequential):
    def __init__(self, expansion=4, drop_p = 0.):
        super().__init__(
            nn.Linear(config.dim_model, expansion * config.dim_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * config.dim_model, config.dim_model),
        )

class TransformerEncoderLayer(nn.Sequential):
'''
input:z_(k-1), dim: [#mini_batch, #patches per each mini_batch, dim_model]
    - num_head: number of heads of MHA
    - drop_p : dropout rate of Dropout right after MHA and FFN
    - forward_expansion: expansion argument of FFN
    - forward_drop_p : dropout rate of FFN

output:z_k, dim: same as dim of input tensor
'''
    def __init__(self, num_head, drop_p =0. forwrad_expansion=4, \
                 forward_drop_p = 0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(nn.LayerNorm(config.dim_model),\
                            nn.MultiHeadAttention(config.dim_model,num_head),\
                            nn.Dropout(drop_p)
                            )),

            ResidualAdd(nn.Sequential(nn.LayerNorm(config.dim_model), \
                                      FFN(expansion=forward_expansion,\
                                          drop_p = forward_drop_p),
                                      nn.Dropout(drop_p)
                                      )))

class TransformerEncoder(nn.Sequential):
'''
input:z_0, dim: [#mini_batch, #patches per each mini_batch, dim_model]
    - num_head: number of heads of MHA
    - drop_p : dropout rate of Dropout right after MHA and FFN
    - forward_expansion: expansion argument of FFN
    - forward_drop_p : dropout rate of FFN

output:z_L, dim: same as dim of input tensor
'''
    def __init__(self, **kwargs):
        super().__init__(*[TransformerEncoderLayer(**kwargs)\
                           for _ in range(config.depth)])

class Classification(nn.Sequential):
    def __init__(self, n_classes = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(config.dim_model),
            nn.Linear(config.dim_model, n_classes))

class ViT(nn.Sequential):
    def __init__(self, n_classes):
        super().__init__(Embedding(),TransformerEncoder(config.num_head),\
                        Classification(n_classes)
                         )

class ReLU_ViT(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.embedding = Embedding()

        self.encoder_layer = nn.TransformerEncoderLayer(config.dim_model, \
                                                        nhead = 8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)

        self.classification = Classification(n_classes)

    def forward(self,img):
        embedded = self.embedding(img)
        encodded = self.encoder(embedded)
        classes = self.classification(encodded)
        return classes



