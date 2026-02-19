import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.core.util import FourierEmbedder, checkpoint


class BroadcastPositionNet(nn.Module):
    def __init__(self,  in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))


    def forward(self, object_boxes, masks,
                object_positive_embeddings, inst_masks):
        B, N, seq_len, C = object_positive_embeddings.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may include padding as placeholder)
        object_xyxy_embedding = self.fourier_embedder(object_boxes).unsqueeze(-2).expand(-1, -1, seq_len, -1)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, 1, -1)

        # replace padding with learnable null embedding
        object_positive_embeddings = object_positive_embeddings * masks + (1 - masks) * positive_null

        object_xyxy_embedding = object_xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs_object = self.linears(torch.cat([object_positive_embeddings, object_xyxy_embedding], dim=-1))

        objs = objs_object

        # assert objs.shape == torch.Size([B, N*3, self.out_dim])
        return objs.view(B, -1, C)