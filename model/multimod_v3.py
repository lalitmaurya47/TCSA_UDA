from model.deeplabv2_v1 import get_deeplab_v2  # make sure this returns nn.Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):  # for 64 channel
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, do_activation=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.do_activation = do_activation

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, q, k, v):

        """
        q shape: 1 x N'x C
        k shape: 1 x N'x C
        v shape: 1 x N'x C
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # activation here
        if self.do_activation:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # activation here
        if self.do_activation:
            output = self.activation(output)

        output = self.layer_norm(output + residual)

        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron module
    """
    def __init__(self, dim, mlp_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = self.mlp(x) + x
        x = self.norm(x)
        return x

class Extractor_DI(nn.Module):
    def __init__(self, n_channels=128, out_channels = 64):
        super(Extractor_DI, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, n_channels*2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(n_channels*2, n_channels*2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(n_channels*2, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            )
    def forward(self, f_all):
        f_di = self.inc(f_all)
        return f_di




class MulModSeg2D(nn.Module):
    def __init__(self, out_channels, backbone='deeplabv2', encoding='rand_embedding',
                 multi_level=True, embedding_path='/content/mod_cls_txt_encoding_heart.pth'):
        super().__init__()
        self.backbone_name = backbone
        self.class_num = out_channels
        self.encoding = encoding

        if backbone == 'deeplabv2':
            self.backbone = get_deeplab_v2(num_classes=out_channels, multi_level=multi_level)
            self.backbone_out_channels = 2048  # Typical for ResNet101-based DeepLabV2
            self.GAP = nn.Sequential(
                nn.GroupNorm(8, self.backbone_out_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(self.backbone_out_channels, 256, kernel_size=1)
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")

        # Task embeddings
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            if embedding_path is None:
                raise ValueError("embedding_path must be provided for word_embedding.")
            embeddings = self.load_embedding(embedding_path)  # Load from file
            self.register_buffer('organ_embedding', embeddings)  # [2, C, 512]
            self.text_to_vision = nn.Linear(512, 256)

        # Transformer fusion

         # Transformer fusion
        self.mha = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        
      
        self.MHA = MultiHeadAttention(n_head=3, d_model=256, d_k=256, d_v=256)
        self.MLP = MultiLayerPerceptron(dim=256, mlp_dim=512)
        #self.EDI = Extractor_DI(n_channels=128, out_channels = 128)
        #self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.BatchNorm2d(256)
        self.fusion_input_proj = nn.Linear(512, 256)

        # === Controller for Dynamic Conv ===
        self.controller_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 128 * 1 * 1 + 128)  # W: (128, 256, 1, 1), b: (128)
        )

      
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )


    def load_embedding(self, embedding_path):
        data = torch.load(embedding_path, map_location='cpu')
        assert isinstance(data, torch.Tensor), "Loaded embedding is not a tensor"
        assert data.ndim == 3 and data.shape[-1] == 512, "Expected shape [2, N, 512]"
        return data

    def forward(self, x_in, modality):
        B = x_in.shape[0]
        cla_feas_src, pred_src_aux, pred_src_main = self.backbone(x_in)
        pred_main = pred_src_main  # [B, 256, H, W]
        H, W = pred_main.shape[2], pred_main.shape[3]
    
        # Global visual feature from GAP
        x_feat = self.GAP(cla_feas_src)  # [B, 256, 1, 1]
        x_feat = x_feat.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, 256]
    
        # Task encoding
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight  # [C, 256]
        elif self.encoding == 'word_embedding':
            if self.organ_embedding.shape[0] == 2:
                if modality == 'MR':
                    task_encoding = F.relu(self.text_to_vision(self.organ_embedding[0]))  # [C, 256]
                elif modality == 'CT':
                    task_encoding = F.relu(self.text_to_vision(self.organ_embedding[1]))  # [C, 256]
                else:
                    raise ValueError(f"Modality {modality} not supported.")
            else:
                task_encoding = F.relu(self.text_to_vision(self.organ_embedding))  # [C, 256]
        
        text_emb = task_encoding
        task_encoding = task_encoding.unsqueeze(0).repeat(B, 1, 1)  # [B, C, 256]
    
        # === Stage 1: Task Encoding × Global Visual Feature ===
        # fusion1, _ = self.mha(query=task_encoding, key=x_feat, value=x_feat)  # [B, C, 256]
        # fusion1 = self.fusion_ffn(fusion1)  # [B, C, 256]
        x_feat_repeated = x_feat.repeat(1, task_encoding.shape[1], 1)  # [B, C, 256]

        # Step 2: Concatenate x_feat with task_encoding
        fused_query = torch.cat([task_encoding, x_feat_repeated], dim=-1)  # [B, C, 512]
        
        # Step 3: Project to 256-dim to make it valid input for attention
        fused_query = F.relu(self.fusion_input_proj(fused_query))  # [B, C, 256]

        fusion1 = self.MHA(fused_query, x_feat, x_feat)  # [B, C, 256]
        fusion1 = self.MLP(fusion1)  # [B, C, 256]
    
       
        # === Controller output generation ===
        # Global pooled controller vector from fusion1
        controller_input = fusion1.mean(dim=1)  # [B, 256]

        # Generate θ_k (weights and biases)
        theta_k = self.controller_mlp(controller_input)  # [B, total_params]

        weight_size = 128 * 256 * 1 * 1
        bias_size = 128

        conv_weights = theta_k[:, :weight_size].view(B, 128, 256, 1, 1)
        conv_biases = theta_k[:, weight_size:].view(B, 128)

        # === Apply dynamic conv ===
        
        for i in range(B):
            out_i = F.conv2d(pred_main[i:i+1], conv_weights[i], conv_biases[i])
            output.append(out_i)
        pred_final = torch.cat(output, dim=0)  # [B, 128, H, W]
      
        # Segmentation head
        out = self.seg_head(pred_final)  # [B, out_channels, H, W]
        #out1 = self.seg_head(pred_final)
        #out2 = self.seg_head(f_di)
        return cla_feas_src, pred_src_aux, out, text_emb, pred_src_main
        #return cla_feas_src, out1, out2

