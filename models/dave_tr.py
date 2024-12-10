import itertools

import torch
from torch import nn
from torchvision.ops import roi_align

from .backbone import Backbone
from .feat_comparison import Feature_Transform
from .boundingbox import BBOX_Network
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F

class COTR(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_objects: int,
        emb_dim: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        use_query_pos_emb: bool,
        zero_shot: bool,
        use_objectness: bool,
        use_appearance: bool,
    ):

        super(COTR, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.use_query_pos_emb = use_query_pos_emb
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_objectness = use_objectness
        self.use_appearance = use_appearance
        self.cosine_sim = nn.CosineSimilarity()
        self.backbone = Backbone(
            backbone_name,
            pretrained=True,
            dilation=False,
            reduction=reduction,
            swav=swav_backbone,
            requires_grad=train_backbone,
        )
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.feat_comp = Feature_Transform()
        self.bbox_network = BBOX_Network(input_dim=6, hidden_dim=64, output_dim=64)
    def forward(self, x, bboxes):
        # backbone
        backbone_features = self.backbone(x).detach()
        # backbone feature : [batch size : 32, 3584, height : 64, wdith : 128]
        bs, _, bb_h, bb_w = backbone_features.size()

        # bboxes : [32, 6, 4] = [batch size, interest number, (x,y,x1,y1)]
        # 6 (number of bounding boxes per image): 
        # Each image has 6 bounding boxes or regions of interest.

        # Get shape of objectness
        shape_or_objectness = self.bbox_network(bboxes)

        bboxes_ = torch.cat(
            [
                torch.arange(bs, requires_grad=False)
                .to(bboxes.device)
                .repeat_interleave(bboxes.shape[1])
                .reshape(-1, 1),
                bboxes.flatten(0, 1),
            ],
            dim=1,
        )
        # bboxes_ : 192, 5 [batch size x interest number, (id,x,y,x1,y1)]
        ROI_feature = roi_align(
                backbone_features,
                boxes=bboxes_,
                output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction,
                aligned=True,
            )
        # Orifinal ROI_feature : [batchsize(32) * interest number(6), 3584, 3, 3]
        ROI_feature = ROI_feature.permute(0, 2, 3, 1)
        # After ROI_feature : [batchsize(32) * interest number(6), 3, 3, 3584]
        ROI_feature_reshpae = ROI_feature.reshape(bs, 6, 3, 3, -1)
        # ROI_feature_reshpae : [batchsize(32), interest number(6), 3, 3, 3584]
        feat_vectors = ROI_feature_reshpae.permute(0, 1, 4, 2, 3)
        # feat_vectors : [batchsize(32), interest number(6), 3584, 3, 3]
        feat_embedding = self.feat_comp(feat_vectors.reshape(bs * 6, 3584, 3, 3))
        # feat_embedding : [batchsize(32) * interest number(6), 6400] -> [batchsize(32), interest number(6), 6400]
        feat_embedding = feat_embedding.reshape(bs, 6, -1)
        # feat_pairs : [6, 32, 6400]
        feat_pairs = feat_embedding.permute(1, 0, 2)
        shape_or_objectness = shape_or_objectness.permute(1, 0, 2)
        combined_features = torch.cat((feat_pairs, shape_or_objectness), dim=-1)
        print("combined_features : ",combined_features.shape)
        sim = list()
        class_ = []
        loss = torch.tensor(0.0).to(feat_pairs.device)
        o = torch.tensor(1).to(feat_pairs.device)
        n = torch.tensor(-1).to(feat_pairs.device)
        # combined_features reshape to [batchsize(32) * interest number(6), 6464]
        # combined_features = combined_features.view(-1, combined_features.shape[-1])
        '''       
        threshold = 0.8
        margin = 0.2
        for f1, f2 in itertools.combinations(zip(combined_features), 2):
            # cosine similarity
            f1_tensor, f2_tensor= f1[0], f2[0]      
            # Calculate cosine similarity
            sim_value = F.cosine_similarity(f1_tensor, f2_tensor, dim=0, eps=1e-8)
            if sim_value >= threshold: 
                loss += (1 - sim_value)  
            else:
                loss += F.relu(margin - sim_value) 
        
            sim.append(sim_value.item())
        num_pairs = len(sim)
        loss = loss / num_pairs if num_pairs > 0 else loss
        return loss, sim
        '''

        #combined_features = combined_features.reshape(bs,6,-1).permute(1, 0, 2)
        #print(combined_features.shape)
        for f1, f2 in itertools.combinations(zip(combined_features, [1, 1, 1, 2, 2, 2]), 2):
            for i in range(f1[0].shape[0]):
                loss += self.cos_loss(f1[0][i], f2[0][i], o if f1[1] == f2[1] else n)
            sim.append(self.cosine_sim(f1[0], f2[0]))
            class_.append([f1[1] == f2[1] for _ in range(bs)])

        return loss, sim


def build_model(args):
    return COTR(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        use_query_pos_emb=args.use_query_pos_emb,
        use_objectness=args.use_objectness,
        use_appearance=args.use_appearance,
    )
