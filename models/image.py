from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.dropout import Dropout2d

from torch.nn.modules.linear import Linear
from .new_efficientnet import Efficientnet_Attv2
import pdb
try:
    from timm.models import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, \
    tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, \
    xception

    from efficientnet_pytorch.model import EfficientNet
except:
    from timm.models import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, \
    tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, \
    xception

    from efficientnet_pytorch.model import EfficientNet


__all__ = ['BinaryClassifier']


encoder_params = {
    "saia_efficientnet-b4": {
        "features": 1792,
        "init_op": partial(Efficientnet_Attv2, model_name='efficientnet-b4')
    },
}




class BinaryClassifier(nn.Module):
    def __init__(self, encoder, num_classes=1, drop_rate=0.2, has_feature=True,feature_dim=128,**kwargs) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"](**kwargs)
        self.global_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(drop_rate)
        self.channel_drop = Dropout2d(drop_rate)
        self.has_feature = has_feature
        self.feature = Linear(encoder_params[encoder]["features"], feature_dim)
        self.fc = Linear(feature_dim, num_classes)

    def forward(self, x,sia=True):
        featuremap = self.encoder.forward_features(x,sia=True)
        x = self.global_pool(featuremap).flatten(1)
        feat = self.feature(x)
        output = self.fc(feat)
        if self.has_feature:
            return output,featuremap
        return output

if __name__ == '__main__':
    model = BinaryClassifier("efficientnet-b5")
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        input = torch.rand(4, 3, 320, 320)
        input = input.cuda()
        out = model(input)
        print(out.shape)
    