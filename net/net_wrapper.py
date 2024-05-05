from net.network import ConditionalUnet1D, ConditionalResidualBlock1D, Conv1dBlock
from net.resnet import get_resnet, replace_submodules, replace_bn_with_gn
from torch import nn


def network(obs_horizon):
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    return nets, noise_pred_net
