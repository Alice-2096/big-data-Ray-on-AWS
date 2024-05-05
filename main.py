from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint, get_dataset_shard
import ray
import ray.train
import torch
import torch.nn as nn
import numpy as np
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from env.env_setup_main import setup_env
from raydata.wrapper import make_ray_dataset
from net.net_wrapper import network

from tqdm.auto import tqdm
import json
import ray.train.torch
print(ray.__version__)

ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
        )
    },
)
print(ray.cluster_resources())

#################### set up environment ####################
# setup_env()

#################### set up dataset and convert to Ray format ##########
s3_path = "s3://lab-bucket-1234/pusht_cchi_v7_replay.zarr"
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
train_dataset, stats = make_ray_dataset(
    s3_path, pred_horizon, obs_horizon, action_horizon)
print("Ray training dataset stats:", stats)

# # create dataloader
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

#################### set up training function ####################


def train_func():
    # create model
    num_epochs = 10
    batch_size = 64
    nets, noise_pred_net = network(obs_horizon)

    # distribute the model to the GPU if available
    print("Prepare model")
    # nets = ray.train.torch.prepare_model(nets)
    # noise_pred_net = ray.train.torch.prepare_model(noise_pred_net)

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=10 * num_epochs
    )
    num_diffusion_iters = 10
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # Access the dataset shard for the training worker
    # train_data_shard = ray.train.get_dataset_shard("train")
    # train_dataloader = train_data_shard.iter_torch_batches(
    # batch_size=batch_size, shuffle=True)
    # print("training data shard stats:", train_data_shard.stats())
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=7,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    print("Prepare dataloader")
    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    print("processing batch")
                    # normalize data in dataset
                    nimage = nbatch['image'][:, :obs_horizon]
                    nagent_pos = nbatch['agent_pos'][:,
                                                     : obs_horizon]
                    naction = nbatch['action']
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2], -1)
                    obs_features = torch.cat(
                        [image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,)
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())


trainer = TorchTrainer(
    train_func,
    # scaling_config=ScalingConfig(
    # num_workers=7),
    # datasets={
    #     "train": dataset
    # },
    run_config=ray.train.RunConfig(storage_path="s3://lab-bucket-1234")
)
result = trainer.fit()
