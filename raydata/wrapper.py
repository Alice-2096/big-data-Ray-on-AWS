import ray
from raydata.dataset import PushTImageDataset


def make_ray_dataset(dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int):
    # Create dataset
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )

    # Convert dataset to Ray format
    stats = dataset.stats
    print("converted dataset to Ray Dataset format")
    # dataset = ray.data.from_torch(dataset)
    return dataset, stats
