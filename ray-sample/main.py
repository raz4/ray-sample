import ray
import ray.tune as tune
from ray.rllib import train

import os
import sys

if __name__ == "__main__":

    # Parse arguments
    train_parser = train.create_parser()

    args = train_parser.parse_args()
    print("Algorithm config:", args.config)

    if args.ray_address is None:
        # start local Ray cluster
        ray.init(include_dashboard=False)
    else:
        # attempt to connect to remote Ray Cluster
        ray.init(address=args.ray_address)

    tune.run(run_or_experiment=args.run,
             config={
                 "env": args.env,
                 "num_gpus": 0,
                 "num_workers": 1,
                 "train_batch_size": 32
             },
             stop=args.stop,
             local_dir='./logs')

    ray.shutdown()
