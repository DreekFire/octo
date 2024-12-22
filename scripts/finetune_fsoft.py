"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small --data_dir=...
"""
from functools import partial
from absl import app, flags, logging
import flax
import jax
import optax
# import tensorflow as tf
import tqdm
import wandb
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch

devices = mesh_utils.create_device_mesh((4,))
mesh = Mesh(devices, axis_names=('a',))
sharding = NamedSharding(mesh, PartitionSpec('a',))
print(sharding)

import yaml
import os
import tyro
from pathlib import Path
import time

from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead, MSEActionHead, DiscreteActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer, TubletTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

from timm.data.loader import MultiEpochsDataLoader

from icrt.util.args import ExperimentConfig
from icrt.data.sequence_dataset import SequenceDataset
from icrt.util.misc import DistributedSubEpochSampler

# FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
# )
# # flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
# flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
# # flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")

# flags.DEFINE_bool(
#     "freeze_transformer",
#     False,
#     "Whether pre-trained transformer weights should be frozen.",
# )

traj_mixture_mode = 'same_task'
discrete_action = True

from tpu_utils import prevent_cross_region

def main(args):
    prevent_cross_region("gs://derek_central_2/ICRT-MT")

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    # tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    # modify for each run
    wandb.init(name=f"action_mc_{traj_mixture_mode}", project="fsoft_octo")

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    # dataset = make_single_dataset(
    #     dataset_kwargs=dict(
    #         name="aloha_sim_cube_scripted_dataset",
    #         data_dir=FLAGS.data_dir,
    #         image_obs_keys={"primary": "top"},
    #         state_obs_keys=["state"],
    #         language_key="language_instruction",
    #         action_proprio_normalization_type=NormalizationType.NORMAL,
    #         absolute_action_mask=[True] * 14,
    #     ),
    #     traj_transform_kwargs=dict(
    #         window_size=1,
    #         future_action_window_size=49,  # so we get 50 actions for our action chunk
    #     ),
    #     frame_transform_kwargs=dict(
    #         resize_size={"primary": (256, 256)},
    #     ),
    #     train=True,
    # )
    # train_data_iter = (
    #     dataset.repeat()
    #     .unbatch()
    #     .shuffle(10000)  # can reduce this if RAM consumption too high
    #     .batch(FLAGS.batch_size)
    #     .iterator()
    # )

    dataset_train = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=None,
        no_aug_vision_transform=None,
        split="train",
    )
    dataset_val = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=None,
        no_aug_vision_transform=None,
        split="val"
    )

    dataset_train.save_split(os.path.join(args.logging_cfg.output_dir, "train_split.json"))
    dataset_val.save_split(os.path.join(args.logging_cfg.output_dir, "val_split.json"))

    wandb.init(entity="derekguo", project="fsoft_icrt")

    # print(f"Shuffling sequences every {1} epochs, epoch: {epoch}")
    dataset_train.shuffle_dataset(0)
    dataset_val.shuffle_dataset(0)
    
    sampler_train = DistributedSubEpochSampler(
        dataset_train, num_replicas=1, rank=0, split_epoch=1, shuffle=True
    )
    sampler_val = DistributedSubEpochSampler(
        dataset_val, num_replicas=1, rank=0, split_epoch=1, shuffle=False
    )
    print("Sampler_train = %s" % str(sampler_train))
    print("length of train sampler: ", len(sampler_train))
    print("Sampler_val = %s" % str(sampler_val))
    print("length of val sampler: ", len(sampler_val))
    data_loader_train = MultiEpochsDataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.shared_cfg.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    train_data_iter = data_loader_train.__iter__()
    if len(sampler_val) > args.shared_cfg.batch_size:
        data_loader_val = MultiEpochsDataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.shared_cfg.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        val_data_iter = data_loader_val.__iter__()
    else:
        data_loader_val = None


    def transform_batch(batch):
        batch = jax.tree_util.tree_map(jnp.array, batch)
        batch = jax.tree_util.tree_map(lambda x: x[:, ::2], batch)
        obs = batch['observation'].transpose((0, 1, 2, 4, 5, 3))
        batch_size, seq_len = obs.shape[:2]
        ret = {
            'observation': {
                'image_primary': obs, # (N, S, T, 224, 224, 3)
                'proprio': batch['proprio'].squeeze(2), # (N, S, p)
                'pad_mask': jnp.ones((obs.shape[:2])) # (N, S)
            },
            'task': {
                'image_primary': obs.squeeze(2)[:, ::32], # (N, S // 32, 224, 224, 3)
            },
            'action': batch['action'].squeeze(2)[..., :-1], # (N, S, a) # TODO: handle EoS dim
            'value': jnp.minimum(batch['rstep'][..., None], 256), #(2 * 0.995 ** batch['rstep'] - 1)[..., None], # closed form formula for terminal reward 1, discount 0.995, intermediate reward -0.005
        }
        return ret

    # batch shape:
    # "observation": torch.Tensor, shape (batch_size, seq_length, num_cameras, 3, 224, 224)
    # "proprio": torch.Tensor, shape (batch_size, seq_length, num_pred_steps, proprio_dim)
    # "action": torch.Tensor, shape (batch_size, seq_length, num_pred_steps, action_dim)

    # need to transform it into:
    # This model is trained with a window size of 2, predicting 7 dimensional actions 4 steps into the future.
    # Observations and tasks conform to the following spec:
    # Observations: {
    #     image_primary: ('batch', 'history_window', 256, 256, 3),
    #     image_wrist: ('batch', 'history_window', 128, 128, 3),
    # }
    # Tasks: {
    #     image_primary: ('batch', 256, 256, 3),
    #     image_wrist: ('batch', 128, 128, 3),
    #     language_instruction: {
    #         attention_mask: ('batch', 16),
    #         input_ids: ('batch', 16),
    #     },
    # }

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(transform_batch, train_data_iter)
    example_batch = next(train_data_iter)
    
    # TODO: handle observation histories (3rd dim not 1)
    def reshape_batch_same_traj(x):
        # (N, S, 1, 224, 224, 3) -> (S, N, 1, 1, 224, 224, 3)
        # (sub-batch, batch, horizon, time, h, w, c)
        return jnp.expand_dims(jnp.swapaxes(x, 0, 1), 2)
    
    def reshape_batch_same_task(x):
        # (N, S, 1, 224, 224, 3) -> (S * N, N, 1, 1, 224, 224, 3)
        batch_size, seq_len, *remaining_shape = x.shape
        x = jnp.broadcast_to(jnp.expand_dims(jnp.swapaxes(x, 0, 1), 0), (batch_size, *x.shape)) # (N, S, N, 1, 224, 224, 3)
        return x.reshape((batch_size * seq_len, batch_size, 1, *remaining_shape)) # (N * S, N, 1, 1, 224, 224, 3)

    def reshape_batch_mixed_task(batch):
        raise NotImplementedError
    
    # def reshape_task_same_traj(x):
    #     seq_len = x.shape[1]
    #     x = x[:, ::32]
    #     if x.ndim > 2:
    #         x = x.squeeze(2)
    #     x = jnp.broadcast_to(jnp.expand_dims(x, 0), (seq_len, *x.shape)) # (N, S // 32, 224, 224, 3) -> (S * N, S // 32, 224, 224, 3)
    #     return x.reshape((-1, *x.shape[2:]))

    # def reshape_task_same_task(x):
    #     batch_size, seq_len, *remaining_shape = x.shape
    #     x = x[:, ::32]
    #     if x.ndim > 2:
    #         x = x.squeeze(2)
    #     # want a different order so they task and observations get paired between each other. same difference as tile vs repeat
    #     x = jnp.expand_dims(x, 1).broadcast_to((x.shape[0], seq_len * batch_size, *x.shape[1:])) # (N, S // 32, 224, 224, 3) -> (N * S * N, S // 32, 224, 224, 3)
    #     return x.reshape((-1, *x.shape[2:]))

    # def reshape_task_mixed_task(batch):
    #     raise NotImplementedError

    batch_reshape_fns = {
        'same_traj': reshape_batch_same_traj,
        'same_task': reshape_batch_same_traj, # TODO: reimplement
        'mixed_task': reshape_batch_mixed_task,
    }
    
    # task_reshape_fns = {
    #     'same_traj': reshape_task_same_traj,
    #     'same_task': reshape_task_same_task,
    #     'mixed_task': reshape_task_mixed_task,
    # }
    
    reshape_fn = batch_reshape_fns[traj_mixture_mode]
    example_batch = jax.tree_util.tree_map(lambda x: x[:1], example_batch)
    for k in ['value', 'action', 'observation']:
        example_batch[k] = jax.tree_util.tree_map(reshape_fn, example_batch[k])
        example_batch[k] = jax.tree_util.tree_map(lambda x: x[0], example_batch[k])

    # task_reshape_fn = task_reshape_fns[traj_mixture_mode]
    # example_batch['task'] = jax.tree_util.tree_map(lambda x: x[:1], example_batch['task'])
    # example_batch['task'] = jax.tree_util.tree_map(task_reshape_fn, example_batch['task'])

    print(jax.tree_util.tree_map(jnp.shape, example_batch))

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    del config["model"]["task_tokenizers"]["language"]
    
    ###
    config["model"]["task_tokenizers"]["primary"] = ModuleSpec.create(
        TubletTokenizer,
        task_keys=["image_primary"],
    )
    config["model"]["observation_tokenizers"]["primary"] = ModuleSpec.create(
        TubletTokenizer,
        shapes=(1, 16, 16),
        obs_keys=["image_primary"],
    )
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    # Fully override the old action head with a new one (for smaller changes, you can use update_module_config)
    del config["model"]["heads"]["action"]
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        pred_horizon=1,
        action_dim=10,
        readout_key="readout_action",
    )
    if discrete_action:
        config["model"]["heads"]["value"] = ModuleSpec.create(
            DiscreteActionHead,
            action_dim=1,
            vocab_size=16,
            # normalization_type="normal",
            low=-64,#-1.5,
            high=320,#1.5,
            readout_key="readout_value",
        )
    else:
        config["model"]["heads"]["value"] = ModuleSpec.create(
            MSEActionHead,
            max_action=320,
            pred_horizon=1,
            action_dim=1,
            readout_key="readout_value",
        )
    config["model"]["readouts"] = {
        "action": 1,
        "value": 1,
    }

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=None, #dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 5e-4, 100), optax.constant_schedule(5e-4)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    # if FLAGS.freeze_transformer:
    #     frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    vmap_axes = ({
        'observation': 0,
        'task': None,
        'action': 0,
        'value': 0,
    },)

    @jax.jit
    def get_values(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)

        dummy_obs = jax.tree_util.tree_map(lambda x: x[0], batch['observation'])
        params = state.model.params
        cached_transformer_embeddings, cache_vars = model.module.apply(
            {"params": params},
            dummy_obs,
            batch["task"],
            jnp.zeros_like(dummy_obs["pad_mask"]),
            train=False,
            rngs={"dropout": dropout_rng},
            method='octo_transformer',
            mutable=["cache", "metadata"],
        )
        
        def vmapped_apply(obs):
            (transformer_embeddings, head_outputs), _ = state.model.module.apply(
                {"params": params, **cache_vars},
                obs,
                {},
                jnp.ones_like(obs["pad_mask"]),
                train=False,
                rngs={"dropout": dropout_rng},
                mutable=["cache", "metadata"],
            )
            return head_outputs

        head_outputs = jax.vmap(vmapped_apply, 0)(batch['observation'])

        return head_outputs['value']

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    def visualize_batch(state, batch):
        rand_id = np.random.randint(batch['observation']['image_primary'].shape[0])
        reshape_fn = batch_reshape_fns[traj_mixture_mode]
        batch = jax.tree_util.tree_map(lambda x: x[rand_id, None], batch)
        for k in ['value', 'action', 'observation']:
            batch[k] = jax.tree_util.tree_map(reshape_fn, batch[k]) # (S, N, 1, 1, 224, 224, 3)
        task_images = batch['task']['image_primary'] # (N, S // 64, 224, 224, 3)
        obs_images = batch['observation']['image_primary'] # (S, N, 1, 1, 224, 224, 3)

        values = get_values(state, batch)

        image_grid = np.stack([task_images[0, :, ::2, ::2], obs_images[::32, 0, 0, 0, ::2, ::2]], axis=0) # (S, 1, 1, 1, 224, 224, 3) -> (2, S // 64, 112, 112, 3)
        image_grid = np.moveaxis(image_grid, 2, 1).reshape(224, -1, 3)
        image_grid = np.moveaxis(inv_normalize(torch.moveaxis(torch.tensor(image_grid), -1, -3)).numpy(), -3, -1)
        fig, axs = plt.subplots(2, figsize=(8,8))
        axs[0].imshow(image_grid)
        axs[1].plot(np.arange(len(values)), batch['value'][:, rand_id].flatten(), color='blue')
        if discrete_action:
            values = values.reshape((values.shape[0], values.shape[-1]))
            axs[1].imshow(values.T, extent=[0, len(values), -64, 320], interpolation='nearest', cmap='Blues', aspect='auto', origin='lower')
        else:
            axs[1].plot(np.arange(len(values)), values, color='red')
        return fig

    def loss_fn_process_task(params, batch, rng, train=True):
        dummy_obs = jax.tree_util.tree_map(lambda x: x[0], batch['observation'])
        cached_transformer_embeddings, cache_vars = model.module.apply(#bound_module.octo_transformer(
            {"params": params},
            dummy_obs,
            batch["task"],
            jnp.zeros_like(dummy_obs["pad_mask"]),
            rngs={"dropout": rng},
            train=train,
            method='octo_transformer',
            mutable=["cache", "metadata"],
        )
        return cache_vars

    def loss_fn_process_obs(params, cache_vars, batch, rng, train=True):
        def vmapped_loss_fn(subbatch):
            transformer_embeddings, _ = model.module.apply(
                {"params": params, **cache_vars},
                subbatch['observation'],
                {},
                subbatch['observation']["pad_mask"],
                rngs={"dropout": rng},
                train=train,
                method='octo_transformer',
                mutable=["cache"],
            )
            loss, metrics = model.module.apply(
                {"params": params},
                transformer_embeddings, # Action head knows to pull out the action readout_key
                subbatch,
                subbatch["observation"]["pad_mask"],
                rngs={"dropout": rng},
                train=train,
                method="loss",
            )
            return loss, metrics

        loss, metrics = jax.vmap(
            vmapped_loss_fn,
            in_axes=0,
            out_axes=0,
        )(batch)
        loss = jnp.sum(loss, axis=0)
        metrics = jax.tree_util.tree_map(partial(jnp.mean, axis=0), metrics)
        return loss, metrics

    @partial(jax.jit, static_argnames=['traj_mixture_mode'])
    def train_step(state, batch, traj_mixture_mode):
        rng, dropout_rng = jax.random.split(state.rng)

        """'observation': {
            'image_primary': obs, # (N, S, 224, 224, 3)
            'proprio': batch['proprio'], # (N, S, T, p)
            'pad_mask': jnp.ones((obs_flat.shape[:2])) # (N, S)
        },
        'task': {
            'image_primary': obs[:, ::32], # (N, S, 224, 224, 3)
        },  
        'action': batch['action'], # (N, S, T, a)
        'value': jnp.broadcast_to((3 * 0.998^jnp.arange(seq_len) - 2)[None, ::-1, None, None], (batch_size, seq_len, 1, 1)) # (N, S, T, 1)"""
        reshape_fn = batch_reshape_fns[traj_mixture_mode]
        # task_reshape_fn = task_reshape_fns[traj_mixture_mode]
        for k in ['value', 'action', 'observation']:
            batch[k] = jax.tree_util.tree_map(reshape_fn, batch[k])

        # TODO: handle mixed tasks

        cache_vars, vjp_fn = jax.vjp(loss_fn_process_task,
            state.model.params, batch, dropout_rng, True
        )

        def scanned_grads(grads, subbatch):
            (loss, metrics), single_grads = jax.value_and_grad(loss_fn_process_obs, has_aux=True, argnums=(0, 1), allow_int=True)(
                state.model.params, cache_vars, subbatch, rng, train=True
            )
                
            return jax.tree_util.tree_map(lambda x, y: x if x.dtype == jax.dtypes.float0 else x + y, grads, single_grads), metrics
        
        # we're doing differentiation in this two-step process to force jax to backprop after each subbatch because we don't have enough memory for the whole batch at once
        scanned_batch = {k: v for k, v in batch.items() if k != "task"}
        scanned_batch = jax.tree_util.tree_map(lambda x: x.reshape(32, -1, *x.shape[1:]), scanned_batch)
        (obs_param_grads, cache_grads), metrics = jax.lax.scan(
            scanned_grads,
            jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x, dtype=jax.dtypes.float0) if isinstance(x, int) or jnp.isdtype(x.dtype, 'integral') else jnp.zeros_like(x),
                (state.model.params, cache_vars)),
            scanned_batch)
        metrics = jax.tree_util.tree_map(partial(jnp.mean, axis=0), metrics)
        all_grads = vjp_fn(cache_grads)
        task_param_grads = all_grads[0]
        grads = jax.tree_util.tree_map(lambda x, y: x + y, obs_param_grads, task_param_grads)

        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, metrics

    # run finetuning loop
    logging.info("Starting finetuning...")
    # with jax.profiler.trace("./tensorboard"):
    for i in tqdm.tqdm(range(5000), total=5000, dynamic_ncols=True):
        batch = next(train_data_iter)
        batch = jax.device_put(batch, sharding)
        if traj_mixture_mode == 'same_task':
            batch['task']['image_primary'] = batch['task']['image_primary'][np.random.permutation(np.arange(batch['task']['image_primary'].shape[0]))]
        train_state, update_info = train_step(train_state, batch, traj_mixture_mode)
        
        if (i + 1) % 20 == 0:
            update_info = jax.device_get(update_info)
            value_plot = wandb.Image(visualize_batch(train_state, batch))
            update_info['value_plot'] = value_plot
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )

        if (i + 1) % 1000 == 0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path="/nfs/nfs2/users/derekguo/experiment_output/fsoft_octo")        
            _, val_info = train_step(train_state, batch, traj_mixture_mode)
            val_info = jax.device_get(val_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"validation": val_info}, sep="/"),
                step=i,
            )

    train_state.model.save_pretrained(step=5000, checkpoint_path="/nfs/nfs2/users/derekguo/experiment_output/fsoft_octo/same_traj")        

if __name__ == "__main__":
    # parsing args
    args = tyro.cli(ExperimentConfig)

    if args.load_config is not None: 
        print("loading configs from file: ", args.load_config)
        assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
        args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader) 

    # creating the output directory and logging directory
    if args.logging_cfg.log_name is not None: 
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # dump the args into a yaml file 
    with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

    main(args)
