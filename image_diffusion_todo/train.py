import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import QuickDrawDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from model import DiffusionModule
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from utils import *
from unet import UNetModel

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    if args.use_cfg:
        save_dir = Path(f"results/cfg_diffusion-{args.sample_method}-{now}")
    else:
        save_dir = Path(f"results/diffusion-{args.sample_method}-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    image_resolution = config.image_resolution
    ds_module = QuickDrawDataModule(
        config.root_dir,
        batch_size=config.batch_size,
        num_workers=4,
        image_resolution=image_resolution,
        category=config.category,
        mode=config.coordinate_mode
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )

    # network = UNet(
    #     T=config.num_diffusion_train_timesteps,
    #     image_resolution=image_resolution,
    #     ch=128,
    #     ch_mult=[1, 2, 2, 2],
    #     attn=[1],
    #     num_res_blocks=4,
    #     dropout=0.1,
    #     use_cfg=args.use_cfg,
    #     cfg_dropout=args.cfg_dropout,
    #     num_classes=getattr(ds_module, "num_classes", None),
    # )

    if image_resolution == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_resolution == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_resolution == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_resolution == 96:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_resolution}")

    attention_ds = []
    for res in "16,8".split(","):
        attention_ds.append(image_resolution // int(res))

    network = UNetModel(
        in_channels=96,
        model_channels=96,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=attention_ds,
        dropout=0.0,
        channel_mult=channel_mult,
        num_classes=1,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                if args.use_cfg:  # Conditional, CFG training
                    samples = ddpm.sample(
                        1,
                        class_label=torch.randint(1, 2, (1,)).to(config.device),
                        return_traj=False,
                    )
                else:  # Unconditional training
                    samples = ddpm.sample(1, return_traj=False)

                pil_images = tensor_to_pil_image(samples)
                # pil_images = draw_full_images(tensor_to_strokes(samples))
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            img, pen_label, cls_label = next(train_it)
            img, pen_label, cls_label = img.to(config.device), pen_label.to(config.device), cls_label.to(config.device)

            if args.use_cfg:  # Conditional, CFG training
                loss = ddpm.get_loss(img, pen_label=pen_label, class_label=cls_label)
            else:  # Unconditional training
                loss = ddpm.get_loss(img, pen_label=pen_label)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--root_dir", type=str, default="../data")
    parser.add_argument("--category", type=str, default="cat")
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--coordinate_mode", type=str, default="direct")
    args = parser.parse_args()
    main(args)
