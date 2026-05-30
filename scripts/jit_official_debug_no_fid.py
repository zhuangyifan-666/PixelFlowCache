#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def debug_evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    import copy
    import math

    import numpy as np
    import torch
    from PIL import Image

    import main_jit  # type: ignore  # noqa: WPS433

    model_without_ddp.eval()
    world_size = main_jit.misc.get_world_size()
    local_rank = main_jit.misc.get_rank()
    num_steps = math.ceil(args.num_images / (batch_size * world_size))
    save_folder = Path(args.output_dir) / (
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method,
            model_without_ddp.steps,
            model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0],
            model_without_ddp.cfg_interval[1],
            args.num_images,
            args.img_size,
        )
    )
    print("Stage 0 debug evaluate: FID disabled; using first num_images ImageNet labels.")
    print("Save to:", save_folder)
    if main_jit.misc.get_rank() == 0:
        save_folder.mkdir(parents=True, exist_ok=True)

    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    class_label_gen_world = np.arange(args.num_images) % args.class_num
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))
        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = min(start_idx + batch_size, args.num_images)
        if start_idx >= args.num_images:
            continue
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_tensor = torch.tensor(labels_gen, dtype=torch.long, device="cuda")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_tensor)

        torch.distributed.barrier()
        sampled_images = ((sampled_images + 1) / 2).detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = start_idx + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)
            Image.fromarray(gen_img).save(save_folder / "{}.png".format(str(img_id).zfill(5)))

    torch.distributed.barrier()
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    torch.distributed.barrier()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    jit_dir = Path(os.environ.get("PFC_JIT_DIR", root / "third_party/JiT")).resolve()
    sys.path.insert(0, str(root / "scripts/jit_stubs"))
    sys.path.insert(0, str(jit_dir))
    os.chdir(jit_dir)

    import main_jit  # type: ignore  # noqa: WPS433

    # Official evaluate() computes FID whenever a TensorBoard writer exists and
    # then deletes the sample folder. Stage 0 only needs debug images.
    main_jit.SummaryWriter = lambda *args, **kwargs: None
    main_jit.evaluate = debug_evaluate

    args = main_jit.get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_jit.main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
