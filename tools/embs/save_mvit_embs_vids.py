import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

# from lavis.models import load_model_and_preprocess
from src.model.finepseudo.network import R2Plus1D

from src.data.embs import VideoDataset, FinePseudoVideoDataset
from src.model.egovlpv2.video_utils import FrameLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation = {}
# a dict to store the activations
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach().cpu()
    return hook

@torch.no_grad()
def main(args):
    # if args.model_type == "coco":
    #     save_dir = args.save_dir.parent / f"finepseudo-vid-embs-large-all"
    # else:
    args.video_dir = Path(args.video_dir).expanduser()
    args.save_dir = Path(args.save_dir).expanduser()
    save_dir = args.save_dir / f"mvit_16x4-vid-embs-{args.frames_video}-all"
    
    save_dir.mkdir(exist_ok=True)

    # dataset = VideoDataset(
    #     video_dir=args.video_dir,
    #     todo_ids=args.todo_ids,
    #     num_shards=args.num_shards,
    #     shard_id=args.shard_id,
    #     frames_video=args.frames_video,
    #     save_dir=save_dir,
    #     image_size=args.image_size,
    # )

    dataset = FinePseudoVideoDataset(
        video_dir=args.video_dir,
        todo_ids=args.todo_ids,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        frames_video=args.frames_video,
        save_dir=save_dir,
        image_size=args.image_size,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print("Creating model")

    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_feature_extractor",
    #     model_type=args.model_type,
    #     is_eval=True,
    #     device=device,
    # )

    # visual_encoder = R2Plus1D(ckpt="/home/an663820/ICCV25/FinePseudo-CoVR/checkpoint/fg288r2plus1d_best_model_acc_72.0713_e96.pth")

    visual_encoder = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=True)
    visual_encoder.to(device)
    visual_encoder.eval()

    
    mvit_sequence_pool = visual_encoder.head.sequence_pool.register_forward_hook(getActivation('mvit_sequence_pool'))


    # dataset.transform = vis_processors["eval"]
    dataset.pixel_size = args.image_size

    # for video_ids, f_idxs, frames in tqdm(loader):
    for video_ids, video_raw in tqdm(loader):
        # frames = frames.to(device)
        video_raw = video_raw.to(device)
        # bs, nf, c, h, w = frames.shape
        # frames = frames.view(bs * nf, c, h, w)

        # frm_feats = model.extract_features({"image": frames}, mode="image")

        # import pdb; pdb.set_trace()
        video_raw_fts = visual_encoder(video_raw)
        # frm_feats = frm_feats.image_embeds_proj.view(bs, nf, 32, 256).cpu()

        video_raw_fts = video_raw_fts.cpu()

        # for video_id, f_idx, frm_feat in zip(video_ids, f_idxs, frm_feats):
        #     # remove the features with f_idx=-1
        #     frm_feat = frm_feat[f_idx > -1]
        #     f_idx = f_idx[f_idx > -1]
        #     if len(f_idx) == 0:
        #         continue
        # import pdb; pdb.set_trace()
        for video_id, video_raw_ft in zip(video_ids, activation["mvit_sequence_pool"]):
            save_pth = save_dir / f"{video_id}.pth"
            if save_pth.exists():
                continue
            save_pth.parent.mkdir(exist_ok=True)

            # torch.save(frm_feat, save_pth)
            torch.save(video_raw_ft, save_pth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=Path, default= "~/FinePseudo-CoVR/datasets/subactions", help="Path to video directory"
    )
    parser.add_argument(
        "--save_dir", type=Path, default= "~/FinePseudo-CoVR/embeddings", help="Path to video directory"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="coco", choices=["coco", "pretrain_vitL"]
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--frames_video", type=int, default=32)
    parser.add_argument("--todo_ids", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224, choices=[224, 364])
    args = parser.parse_args()

    main(args)
