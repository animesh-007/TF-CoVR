import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from lavis.models import load_model_and_preprocess

from src.data.embs import VideoDataset
from src.model.internvid.internvideo.build_internvideo import build_internvideo
from src.model.internvid.Qformer import BertConfig, BertLMHeadModel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_Qformer(cls=0, num_query_token=32, vision_width=768, cross_attention_freq=2):
        # encoder_config = BertConfig.from_pretrained("bert-large-uncased")
        # encoder_config.encoder_width = vision_width
        # encoder_config.add_cross_attention = True
        # encoder_config.cross_attention_freq = cross_attention_freq
        # encoder_config.query_length = num_query_token

        # # Load BERT-Large instead of BERT-Base
        # Qformer = BertLMHeadModel.from_pretrained("bert-large-uncased", config=encoder_config)

        # # Adjust query tokens
        # query_tokens = nn.Parameter(
        #     torch.zeros(1, num_query_token, encoder_config.hidden_size)  # Now hidden_size=1024
        # )
        # query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        encoder_config.hidden_dropout_prob = 0.1
        encoder_config.attention_probs_dropout_prob = 0.1
        encoder_config.drop_path_list = [x.item() for x in torch.linspace(0, 0., encoder_config.num_hidden_layers)]
        # logger.info(f"Drop_path:{encoder_config.drop_path_list}")
        # logger.info(encoder_config)
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )                 
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens


@torch.no_grad()
def main(args):
    
    args.save_dir = args.save_dir.expanduser()
    args.video_dir = args.video_dir.expanduser()

    if args.model_type == "coco":
        save_dir = args.save_dir / f"internvid-vid-embs-large-all"
    else:
        save_dir = args.save_dir / f"internvid-vid-embs-{args.model_type}-all"
    save_dir.mkdir(exist_ok=True)

    # args.save_dir = args.save_dir.expanduser()
    # args.video_dir = args.video_dir.expanduser()
    
    # save_dir = args.save_dir / f"blip2-vid-embs-{args.model_type}-all"


    dataset = VideoDataset(
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

    # _, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_feature_extractor",
    #     model_type=args.model_type,
    #     is_eval=True,
    #     device=device,
    # )

    model = build_internvideo(cfg="/home/animesh/FineGym-CoVR-Github/src/model/internvid/internvideo/internvideo2_stage2_config_vision.py", model_path="/home/animesh/FineGym-CoVR-Github/checkpoint/InternVideo2-stage2_1b-224p-f4.pt")
    model.eval()
    model.to(device)

    Qformer_model, query_tokens = init_Qformer(num_query_token=32, vision_width=1408, cross_attention_freq=2)
    Qformer_model.to('cuda')
    Qformer_model.eval()
    
    # dataset.transform = vis_processors["eval"]
    # dataset.pixel_size = args.image_size

    for video_ids, f_idxs, frames in tqdm(loader):
        frames = frames.to(device)
        bs, nf, c, h, w = frames.shape
        # import pdb; pdb.set_trace()
        frm_feats = model.encode_vision(frames, test=True)[0]

        ref_img_atts = torch.ones(frm_feats.size()[:-1], dtype=torch.long).to('cuda')

        query_tokens = query_tokens.expand(frm_feats.shape[0], -1, -1).to('cuda')

        # import pdb; pdb.set_trace()
        query_output = Qformer_model.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=frm_feats.detach().clone(),
            encoder_attention_mask=ref_img_atts,
            # use_cache=True,
            return_dict=True,
        )

        # frm_feats = model.extract_features({"image": frames}, mode="image")
        # frm_feats = frm_feats.image_embeds_proj.view(bs, nf, 32, 256).cpu()

        for video_id, f_idx, frm_feat in zip(video_ids, f_idxs, query_output.last_hidden_state):
            # remove the features with f_idx=-1
            # frm_feat = frm_feat[f_idx > -1]
            # f_idx = f_idx[f_idx > -1]
            # if len(f_idx) == 0:
            #     continue
            # import pdb; pdb.set_trace()
            save_pth = save_dir / f"{video_id}.pth"
            if save_pth.exists():
                continue
            save_pth.parent.mkdir(exist_ok=True)

            torch.save(frm_feat, save_pth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=Path, required=True, help="Path to video directory"
    )
    parser.add_argument(
        "--save_dir", type=Path, default= "/home/animesh/FinePseudo-CoVR/embeddings2/internvid", help="Path to video directory"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model_type", type=str, default="coco", choices=["coco", "pretrain_vitL"]
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--frames_video", type=int, default=15)
    parser.add_argument("--todo_ids", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224, choices=[224, 364])
    args = parser.parse_args()

    main(args)
