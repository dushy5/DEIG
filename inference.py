import os
import argparse
import random
from functools import partial

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file

from ldm.util import instantiate_from_config, draw_boxes_on_image
from ldm.models.diffusion.euler import EulerSampler

device = "cuda"
clip_text_feature_dict = dict()


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1, 0, 0]

    assert len(type) == 3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length

    assert len(alphas) == length

    return alphas


def load_ckpt(ckpt_path, config_path, use_fp16=False, use_community=None):
    saved_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = OmegaConf.load(config_path)

    if use_community:
        # change community model here
        from huggingface_hub import hf_hub_download
        community_model_path = hf_hub_download(
            repo_id="moiu2998/mymo",                     
            filename="realisticVisionV60B1_v51VAE.safetensors",
            cache_dir="./checkpoints"
        )
        community_model = load_file(community_model_path)

        # diffusion
        diffusion_prefix = "model.diffusion_model."
        community_model_diffusion = {k[len(diffusion_prefix):]: v for k, v in community_model.items() if k.startswith(diffusion_prefix)}
        for k in community_model_diffusion:
            if k in saved_ckpt['model']:
                saved_ckpt['model'][k] = community_model_diffusion[k]

        # VAE
        vae_prefix = "first_stage_model."
        community_model_vae = {k[len(vae_prefix):]: v for k, v in community_model.items() if k.startswith(vae_prefix)}
        for k in community_model:
            if k in saved_ckpt['autoencoder']:
                saved_ckpt['autoencoder'][k] = community_model_vae[k]

    dtype = torch.float16 if use_fp16 else torch.float32

    model = instantiate_from_config(config['model']).to(device, dtype=dtype).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device, dtype=dtype).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device, dtype=dtype)
    text_encoder = instantiate_from_config(config['text_encoder']).to(device, dtype=dtype).eval()

    model.load_state_dict(saved_ckpt['model'])
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    return model, autoencoder, text_encoder, diffusion, config


def get_attmask_w_box(attn_masks, box, idx=None):
    x1, y1, x2, y2 = int(np.round(box[0] * 512)), int(np.round(box[1] * 512)), int(np.round(box[2] * 512)), int(np.round(box[3] * 512))
    attn_masks[y1:y2, x1:x2] = 1
    return attn_masks


@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30, device=device, half=False, encoder=None):

    object_phrases, object_images = meta.get("object_phrases"), meta.get("object_images")
    object_images = [None] * len(object_phrases) if object_images is None else object_images
    object_phrases = [None] * len(object_images) if object_phrases is None else object_phrases

    object_boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs, 1)
    object_text_embeddings = torch.zeros(max_objs, 128, 2048)
    object_masks = torch.zeros(max_objs, 512, 512)

    object_text_features = []

    for object_phrase \
            in object_phrases:
        object_text_features.append(encoder(object_phrase, max_length=128))


    for idx, (object_box,
              object_text_feature
              ) \
            in enumerate(zip(meta['object_boxes'],
                             object_text_features)):
        if idx >= max_objs:
            break
        object_boxes[idx] = torch.tensor(object_box)
        masks[idx] = 1
        object_masks[idx] = get_attmask_w_box(object_masks[idx], object_box, idx)
        object_text_embeddings[idx] = object_text_feature
            
    inst_masks = object_masks
    if half:
        object_boxes = object_boxes.half()
        masks = masks.half()
        object_text_embeddings = object_text_embeddings.half()
        inst_masks = inst_masks.half()

    out = {
        "object_boxes": object_boxes.unsqueeze(0).repeat(batch, 1, 1),
        "masks": masks.unsqueeze(0).repeat(batch, 1, 1),
        "object_text_embeddings": object_text_embeddings.unsqueeze(0).repeat(batch, 1, 1, 1),
        "inst_masks": inst_masks.unsqueeze(0).repeat(batch, 1, 1, 1)}

    return batch_to_device(out, device)


@torch.no_grad()
def run(meta, starting_noise=None):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"], args.config, use_fp16=args.fp16, use_community=args.community)

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # - - - - - prepare batch - - - - - #
    batch = prepare_batch(meta, config.batch_size, encoder=text_encoder, half=args.fp16)

    # - - - - - generate prompt context - - - - - #
    context = text_encoder([meta["prompt"]] * config.batch_size, max_length=128)
    uc = text_encoder(config.batch_size * [""], max_length=128)
    if args.negative_prompt is not None:
        uc = text_encoder(config.batch_size * [args.negative_prompt], max_length=128)

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    sampler = EulerSampler(diffusion, model, 
                        alpha_generator_func=alpha_generator_func,
                        set_alpha_scale=set_alpha_scale)
    steps = 40
    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None  # used as model input

    # - - - - - input - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
        x=starting_noise,
        timesteps=None,
        context=context,
        grounding_input=grounding_input,
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,
    )

    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

    samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                  mask=inpainting_mask, x0=z0)
    
    samples_fake = autoencoder.decode(samples_fake)

    # - - - - - save - - - - - #
    output_folder = os.path.join(args.folder, meta["save_folder_name"])
    os.makedirs(output_folder, exist_ok=True)

    start = len(os.listdir(output_folder))
    image_ids = list(range(start, start + config.batch_size))
    print(image_ids)
    
    object_boxes = meta.get("object_boxes", [])
    object_phrases = meta.get("object_phrases", [])
    
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = str(int(image_id)) + '.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        
        if object_boxes and object_phrases:
            sample = draw_boxes_on_image(sample, object_boxes, object_phrases, "Rainbow-Party-2.ttf")
        
        sample.save(os.path.join(output_folder, img_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="generation_samples", help="root folder for output")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path to config")
    parser.add_argument("--community", action='store_true', help="use community model")
    parser.add_argument("--fp16", action='store_true', help="use FP16 for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str, default='longbody, lowres, distorted face, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    meta_list = [

        dict(
            ckpt="checkpoints/model.pth",
            prompt='donuts',
            object_phrases=[
                            "a brown donut",
                            "a blue donut",
                            "a brown donut",
                            "a green donut",
                            "a white donut",
                            "a blue donut",
                            ],
            object_boxes=[
                            [0.3203, 0.4702, 0.5610, 0.8793],
                            [0.0435, 0.4110, 0.2960, 0.7826],
                            [0.1685, 0.1838, 0.3904, 0.4869],
                            [0.0494, 0.1524, 0.2383, 0.4760],
                            [0.0000, 0.0938, 0.1680, 0.2975],
                            [0.2193, 0.0689, 0.3639, 0.2034],
                            ],
            alpha_type=[1.0, 0.0, 0.0],
            save_folder_name="examples"),

        dict(
            ckpt="checkpoints/model.pth",
            prompt='3 person in a grass land',
            object_phrases=['a man wearing red hoodie, white pants, pink shoes',
                            'A little girl wearing a green dress and yellow hat and black boots',
                            'a woman in a white shirt with red accents, blue shorts, white shoes and sunglasses'],
            object_boxes=[[0.08235677083333337, 0.11374205221861455, 0.4530277665043291, 0.9227247362012986],
                            [0.38538707386363646, 0.46547365395021634, 0.6072485457251083, 0.947075385551948],
                            [0.536902225378788, 0.11644767992424221, 0.8724000608766236, 0.9173134807900432]],
            alpha_type=[1.0, 0.0, 0.0],
            save_folder_name="examples"),

        dict(
            ckpt="checkpoints/model.pth",
            prompt="7 person standing on the ground",
            object_phrases=[
                "a person wearing a blue jacket and black pants",
                "a person wearing a red hoodie and gray pants",
                "a person wearing a green coat and beige pants with a black hat",
                "a person wearing a black jacket and jeans with a red hat",
                "a person wearing a brown sweater and khaki pants with a white hat",
                "a person wearing a purple shirt and dark blue jeans",
                "a person wearing a white T-shirt and black shorts with a blue cap"
            ],
            object_boxes=[
                [0.0161, 0.1246, 0.2095, 0.8673],
                [0.2312, 0.1313, 0.3962, 0.8673],
                [0.3962, 0.0623, 0.6424, 0.8510],
                [0.5951, 0.1151, 0.7899, 0.8794],
                [0.8115, 0.1435, 0.9860, 0.8821],
                [0.1459, 0.4127, 0.2731, 0.9674],
                [0.5288, 0.3830, 0.7452, 0.9484]
            ],
            alpha_type=[1.0, 0.0, 0.0],
            save_folder_name="examples"
        )
    ]
    if args.fp16:
        starting_noise = torch.randn(args.batch_size, 4, 64, 64, device=device, dtype=torch.float16)
    else:
        starting_noise = torch.randn(args.batch_size, 4, 64, 64, device=device, dtype=torch.float32)

    for meta in meta_list:
        run(meta, starting_noise)