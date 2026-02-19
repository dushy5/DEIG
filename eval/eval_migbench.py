import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import argparse
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from inference import alpha_generator, load_ckpt, prepare_batch, set_alpha_scale
from ldm.models.diffusion.euler import EulerSampler
from ldm.util import instantiate_from_config

import cv2
import supervision as sv
import torchvision
from groundingdino.util.inference import Model
from segment_anything.segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as mask_utils
import groundingdino.datasets.transforms as T

device = "cuda"

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "eval/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"

grounding_dino_model = None
sam_predictor = None
clip_model = None
clip_processor = None

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def load_eval_models(need_clip=False):
    global grounding_dino_model, sam_predictor, clip_model, clip_processor
    
    if grounding_dino_model is None:
        print("Loading GroundingDINO model...")
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                                      model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    
    if sam_predictor is None:
        print("Loading SAM model...")
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device='cuda')
        sam_predictor = SamPredictor(sam)
    
    if need_clip and clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained('checkpoints/clip/').cuda().eval()
        clip_processor = CLIPProcessor.from_pretrained('checkpoints/clip/')

def calc_clip_score(image, prompt, need_template=False):
    prompt_list = []
    if need_template:
        for text_template in imagenet_templates:
            filled_text = text_template.format(prompt)
            prompt_list.append(filled_text)
    else:
        prompt_list.append(prompt)

    inputs = clip_processor(text=prompt_list, images=image, return_tensors='pt', padding=True)
    for key in inputs.keys():
        inputs[key] = inputs[key].cuda().detach()

    outputs = clip_model(**inputs)
    torch.cuda.empty_cache()
    logits_per_image = outputs.logits_per_image

    return torch.mean(logits_per_image).cpu()

def check_on_color_cv(image=None, class_name=None, color_dict=None, color_type=None, args=None, image_path=None):
    dist_image = np.empty(image.shape, image.dtype)
    dist_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV, dst=dist_image)
    if isinstance(color_dict, list):
        mask = np.zeros([512, 512], np.uint8)
        for color_dic in color_dict:
            lower = color_dic['Lower']
            upper = color_dic['Upper']
            result_mask = cv2.inRange(dist_image, lower, upper) / 255
            mask = np.logical_or(result_mask, mask).astype(np.int_)
        mask = mask * 255
    else:
        lower = color_dict['Lower']
        upper = color_dict['Upper']
        mask = cv2.inRange(dist_image, lower, upper)
    result_mask = mask
    return result_mask

def check_on_image(image=None, prompt=None, gt_bbox=None, attr=None, box_t=0.25, text_t=0.25, miou_threshold=0.5, args=None, image_path=None):
    color_dict = {
        'red': [{'Lower': np.array([0, 50, 70]), 'Upper': np.array([9, 255, 255])},
                {'Lower': np.array([159, 50, 70]), 'Upper': np.array([180, 255, 255])}],
        'blue': {'Lower': np.array([90, 50, 70]), 'Upper': np.array([128, 255, 255])},
        'yellow': {'Lower': np.array([25, 50, 70]), 'Upper': np.array([35, 255, 255])},
        'green': {'Lower': np.array([36, 50, 70]), 'Upper': np.array([89, 255, 255])},
        'black': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 30])},
        'white': {'Lower': np.array([0, 0, 221]), 'Upper': np.array([180, 43, 255])},
        'brown': {'Lower': np.array([6, 43, 35]), 'Upper': np.array([25, 255, 255])},
    }

    CLASSES = [prompt]
    BOX_THRESHOLD = box_t
    TEXT_THRESHOLD = text_t
    NMS_THRESHOLD = 0.8

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    if detections.xyxy.shape[0] > 0:
        pred_bbox = detections.xyxy
        min_x = np.maximum(pred_bbox[:, 0], gt_bbox[0])
        max_x = np.minimum(pred_bbox[:, 2], gt_bbox[2])
        min_y = np.maximum(pred_bbox[:, 1], gt_bbox[1])
        max_y = np.minimum(pred_bbox[:, 3], gt_bbox[3])
        iw = np.maximum(max_x - min_x, 0.)
        ih = np.maximum(max_y - min_y, 0.)
        insert_area = iw * ih
        union_area = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1]) + \
                     (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]) - insert_area
        iou = insert_area / union_area
        ovmax = np.max(iou)
        if ovmax < miou_threshold:
            return 0, 0, ovmax
        else:
            success_flag = 1
            miou = ovmax
    else:
        return 0, 0, 0.0

    def segment(sam_predictor_local, image_local, xyxy):
        sam_predictor_local.set_image(image_local)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor_local.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            maskk = np.asfortranarray(masks[index])
            maskk = mask_utils.encode(maskk)
            maskk['counts'] = maskk['counts'].decode('utf-8')
            result_masks.append(maskk)
        return result_masks

    detections.mask = segment(
        sam_predictor_local=sam_predictor,
        image_local=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=gt_bbox[None, :]
    )

    mask_obj = mask_utils.decode(detections.mask)
    color_dic = color_dict[attr]

    segment_mask = torch.from_numpy(mask_obj)
    detect_mask = torch.zeros(size=(512, 512))

    for mask_id in range(segment_mask.shape[2]):
        mask = segment_mask[:, :, mask_id]
        detect_mask = torch.logical_or(mask, detect_mask).int()

    image_mid = image * (detect_mask.unsqueeze(-1).detach().numpy())
    rev_mask = (1 - detect_mask).unsqueeze(-1).detach().numpy()
    color_bg = np.zeros([512, 512, 3]).astype(np.uint8) + 127
    image_mid = image_mid + rev_mask * color_bg
    image_mid = image_mid.astype(np.uint8)

    color_mask = check_on_color_cv(image_mid, prompt, color_dic, attr, args, image_path)
    color_mask = torch.from_numpy(color_mask)
    final_mask = torch.logical_and(detect_mask, color_mask).int()

    if torch.sum(detect_mask) == 0.0 or torch.sum(final_mask) / torch.sum(detect_mask) < 0.2:
        attr_flag = 0
        miou = 0.0
    else:
        attr_flag = 1

    return success_flag, attr_flag, miou

def run_evaluation(image_dir, coco_context, args):
    print("Evaluation Start")
    
    # Load evaluation models
    load_eval_models(need_clip=args.need_clip_score)
    
    if not os.path.exists(image_dir):
        print('There is no picture!!!!')
        return
    
    image_path_list = os.listdir(image_dir)
    miou_threshold = args.miou_threshold

    need_check_instance = args.need_sucess_ratio or args.need_local_clip or args.need_instance_sucess_ratio or args.need_miou_score
    need_segment_instance = args.need_sucess_ratio or args.need_instance_sucess_ratio or args.need_miou_score
    need_crop_instance = args.need_local_clip

    # Initialize statistics variables
    clip_record = 0.0
    clip_count = 0
    loca_clip_record = 0.0
    loca_clip_count = 0
    miou_record = 0.0
    miou_count = 0
    miou_level_record = [0.0, 0.0, 0.0, 0.0, 0.0]
    miou_level_count = [0, 0, 0, 0, 0]
    sucess_record = 0.0
    sucess_count = 0
    success_level_record = [0, 0, 0, 0, 0]
    sucess_level_count = [0, 0, 0, 0, 0]
    inst_suceess_count = 0
    inst_count = 0
    inst_success_level_count = [0, 0, 0, 0, 0]
    inst_level_count = [0, 0, 0, 0, 0]

    metric_result = {}

    for image_path in tqdm(image_path_list, desc="Evaluating"):
        # Read gt_label
        image_id = image_path.split('_')[-1].split('.')[0]
        if image_id not in coco_context:
            # Try using filename without extension as id
            image_id = os.path.splitext(image_path)[0]
        if image_id not in coco_context:
            continue
            
        coco_info = coco_context[image_id]
        caption = coco_info['caption']
        gt_bbox_list = coco_info['segment']

        # Read image
        image_abs_path = os.path.join(image_dir, image_path)
        image = cv2.imread(image_abs_path)
        if image is None:
            continue
        if image.shape[0] != 512:
            image = cv2.resize(image, dsize=(512, 512))

        level = len(gt_bbox_list) - 2

        if args.need_clip_score:
            clip_score = calc_clip_score(image, caption)
            clip_record = clip_record + clip_score.item()
            clip_count = clip_count + 1

        if need_check_instance:
            if need_segment_instance:
                sucess_obj_per_image = 1
                sucess_attr_per_image = 1
                for gt_instance in gt_bbox_list:
                    label_w_attr = gt_instance['label']
                    label = " ".join(label_w_attr.split(" ")[2:])
                    attr = label_w_attr.split(" ")[1]
                    gt_bbox = np.array(gt_instance['bbox']) * 512

                    if args.need_sucess_ratio or args.need_instance_sucess_ratio or args.need_miou_score:
                        sucess_obj, sucess_attr, miou = check_on_image(image, label, gt_bbox, attr, 
                                                                        miou_threshold=miou_threshold, 
                                                                        args=args, image_path=image_path)
                        sucess_obj_per_image = sucess_obj_per_image * sucess_obj
                        sucess_attr_per_image = sucess_attr_per_image * sucess_attr

                    if args.need_miou_score:
                        miou_record = miou_record + miou
                        miou_count = miou_count + 1
                        miou_level_record[level] = miou_level_record[level] + miou
                        miou_level_count[level] = miou_level_count[level] + 1
                    if args.need_instance_sucess_ratio:
                        inst_count = inst_count + 1
                        inst_level_count[level] = inst_level_count[level] + 1
                        if sucess_obj and sucess_attr:
                            inst_suceess_count = inst_suceess_count + 1
                            inst_success_level_count[level] = inst_success_level_count[level] + 1

                if args.need_sucess_ratio:
                    if sucess_obj_per_image * sucess_attr_per_image == 1:
                        sucess_record = sucess_record + 1
                        success_level_record[level] = success_level_record[level] + 1
                    sucess_count = sucess_count + 1
                    sucess_level_count[level] = sucess_level_count[level] + 1

            if need_crop_instance:
                for instance in gt_bbox_list:
                    inst_bbox = instance['bbox']
                    inst_label = instance['label']
                    cropped_image = image[int(512 * inst_bbox[1]):int(512 * inst_bbox[3]),
                                          int(512 * inst_bbox[0]):int(512 * inst_bbox[2]), :]
                    cropped_image = cv2.resize(cropped_image, (512, 512))

                    if args.need_local_clip:
                        local_clip_score = calc_clip_score(cropped_image, inst_label, need_template=True)
                        loca_clip_record = loca_clip_record + local_clip_score.item()
                        loca_clip_count = loca_clip_count + 1

    # Save evaluation results
    print(f'\nHere is the metric::')
    if args.need_clip_score and clip_count > 0:
        clip_score = clip_record / clip_count
        metric_result['clip_score'] = clip_score
        print(f'CLIP score : {clip_score}')

    if args.need_local_clip and loca_clip_count > 0:
        local_clip_score = loca_clip_record / loca_clip_count
        metric_result['local_clip_score'] = local_clip_score
        print(f'Local CLIP: {local_clip_score}')

    if args.need_sucess_ratio and sucess_count > 0:
        sucess_ratio = sucess_record / sucess_count
        sucess_level_ratio = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(5):
            if sucess_level_count[i] > 0:
                sucess_level_ratio[i] = success_level_record[i] / sucess_level_count[i]
        metric_result['sucess_ratio'] = sucess_ratio
        metric_result['success_level_ratio'] = sucess_level_ratio
        print(f'SUCCESS RATIO: {sucess_ratio}')
        print(f'SUCCESS LEVEL RATIO: {sucess_level_ratio}')

    if args.need_instance_sucess_ratio and inst_count > 0:
        inst_level_sr = [0.0, 0.0, 0.0, 0.0, 0.0]
        inst_sr = inst_suceess_count / inst_count
        for i in range(5):
            if inst_level_count[i] > 0:
                inst_level_sr[i] = inst_success_level_count[i] / inst_level_count[i]
        metric_result['inst_sucess_ratio'] = inst_sr
        metric_result['inst_level_sucess_ratio'] = inst_level_sr
        print(f'INST SUCCESS RATIO: {inst_sr}')
        print(f'INST Level SUCCESS RATIO: {inst_level_sr}')

    if args.need_miou_score and miou_count > 0:
        miou_level_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        miou_score = miou_record / miou_count
        for i in range(5):
            if miou_level_count[i] > 0:
                miou_level_score[i] = miou_level_record[i] / miou_level_count[i]
        metric_result['miou'] = miou_score
        metric_result['miou_level'] = miou_level_score
        print(f'MIOU SCORE: {miou_score}')
        print(f'MIOU LEVEL SCORE : {miou_level_score}')

    metric_result['metric_name'] = args.metric_name
    metric_result['image_path'] = image_dir

    result = json.dumps(metric_result, indent=2)
    output_file = f'./{args.folder}/metric_{args.metric_name}.json'
    with open(output_file, 'w') as output_f:
        output_f.write(result)
    print(f'\nEvaluation result saved to: {output_file}')
    print('Evaluation is Over!!!')

@torch.no_grad()
def run_batch(meta, starting_noise=None):
    # - - - - - prepare models - - - - - #
    print(f"Loading ckpt from {meta[0]['ckpt']}")
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta[0]["ckpt"], args.config, use_fp16=args.fp16, use_community=args.community)

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    for i in tqdm(range(len(meta))):
        # - -` - - - prepare batch - - - - - #
        batch = prepare_batch(meta[i], config.batch_size, encoder=text_encoder, half=args.fp16)

        # - - - - - generate prompt context - - - - - #
        context = text_encoder([meta[i]["prompt"]] * config.batch_size, max_length=128)
        uc = text_encoder(config.batch_size * [""], max_length=128)
        if args.negative_prompt is not None:
            uc = text_encoder(config.batch_size * [args.negative_prompt], max_length=128)

        # - - - - - sampler - - - - - #
        alpha_generator_func = partial(alpha_generator, type=meta[i].get("alpha_type"))
        sampler = EulerSampler(diffusion, model, 
                        alpha_generator_func=alpha_generator_func,
                        set_alpha_scale=set_alpha_scale)
        steps = 40

        # - - - - - inpainting related - - - - - #
        inpainting_mask = z0 = None  # used for replacing known region in diffusion process
        inpainting_extra_input = None  # used as model input

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
        output_folder = args.folder
        os.makedirs(output_folder, exist_ok=True)

        start = len(os.listdir(output_folder))
        image_ids = list(range(start, start + config.batch_size))
        
        for image_id, sample in zip(image_ids, samples_fake):
            img_name = meta[i]['file_name']
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(os.path.join(output_folder, img_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="/group/40009/shiyandu/training/interactdiffusion/generation_samples/before_mig_70k/mig_bench_coco_120k_2", help="root folder for output")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path to config")
    parser.add_argument("--community", type=str, default=None, help="path to community model")
    parser.add_argument("--fp16", action='store_true', help="use FP16 for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--seed", type=int, default=123, help="random seed for reproducibility")
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
    
    parser.add_argument("--run_eval", action='store_true', help="Run evaluation after inference")
    parser.add_argument('--need_clip_score', action='store_true', help="Calculate CLIP score")
    parser.add_argument('--need_sucess_ratio', action='store_true', help="Calculate success ratio")
    parser.add_argument('--need_local_clip', action='store_true', help="Calculate local CLIP score")
    parser.add_argument('--need_miou_score', action='store_true', help="Calculate MIOU score")
    parser.add_argument('--need_instance_sucess_ratio', action='store_true', help="Calculate instance success ratio")
    parser.add_argument('--miou_threshold', type=float, default=0.5, help="MIOU threshold")
    parser.add_argument('--metric_name', type=str, default='migbench', help="Evaluation metric name")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open("eval/mig_bench_coco.json", 'r') as file:
        res = json.load(file)
    keys = list(res.keys())

    # split the image_ids into num_jobs
    n_imgs_per_job = len(keys) // args.num_jobs + 1
    start_index = args.job_index * n_imgs_per_job
    end_index = min((args.job_index + 1) * n_imgs_per_job, len(keys))
    print("start_index: ", start_index)
    print("end_index: ", end_index)

    meta_list_new = []
    for i in keys[start_index:end_index]:
        r = res[i]
        object_phrases = []
        object_boxes = []
        res_l = r['segment']

        for e in res_l:
            object_phrases.append(e['label'])
            object_boxes.append(e['bbox'])

        caption = r['caption']

        meta_list_new.append(dict(
            ckpt="checkpoints/model.pth",
            prompt=caption,
            object_phrases=object_phrases,
            object_boxes=object_boxes,
            alpha_type=[1.0, 0.0, 0.0],
            img_id='',
            file_name= i + '.jpg'
        ))

    meta_list_filtered = []
    for meta_item in meta_list_new:
        output_path = os.path.join(args.folder, meta_item['file_name'])
        if not os.path.exists(output_path):
            meta_list_filtered.append(meta_item)
    
    # Skip inference if all images already exist
    if len(meta_list_filtered) == 0:
        print("All images already exist, skipping inference stage")
    else:
        starting_noise = torch.randn(args.batch_size, 4, 64, 64, device=device, dtype=torch.float16) if args.fp16 else torch.randn(args.batch_size, 4, 64, 64, device=device, dtype=torch.float32)
        # Run inference
        run_batch(meta_list_filtered, starting_noise)
    
    # Run evaluation after inference
    if args.run_eval:
        run_evaluation(args.folder, res, args)