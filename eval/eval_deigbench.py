import os
import sys
import json
import random
import argparse
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
import torchvision
import supervision as sv
from PIL import Image
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from transformers import CLIPModel, CLIPProcessor
from pycocotools import mask as mask_utils
from groundingdino.util.inference import Model
from qwen_vl_utils import process_vision_info

from inference import alpha_generator, load_ckpt, prepare_batch, set_alpha_scale
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.euler import EulerSampler
from ldm.util import instantiate_from_config

device = "cuda"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "eval/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"

# Global variables for lazy loading
grounding_dino_model = None
qwen_model = None
qwen_processor = None
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

def load_eval_models(need_clip=False, need_qwen=True):
    """Lazy load evaluation models"""
    global grounding_dino_model, qwen_model, qwen_processor, clip_model, clip_processor
    
    if grounding_dino_model is None:
        print("Loading GroundingDINO model...")
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                                      model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    
    if need_qwen and qwen_model is None:
        print("Loading Qwen2.5-VL model...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "checkpoints/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        qwen_processor = AutoProcessor.from_pretrained("checkpoints/Qwen2.5-VL-7B-Instruct")
    
    if need_clip and clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained('checkpoints/clip/').cuda().eval()
        clip_processor = CLIPProcessor.from_pretrained('checkpoints/clip/')

def calc_clip_score(image, prompt, need_template=False):
    """Calculate CLIP score"""
    prompt_list = []
    if need_template:
        for text_template in imagenet_templates:
            filled_text = text_template.format(prompt)
            prompt_list.append(filled_text)
    else:
        prompt_list.append(prompt)

    inputs = clip_processor(text=prompt_list, images=image, return_tensors='pt', padding=True, truncation=True)
    for key in inputs.keys():
        inputs[key] = inputs[key].cuda().detach()

    outputs = clip_model(**inputs)
    torch.cuda.empty_cache()
    logits_per_image = outputs.logits_per_image

    return torch.mean(logits_per_image).cpu()

def get_subject_and_object_using_GLM(image, prompt):
    """Use Qwen model to check if image matches description"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Does this image meet the description '{}'? If so, please answer a single 'yes'; \
                 if not, please answer a single 'no'".format(prompt)},
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text

def check_on_image(image=None, label=None, prompt=None, gt_bbox=None, attr=None, box_t=0.25, text_t=0.25, miou_threshold=0.5):
    """Check object detection and attributes in image"""
    attr_flag = 0
    success_flag = 0

    CLASSES = [label]
    BOX_THRESHOLD = box_t
    TEXT_THRESHOLD = text_t
    NMS_THRESHOLD = 0.8

    # Detect objects
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

    # Use Qwen model to verify attributes
    x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    cv_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    pil_image = pil_image.crop((x, y, x + w, y + h))
    response = get_subject_and_object_using_GLM(pil_image, prompt)

    if "yes" in response[0]:
        attr_flag = 1
    else:
        attr_flag = 0

    return success_flag, attr_flag, miou

def run_evaluation(image_dir, coco_context, args):
    """Run evaluation"""
    print("Evaluation Start")
    
    # Load evaluation models
    load_eval_models(need_clip=args.need_clip_score, need_qwen=True)
    
    if not os.path.exists(image_dir):
        print('There is no picture!!!!')
        return
    
    miou_threshold = args.miou_threshold

    need_check_instance = args.need_sucess_ratio or args.need_local_clip or args.need_instance_sucess_ratio or args.need_miou_score or args.need_attribute_stats
    need_segment_instance = args.need_sucess_ratio or args.need_instance_sucess_ratio or args.need_miou_score or args.need_attribute_stats
    need_crop_instance = args.need_local_clip

    # Initialize statistics variables
    clip_record = 0.0
    clip_count = 0
    loca_clip_record = 0.0
    loca_clip_count = 0
    miou_record = 0.0
    miou_count = 0
    sucess_record = 0.0
    sucess_count = 0
    inst_suceess_count = 0
    inst_count = 0

    # Person color statistics
    person_color_stats = {
        1: {"total": 0, "correct": 0},
        2: {"total": 0, "correct": 0},
        3: {"total": 0, "correct": 0}
    }
    
    # Object four-strategy statistics
    object_strategy_stats = {
        "color_only": {"total": 0, "correct": 0},
        "color_material": {"total": 0, "correct": 0},
        "color_texture": {"total": 0, "correct": 0},
        "color_material_texture": {"total": 0, "correct": 0}
    }

    metric_result = {}

    for item in tqdm(coco_context, desc="Evaluating"):
        caption = item['caption']
        
        # Read image
        image_abs_path = os.path.join(image_dir, item['file_name'][:-4] + '.jpg')
        image = cv2.imread(image_abs_path)
        if image is None:
            continue
        if image.shape[0] != 512:
            image = cv2.resize(image, dsize=(512, 512))

        if args.need_clip_score:
            clip_score = calc_clip_score(image, caption)
            clip_record = clip_record + clip_score.item()
            clip_count = clip_count + 1

        if need_check_instance:
            if need_segment_instance:
                sucess_obj_per_image = 1
                sucess_attr_per_image = 1
                for i in item['instances']:
                    label_w_attr = i['description']
                    label = i['category']
                    gt_bbox = np.array(i['bbox']) * 512
                    
                    eval_info = i.get('eval_info', {})
                    
                    if args.need_sucess_ratio or args.need_instance_sucess_ratio or args.need_miou_score or args.need_attribute_stats:
                        sucess_obj, sucess_attr, miou = check_on_image(image, label, label_w_attr, gt_bbox, attr=None, miou_threshold=miou_threshold)
                        sucess_obj_per_image = sucess_obj_per_image * sucess_obj
                        sucess_attr_per_image = sucess_attr_per_image * sucess_attr

                        # Detailed attribute statistics logic
                        if args.need_attribute_stats and eval_info:
                            eval_type = eval_info.get('type', '')
                            eval_metrics = eval_info.get('eval_metrics', {})
                            
                            if eval_type == "person":
                                color_info = eval_metrics.get('color', {})
                                num_colors = color_info.get('num_colors', 0)
                                if num_colors in person_color_stats:
                                    person_color_stats[num_colors]["total"] += 1
                                    if sucess_obj and sucess_attr:
                                        person_color_stats[num_colors]["correct"] += 1
                                        
                            elif eval_type == "object":
                                material_info = eval_metrics.get('material', {})
                                texture_info = eval_metrics.get('texture', {})
                                has_material = material_info.get('has_material', False)
                                has_texture = texture_info.get('has_texture', False)
                                
                                if not has_material and not has_texture:
                                    strategy_key = "color_only"
                                elif has_material and not has_texture:
                                    strategy_key = "color_material"
                                elif not has_material and has_texture:
                                    strategy_key = "color_texture"
                                else:
                                    strategy_key = "color_material_texture"
                                
                                object_strategy_stats[strategy_key]["total"] += 1
                                if sucess_obj and sucess_attr:
                                    object_strategy_stats[strategy_key]["correct"] += 1

                    if args.need_miou_score:
                        miou_record = miou_record + miou
                        miou_count = miou_count + 1

                    if args.need_instance_sucess_ratio:
                        inst_count = inst_count + 1
                        if sucess_obj and sucess_attr:
                            inst_suceess_count = inst_suceess_count + 1
                        
                if args.need_sucess_ratio:
                    if sucess_obj_per_image * sucess_attr_per_image == 1:
                        sucess_record = sucess_record + 1
                    sucess_count = sucess_count + 1

            if need_crop_instance:
                for i in item['instances']:
                    inst_label = i['description']
                    inst_bbox = np.array(i['bbox'])
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
        metric_result['sucess_ratio'] = sucess_ratio
        print(f'SUCCESS RATIO: {sucess_ratio}')

    if args.need_instance_sucess_ratio and inst_count > 0:
        inst_sr = inst_suceess_count / inst_count
        metric_result['inst_sucess_ratio'] = inst_sr
        print(f'INST SUCCESS RATIO: {inst_sr}')

    if args.need_miou_score and miou_count > 0:
        miou_score = miou_record / miou_count
        metric_result['miou'] = miou_score
        print(f'MIOU SCORE: {miou_score}')

    # Detailed attribute statistics output
    if args.need_attribute_stats:
        print("\n" + "=" * 80)
        print("Detailed Attribute Accuracy Statistics")
        print("=" * 80)
        
        # Person color accuracy (C1/C2/C3)
        print("\nMetric 1 - Person Color Accuracy:")
        total_persons = 0
        total_person_correct = 0
        person_accuracy_output = {}
        for num_colors in [1, 2, 3]:
            total = person_color_stats[num_colors]["total"]
            correct = person_color_stats[num_colors]["correct"]
            accuracy = (correct / total) if total > 0 else 0
            total_persons += total
            total_person_correct += correct
            level_name = f"C{num_colors}"
            person_accuracy_output[level_name] = round(accuracy, 4)
            print(f"  {level_name}: total={total}, correct={correct}, MAA={accuracy:.4f}")

        overall_person_accuracy = (total_person_correct / total_persons) if total_persons > 0 else 0
        print(f"  Overall person color accuracy: {overall_person_accuracy:.4f} ({total_person_correct}/{total_persons})")

        # Object four-strategy accuracy (L1/L2/L3/L4)
        print("\nMetric 2 - Object Four Description Strategy Accuracy:")
        total_objects_strategy = 0
        total_strategy_correct = 0
        object_accuracy_output = {}
        
        strategy_keys = ["color_only", "color_material", "color_texture", "color_material_texture"]
        for i, strategy_key in enumerate(strategy_keys, 1):
            total = object_strategy_stats[strategy_key]["total"]
            correct = object_strategy_stats[strategy_key]["correct"]
            accuracy = (correct / total) if total > 0 else 0
            total_objects_strategy += total
            total_strategy_correct += correct
            level_name = f"L{i}"
            object_accuracy_output[level_name] = round(accuracy, 4)
            print(f"  {level_name}: total={total}, correct={correct}, MAA={accuracy:.4f}")

        overall_strategy_accuracy = (total_strategy_correct / total_objects_strategy) if total_objects_strategy > 0 else 0
        print(f"  Overall object strategy accuracy: {overall_strategy_accuracy:.4f} ({total_strategy_correct}/{total_objects_strategy})")

        # Save detailed statistics results
        metric_result['person_color_stats'] = person_color_stats
        metric_result['object_strategy_stats'] = object_strategy_stats
        
        # Save accuracy for each class (without percentage)
        metric_result['person_accuracy'] = person_accuracy_output
        metric_result['object_accuracy'] = object_accuracy_output
        
        metric_result['overall_accuracies'] = {
            "person_color_accuracy": round(overall_person_accuracy, 4),
            "object_strategy_accuracy": round(overall_strategy_accuracy, 4),
            "person_total": total_persons,
            "person_correct": total_person_correct,
            "object_total": total_objects_strategy,
            "object_correct": total_strategy_correct
        }

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
        if config.no_plms:
            sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                                  set_alpha_scale=set_alpha_scale)
            steps = 250
        else:
            sampler = EulerSampler(diffusion, model, 
                            alpha_generator_func=alpha_generator_func,
                            set_alpha_scale=set_alpha_scale)
            steps = 30

        # - - - - - inpainting related - - - - - #
        inpainting_mask = z0 = None  # used for replacing known region in diffusion process
        inpainting_extra_input = None  # used as model input

        # - - - - - input for interactdiffusion - - - - - #
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
        output_folder = os.path.join(args.folder)
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
    parser.add_argument("--folder", type=str, default="generation_samples/deigbench", help="root folder for output")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path to config")
    parser.add_argument("--community", type=str, default=None, help="path to community model")
    parser.add_argument("--fp16", action='store_true', help="use FP16 for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
    
    # Evaluation related parameters
    parser.add_argument("--run_eval", action='store_true', help="Run evaluation after inference")
    parser.add_argument('--need_clip_score', action='store_true', help="Calculate CLIP score")
    parser.add_argument('--need_sucess_ratio', action='store_true', help="Calculate success ratio")
    parser.add_argument('--need_local_clip', action='store_true', help="Calculate local CLIP score")
    parser.add_argument('--need_miou_score', action='store_true', help="Calculate MIOU score")
    parser.add_argument('--need_instance_sucess_ratio', action='store_true', help="Calculate instance success ratio")
    parser.add_argument('--need_attribute_stats', action='store_true', help="Detailed attribute statistics")
    parser.add_argument('--miou_threshold', type=float, default=0.5, help="MIOU threshold")
    parser.add_argument('--metric_name', type=str, default='deigbench', help="Evaluation metric name")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert args.batch_size == 1, 'now only support bs=1, because every image saved with same name'

    with open("eval/deig_bench_coco.json", 'r') as file:
        coco_data = json.load(file)
    
    # Compatible with old and new formats
    if 'results' in coco_data:
        res = coco_data['results']
    else:
        res = coco_data
    
    n_imgs_per_job = len(res) // args.num_jobs + 1
    start_index = args.job_index * n_imgs_per_job
    end_index = min((args.job_index + 1) * n_imgs_per_job, len(res))
    print("start_index: ", start_index)
    print("end_index: ", end_index)


    meta_list_new = []
    for r in res[start_index:end_index]:
        object_phrases = []
        object_boxes = []
        res_l = r['instances']
        for e in res_l:
            object_phrases.append(e['description'])
            object_boxes.append(e['bbox'])
        caption = ', '.join(object_phrases)

        meta_list_new.append(dict(
            ckpt="checkpoints/model.pth",
            prompt=caption,
            object_phrases=object_phrases,
            object_boxes=object_boxes,
            alpha_type=[1.0, 0.0, 0.0],
            img_id='',
            file_name= r['file_name']
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
        run_evaluation(args.folder, res[start_index:end_index], args)