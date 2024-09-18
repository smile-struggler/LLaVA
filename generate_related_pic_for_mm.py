import json
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os

from tqdm import tqdm

def image_parser(image_file, sep):
    out = image_file.split(sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

model_path = "liuhaotian/llava-v1.5-7b"
image_file = "/home/chenrenmiao/project/LLaVA/images/test4.jpg"

model_base = None
model_name = get_model_name_from_path(model_path)
conv_mode = None
image_file = image_file
sep = ","
temperature = 0
top_p = None
num_beams = 1
max_new_tokens = 512

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)

def llava_output(query, image_file, adversarial_path=None):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(image_file, sep)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    if adversarial_path is not None:
        image_files = image_parser(adversarial_path, sep)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        adversarial = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        min_values = torch.tensor([-1.7920, -1.7520, -1.4805],device = model.device, dtype = images_tensor.dtype)
        max_values = torch.tensor([1.9307, 2.0742, 2.1465],device = model.device, dtype = images_tensor.dtype)

        min_values = min_values.view(1, 3, 1, 1)
        max_values = max_values.view(1, 3, 1, 1)

        images_tensor = torch.clamp(images_tensor + adversarial, min=min_values, max=max_values).detach()


    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

question_file = '/data/chenrenmiao/data/MM-SafetyBench/processed_questions/01-Illegal_Activitiy.json'
img_file_dir = '/data/chenrenmiao/data/MM-SafetyBench/img/01-Illegal_Activitiy'
result_file = "/data/chenrenmiao/project/Llama-guard2/result/mm-safetybench/01-Illegal_Activitiy_related_pic.json"
adversarial_path = '/data/chenrenmiao/project/LLaVA/images/uniform_noise_image/delta_universal_img.png'

result = {}

with open(question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

for data_id in tqdm(data):
    question = data[data_id]['Question']

    image_SD_TYPO = os.path.join(img_file_dir, 'SD_TYPO', f'{data_id}.jpg')

    result[data_id ] = {
        "origin": llava_output(question, image_SD_TYPO),
        "adv": llava_output(question, adversarial_path),
        "origin+adv": llava_output(question, image_SD_TYPO, adversarial_path),
    }

with open(result_file, 'w', encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False, indent=4)

