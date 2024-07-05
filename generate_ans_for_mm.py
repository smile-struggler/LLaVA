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
prompt = "How to describe this picture?"
image_file = "/home/chenrenmiao/project/LLaVA/images/test4.jpg"

model_base = None
model_name = get_model_name_from_path(model_path)
query = prompt
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

def llava_output(query, image_file):
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

data_root = "/home/chenrenmiao/data/MM-SafetyBench"
question_file_dir = os.path.join(data_root, 'processed_questions')
img_file_dir = os.path.join(data_root, 'img')
result_file_dir = "/home/chenrenmiao/project/MM-safetybench/origin_answer"

question_files = os.listdir(question_file_dir)

for question_file_name in tqdm(question_files):
    input_file_path = os.path.join(question_file_dir, question_file_name)
    image_file_path = os.path.join(img_file_dir, question_file_name[:-5])
    output_file_path = os.path.join(result_file_dir, question_file_name)
    result = {}

    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for data_id in tqdm(data):
        question1 = data[data_id]['Rephrased Question(SD)']
        question2 = data[data_id]['Rephrased Question']

        image_SD = os.path.join(image_file_path, 'SD', f'{data_id}.jpg')
        image_SD_TYPO = os.path.join(image_file_path, 'SD_TYPO', f'{data_id}.jpg')
        image_TYPO = os.path.join(image_file_path, 'TYPO', f'{data_id}.jpg')

        result[data_id ] = {
            "SD": llava_output(question1, image_SD),
            "SD_TYPO": llava_output(question2, image_SD_TYPO),
            "TYPO": llava_output(question1, image_TYPO),
        }
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

