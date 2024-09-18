from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Please analyze the attached image. Each pixel's RGB values represent the ASCII codes for three consecutive characters. Read the pixel values from left to right, top to bottom, and convert the RGB channel values back to corresponding ASCII characters to reconstruct the hidden message. Stop decoding when you encounter a pixel where all three channels are zero."
image_file = "/data/chenrenmiao/encoded_image_336x336.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)