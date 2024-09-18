import numpy as np
from PIL import Image

# 设置一个具体的随机种子，确保每次生成的图片都相同
random_seed = 1
np.random.seed(random_seed)  # 你可以选择任何整数作为种子

# 生成3x336x336的高斯随机数
image_data = np.random.rand(336, 336, 3) * 255

# 显示图像
image = Image.fromarray(image_data.astype(np.uint8))
image.save(f'./images/uniform_noise_image/{random_seed}.png')