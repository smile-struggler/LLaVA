from PIL import Image
import os

def blend_images(image_path1, image_path2, output_path):
	# 打开两张图片
	image1 = Image.open(image_path1)
	image2 = Image.open(image_path2)

	# 调整两张图片的大小，使它们相同
	image2 = image2.resize(image1.size)

	# 将两张图片以1:1的比例叠加在一起
	blended_image = Image.blend(image1, image2, alpha=0.5)

	# 保存结果图片
	blended_image.save(output_path)

# 示例用法
data_root = '/data/chenrenmiao/project/LLaVA/images/uniform_noise_image/'
blend_images(os.path.join(data_root,'unlimit_attack_image_0.png'), os.path.join(data_root,'unlimit_attack_image_1.png'), os.path.join(data_root,'unlimit_attack_image_0+1.png'))