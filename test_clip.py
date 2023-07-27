from PIL import Image
from clip_interrogator import Config, Interrogator

# open image and convert to RGB
# the conversion is necessary because the default mode is RGBA
image = Image.open("/Users/sergiohlb/Documents/madruga1.png").convert('RGB')
# create an interrogator with the default model
# 
# laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
# laion/CLIP-ViT-g-14-laion2B-s12B-b42K
# ===============================================================================
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
# interrogate the image and print the results
print(ci.interrogate(image))