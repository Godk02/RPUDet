
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import os
 
model_id = "/home/qk/data/diffution_model/byq/model"
print("sc")

img_path = '/home/qk/data/diffution_model/byq/img_2000'
if not os.path.exists(img_path): os.mkdir(img_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device ='cuda'
print("good")
# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(
    model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
print("nice")
ddpm.to(device)

print("nice2")
for i in range(2000):
    # run pipeline in inference (sample random noise and denoise)
    image = ddpm().images[0]
    # save image
   
    image.save(os.path.join(img_path,f'{i}.png'))
  
    #image.convert('L').save(os.path.join(img_path, f'{i}.png'))
  
    if i % 10 == 0: print(f"i={i}")
 