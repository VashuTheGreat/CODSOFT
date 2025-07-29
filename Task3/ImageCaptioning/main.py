from transformers import BlipProcessor, BlipForConditionalGeneration

from PIL import Image



import requests


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img = Image.open("dog.webp")
inputs = processor(images=img, return_tensors="pt")


out = model.generate(**inputs)


print(processor.decode(out[0], skip_special_tokens=True))
