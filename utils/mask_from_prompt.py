import torch
import cv2
import requests
import os.path

from repositories.clipseg.models.clipseg import CLIPDensePredT
from PIL import ImageChops, Image, ImageOps
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy


mask_prompt = "bear"

negative_mask_prompt = ""
mask_precision = 100.0
mask_padding = 0.0
brush_mask_mode = 'discard'


debug = False


def download_file(filename, url):
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
def pil_to_cv2(img):
    return (cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR))
def gray_to_pil(img):
    return (Image.fromarray(cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)))

def center_crop(img,new_width,new_height):
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    return(img.crop((left, top, right, bottom)))

def overlay_mask_part(img_a,img_b,mode):
    if (mode == 0):
        img_a = ImageChops.darker(img_a, img_b)
    else: img_a = ImageChops.lighter(img_a, img_b)
    return(img_a)

def process_mask_parts(these_preds,these_prompt_parts,mode,final_img = None):
    for i in range(these_prompt_parts):
        filename = f"mask_{mode}_{i}.png"
        plt.imsave(filename,torch.sigmoid(these_preds[i][0]))

        # TODO: Figure out how to convert the plot above to numpy instead of re-loading image
        img = cv2.imread(filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, bw_image) = cv2.threshold(gray_image, mask_precision, 255, cv2.THRESH_BINARY)

        if (mode == 0): bw_image = numpy.invert(bw_image)

        if (debug):
            print(f"bw_image: {bw_image}")
            print(f"final_img: {final_img}")

        # overlay mask parts
        bw_image = gray_to_pil(bw_image)
        if (i > 0 or final_img is not None):
            bw_image = overlay_mask_part(bw_image,final_img,mode)

        # For debugging only:
        if (debug): bw_image.save(f"processed_{filename}")

        final_img = bw_image

    return(final_img)

def get_mask(img_url):
    src_img = Image.open(img_url).convert("RGB")

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval();
    model_dir = "./repositories/clipseg/weights"
    os.makedirs(model_dir, exist_ok=True)
    d64_file = f"{model_dir}/rd64-uni.pth"
    d16_file = f"{model_dir}/rd16-uni.pth"
    delimiter_string = "|"
    
    # Download model weights if we don't have them yet
    if not os.path.exists(d64_file):
        print("Downloading clipseg model weights...")
        download_file(d64_file,"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni.pth")
        download_file(d16_file,"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth")
        # Mirror: 
        # https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth
        # https://github.com/timojl/clipseg/raw/master/weights/rd16-uni.pth
    
    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(d64_file, map_location=torch.device('cuda:7')), strict=False);			

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    img = transform(src_img).unsqueeze(0)

    prompts = mask_prompt.split(delimiter_string)
    prompt_parts = len(prompts)
    negative_prompts = negative_mask_prompt.split(delimiter_string)
    negative_prompt_parts = len(negative_prompts)

    # predict
    with torch.no_grad():
        preds = model(img.repeat(prompt_parts,1,1,1), prompts)[0]
        negative_preds = model(img.repeat(negative_prompt_parts,1,1,1), negative_prompts)[0]

    #tests
    if (debug):
        print("Check initial mask vars before processing...")
        print(f"p.image_mask: {p.image_mask}")
        print(f"p.latent_mask: {p.latent_mask}")
        print(f"p.mask_for_overlay: {p.mask_for_overlay}")

    if (brush_mask_mode == 1 and p.image_mask is not None):
        final_img = p.image_mask.convert("RGBA")
    else: final_img = None

    # process masking
    final_img = process_mask_parts(preds,prompt_parts,1,final_img)

    # process negative masking
    if (brush_mask_mode == 2 and p.image_mask is not None):
        p.image_mask = ImageOps.invert(p.image_mask)
        p.image_mask = p.image_mask.convert("RGBA")
        final_img = overlay_mask_part(final_img,p.image_mask,0)
    if (negative_mask_prompt): final_img = process_mask_parts(negative_preds,negative_prompt_parts,0,final_img)

    # Increase mask size with padding
    if (mask_padding > 0):
        aspect_ratio = p.init_images[0].width / p.init_images[0].height
        new_width = p.init_images[0].width+mask_padding*2
        new_height = round(new_width / aspect_ratio)
        final_img = final_img.resize((new_width,new_height))
        final_img = center_crop(final_img,p.init_images[0].width,p.init_images[0].height)

    return (final_img)
                

for i in range(0,5):
    img_url = f"~/code/Context-Aware-Image-Inpainting/images/bear_plushie/0{i}.jpg"
    mask_img = get_mask(img_url)
    mask_img.save("../exp/masks/"+mask_prompt+f"_mask{i}.png")


# img_url = f"~/code/Context-Aware-Image-Inpainting/images/512fruit.png"
# mask_img = get_mask(img_url)
# mask_img.save("../images/512fruit_mask.png")

