from PIL import Image


for i in range(5):
    bg_img = Image.open("~/code/Context-Aware-Image-Inpainting/exp/source.jpg")
    obj_img = Image.open(f"~/code/Context-Aware-Image-Inpainting/images/dreambooth_dataset/dog6/dog{i}.jpeg")
    obj_img = obj_img.resize(bg_img.size)
    obj_mask = Image.open(f"~/code/Context-Aware-Image-Inpainting/exp/masks/mask{i}.png")
    # print(bg_img.size)
    # print(obj_img.size)
    # print(obj_mask.size)

    obj_mask = obj_mask.convert("L")

    result_img = Image.new("RGB", bg_img.size)
    for x in range(bg_img.width):
        for y in range(bg_img.height):
            if obj_mask.getpixel((x, y)) == 255: 
                # result_img.putpixel((x, y), (255,255,255))
                result_img.putpixel((x, y), obj_img.getpixel((x, y)))
            else:  
                result_img.putpixel((x, y), bg_img.getpixel((x, y)))


    result_img.save(f"../exp/merged_imgs/m{i}.jpg")