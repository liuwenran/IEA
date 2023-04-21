import numpy as np 
import cv2
from PIL import Image, ImageDraw
import torch 
from torch.cuda.amp import autocast
import gradio as gr

from segment_anything import build_sam, SamAutomaticMaskGenerator 
from diffusers import StableDiffusionInpaintPipeline 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="/nvme/liuwenran/branches/liuwenran/dev-sdi/mmediting/resources/sam_model/sam_vit_h_4b8939.pth").to(device))
print('load segement anything model.')


sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    safety_checker=None,
    revision="fp16",
    torch_dtype=torch.float16,
)
sd_pipe = sd_pipe.to(device)
print('load sd model.')


def crop_image_pillow(img, divide=8):
    width, height = img.size
    if width > 960:
        img = img.resize((960, int(960 / width * height)))
        width, height = img.size
    if height > 960:
        img = img.resize((int(960 / height * width), 960))
        width, height = img.size

    # print(width, height)
    left = width - width // divide * divide
    top = height - height // divide * divide
    right = width
    bottom = height

    img = img.crop((left, top, right, bottom))

    return img


def crop_image(img, divide=8):
    height, width, c = img.shape
    if width > 960:
        img = cv2.resize(img, (960, int(960 / width * height)))
        width, height, c = img.shape
    if height > 960:
        img = cv2.resize(img, (int(960 / height * width), 960))
        width, height, c = img.shape

    top = height - height // divide * divide
    left = width - width // divide * divide
    img = img[top:, left:, :]
    return img


def segment(
    # clip_threshold: float,
    pathes: str,
    # segment_query: str,
    text_prompt: str,
):
    image_path = pathes['image']
    image = cv2.imread(image_path)
    image = crop_image(image)
    # import ipdb;ipdb.set_trace();
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    draw_mask_path = pathes['mask']
    draw_mask = cv2.imread(draw_mask_path)
    draw_mask = crop_image(draw_mask)
    gray = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
    draw_mask = gray > 0

    indices = []
    for i, mask in enumerate(masks):
        bitwise_res = cv2.bitwise_and(mask['segmentation'].astype('uint8'), draw_mask.astype('uint8'))
        if np.sum(bitwise_res) > 0:
            indices.append(i)

    segmentation_masks = []

    for seg_idx in indices:
        segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        segmentation_masks.append(segmentation_mask_image)

    original_image = Image.open(image_path)
    original_image = crop_image_pillow(original_image)

    overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 255))
    overlay_color = (255, 255, 255, 0)

    draw = ImageDraw.Draw(overlay_image)
    for segmentation_mask_image in segmentation_masks:
        draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

    mask_image = overlay_image.convert("RGB") 
    
    gen_image = sd_pipe(prompt=text_prompt,
                        image=original_image,
                        mask_image=mask_image,
                        width=original_image.size[0],
                        height=original_image.size[1]).images[0] 

    return mask_image, gen_image 


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## Inpainting with Stable Diffusin and SAM')
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", tool='sketch')
            prompt = gr.Textbox(label='Prompt')
            run_button = gr.Button(label='Run')
            # with gr.Accordion('Advanced options', open=False):
            #     a_prompt = gr.Textbox(
            #         label='Added Prompt',
            #         value='best quality, extremely detailed')
            #     n_prompt = gr.Textbox(
            #         label='Negative Prompt',
            #         value='longbody, lowres, bad anatomy, bad hands, ' +
            #         'missing fingers, extra digit, fewer digits, '
            #         'cropped, worst quality, low quality')
            #     controlnet_conditioning_scale = gr.Slider(
            #         label='Control Weight',
            #         minimum=0.0,
            #         maximum=2.0,
            #         value=0.7,
            #         step=0.01)
            #     width = gr.Slider(
            #         label='Image Width',
            #         minimum=256,
            #         maximum=768,
            #         value=512,
            #         step=64)
            #     height = gr.Slider(
            #         label='Image Width',
            #         minimum=256,
            #         maximum=768,
            #         value=512,
            #         step=64)

        with gr.Column():
            mask_out = gr.Image(label='Result', elem_id='mask-output')
            image_out = gr.Image(label='Result', elem_id='image-output')

    ips = [image_input, prompt]
    run_button.click(fn=segment, inputs=ips, outputs=[mask_out, image_out])

block.launch(server_name='0.0.0.0')

