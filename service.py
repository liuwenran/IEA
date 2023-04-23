import numpy as np
import cv2
from PIL import Image
import torch
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

# mask_generator = None
# sd_pipe = None

global origin_image_path
origin_image_path = None
global incremental_mask
incremental_mask = None


def crop_image_pillow(img, divide=8):
    width, height = img.size
    if width > 960:
        img = img.resize((960, int(960 / width * height)))
        width, height = img.size
    if height > 960:
        img = img.resize((int(960 / height * width), 960))
        width, height = img.size

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
        height, width, c = img.shape
    if height > 960:
        img = cv2.resize(img, (int(960 / height * width), 960))
        height, width, c = img.shape

    top = height - height // divide * divide
    left = width - width // divide * divide
    img = img[top:, left:, :]
    return img


def segment(
    text_prompt: str,
):
    global origin_image_path
    original_image = Image.open(origin_image_path)
    original_image = crop_image_pillow(original_image)

    global incremental_mask
    output_draw_mask = incremental_mask * 255
    output_draw_mask = np.expand_dims(output_draw_mask, axis=2)
    output_draw_mask = np.repeat(output_draw_mask, repeats=3, axis=2)
    output_draw_mask = Image.fromarray(output_draw_mask)

    gen_image = sd_pipe(prompt=text_prompt,
                        image=original_image,
                        mask_image=output_draw_mask,
                        width=original_image.size[0],
                        height=original_image.size[1]).images[0]

    return gen_image


def clear_cache():
    global origin_image_path
    origin_image_path = None
    global incremental_mask
    incremental_mask = None

    return None, None, None, None


def preview(pathes, use_drawed_mask):
    image_path = pathes['image']
    draw_mask_path = pathes['mask']

    global origin_image_path
    if origin_image_path is None:
        origin_image_path = image_path

    draw_mask = cv2.imread(draw_mask_path)
    draw_mask = crop_image(draw_mask)
    gray = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
    draw_mask = gray > 0
    draw_mask = draw_mask.astype('uint8')

    if not use_drawed_mask:
        image = cv2.imread(image_path)
        image = crop_image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)

        indices = []
        for i, mask in enumerate(masks):
            bitwise_res = cv2.bitwise_and(mask['segmentation'].astype('uint8'), draw_mask)
            if np.sum(bitwise_res) > 0:
                indices.append(i)

        for seg_idx in indices:
            draw_mask = cv2.bitwise_or(masks[seg_idx]["segmentation"].astype('uint8'), draw_mask)

    global incremental_mask
    if incremental_mask is None:
        incremental_mask = draw_mask
    else:
        incremental_mask = cv2.bitwise_or(incremental_mask, draw_mask)

    output_draw_mask = incremental_mask * 255
    output_draw_mask = np.expand_dims(output_draw_mask, axis=2)
    output_draw_mask = np.repeat(output_draw_mask, repeats=3, axis=2)
    output_draw_mask = Image.fromarray(output_draw_mask)

    mask_image_binary = 1 - incremental_mask
    nb = np.expand_dims(mask_image_binary, axis=2)
    nm = np.repeat(nb, repeats=3, axis=2)

    image = cv2.imread(image_path)
    image = crop_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image_masked = image * nm
    original_image_masked = Image.fromarray(original_image_masked)

    return original_image_masked, output_draw_mask, original_image_masked


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## Inpainting with Stable Diffusin and SAM')
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", tool='sketch', brush_radius=20)
            use_drawed_mask = gr.Checkbox(label='Use drawed mask.', value=False)
            preview_button = gr.Button(value='Generate Mask', label='Generate Mask')
            prompt = gr.Textbox(label='Prompt')
            run_button = gr.Button(label='Run')
            image_out = gr.Image(label='Inpainting Result', elem_id='inpainting-output').style(height=524)

        with gr.Column():
            preview_image = gr.Image(type="filepath", tool='sketch').style(height=524)
            mask_out = gr.Image(label='Mask Result', elem_id='mask-output').style(height=524)

    with gr.Row():
        clear_btn = gr.Button(value='Clear Cache', label='Clear Cache')

    preview_btn_inpus = [image_input, use_drawed_mask]
    preview_btn_outputs = [preview_image, mask_out, image_input]
    preview_button.click(fn=preview,
                         inputs=preview_btn_inpus,
                         outputs=preview_btn_outputs)

    run_button.click(fn=segment, inputs=[prompt], outputs=[image_out])

    clear_btn.click(fn=clear_cache, outputs=[preview_image, mask_out, image_input, image_out])

block.launch(server_name='0.0.0.0', server_port=8151, share=False)

