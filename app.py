import gradio as gr
import torch, os
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

PORT = os.getenv("PORT", 7537)

torch.set_float32_matmul_precision(["high", "highest"][0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
birefnet.to(device)

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    processed_image = process(im)
    return (processed_image, origin)

def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def process_file(f):
    name_path = f.rsplit(".", 1)[0] + ".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path


def apply_fade_effect(image_path, blur_strength=100, edge_thickness=20):
    """Applies fade effect and returns the file path of the modified image."""
    image_pil = Image.open(image_path).convert("RGBA")
    image = np.array(image_pil)

    alpha = image[:, :, 3]  # Extract alpha channel

    # Detect edges
    edges = cv2.Canny(alpha, 100, 200)

    # Thicken edges
    kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Apply Gaussian blur
    blur_strength = max(1, blur_strength // 2 * 2 + 1)  # Ensure odd value
    blurred_edges = cv2.GaussianBlur(edges_dilated, (blur_strength, blur_strength), 0)

    # Normalize and invert the alpha effect
    blurred_edges = 255 - cv2.normalize(blurred_edges, None, 0, 255, cv2.NORM_MINMAX)

    # Blend with original alpha channel
    new_alpha = np.minimum(alpha, blurred_edges)
    image[:, :, 3] = new_alpha  

    # Convert back to PIL Image
    faded_image = Image.fromarray(image)

    output_path = "faded_image.png"
    faded_image.save(output_path, format="PNG")
    return output_path


with gr.Blocks() as app:
    gr.Markdown("# Image Processor App")
    
    with gr.Tabs():
        with gr.Tab("Remove Background"):
            with gr.Row():
                with gr.Column():
                    input_file = gr.File(label="Upload Image", file_types=["image"])
                with gr.Column():
                    output_image = gr.Image(label="Processed Image (PNG with Transparency)")
            process_button = gr.Button("Remove Background")
            process_button.click(process_file, inputs=input_file, outputs=output_image)

        with gr.Tab("Apply Fade Effect"):
            with gr.Row():
                with gr.Column():
                    processed_image = gr.File(label="Upload Processed PNG")
                with gr.Column():
                    faded_output = gr.File(label="Faded PNG Image")
            with gr.Row():
                with gr.Column():
                    blur_slider = gr.Slider(minimum=10, maximum=1000, step=10, label="Blur Strength", value=100)
                with gr.Column():
                    edge_slider = gr.Slider(minimum=10, maximum=300, step=10, label="Edge Thickness", value=20)
            
            fade_button = gr.Button("Apply Fade")
            fade_button.click(apply_fade_effect, inputs=[processed_image, blur_slider, edge_slider], outputs=faded_output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(PORT), share=True)