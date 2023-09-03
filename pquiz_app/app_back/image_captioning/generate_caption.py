import torch
from PIL import Image
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, AutoTokenizer

def generate_caption(path):
    # Load the saved checkpoint and model
    model_path = "/Users/hyundolee/campus_project/pquiz_app/app_back/image_captioning/hugging_ckpt"
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the input image
    image_path = path
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = image_transforms(image).unsqueeze(0).to(device)

    # Generate caption
    caption = model.generate(
        pixel_values=image,
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.sep_token_id,
        max_length=256,
        num_beams=1,
        no_repeat_ngram_size=3,
        length_penalty=2.0
    )

    # Decode the generated caption
    generated_caption = tokenizer.decode(caption[0], skip_special_tokens=True)

    print("Generated Caption:", generated_caption)

    return generated_caption
