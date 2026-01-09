import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-large-handwritten",
    image_processor_kwargs={"do_rescale": False, "size": {"height": 384, "width": 384}}
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten"
).to(device)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

for p in model.parameters():
    p.requires_grad = False

for p in model.decoder.output_projection.parameters():
    p.requires_grad = True

img_tfm = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

class OCRDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        img = img_tfm(img)
        px = processor(images=img, return_tensors="pt", do_rescale=False).pixel_values.squeeze(0)
        lbl = processor.tokenizer(
            item["label"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        lbl[lbl == processor.tokenizer.pad_token_id] = -100
        return px, lbl

ds = load_dataset("vklinhhh/imgur5k_words", split="train")
train_ds = OCRDataset(ds)

loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)

model.train()
for ep in range(5):
    for px, lb in loader:
        px, lb = px.to(device), lb.to(device)
        loss = model(pixel_values=px, labels=lb).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {ep+1} loss {loss.item():.4f}")

model.save_pretrained("trocr-finetuned")
processor.save_pretrained("trocr-finetuned")
