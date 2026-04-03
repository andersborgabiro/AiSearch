# https://github.com/mlfoundations/open_clip
# https://openai.com/index/clip/
# https://pytorch.org/get-started/locally/

import pathlib
import html
import torch
import open_clip
from PIL import Image
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
BATCH_SIZE = 64


# Generate HTML markup with scaled images and links to full size images
def build_markup(path):
    path_url = pathlib.Path(path).as_uri()

    markup = ""
    markup += "<p style='text-align: center;'>"
    markup += "<a href='" + html.escape(path_url) + "' target='_blank'>"
    markup += (
        "<img src='"
        + html.escape(path_url)
        + "' style='max-width: 320px; max-height: 320px; display: block; margin: 0 auto;'/>"
    )
    markup += "<br/>"
    markup += html.escape(path)
    markup += "</a>"
    markup += "</p>"
    markup += "\n"

    return markup


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-L-14"
pretrained = "openai"

print(f"Loading CLIP model ({model_name}) on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained, force_quick_gelu=True
)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

path = input("Image path: ")

index_file = Path(path) / "aisearch_index.pt"

image_paths = []
image_features_list = []

if index_file.exists():
    print(f"Loading existing index from '{index_file}'...")
    saved = torch.load(index_file, weights_only=False)
    image_paths = saved["image_paths"]
    image_features_list = [saved["image_features"]]
    print(f"Loaded {len(image_paths)} indexed images.")

indexed_set = set(image_paths)
all_candidate_paths = [
    str(p) for p in Path(path).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
]
candidate_paths = [p for p in all_candidate_paths if p not in indexed_set]

if candidate_paths:
    print(f"Found {len(candidate_paths)} new images. Encoding...")
    new_image_paths = []
    with torch.no_grad():
        for i in range(0, len(candidate_paths), BATCH_SIZE):
            batch_paths = candidate_paths[i : i + BATCH_SIZE]
            images = []
            valid = []
            for p in batch_paths:
                try:
                    img = preprocess(Image.open(p).convert("RGB"))
                    images.append(img)
                    valid.append(p)
                except Exception:
                    pass
            if not images:
                continue
            batch = torch.stack(images).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features_list.append(feats.cpu())
            new_image_paths.extend(valid)
            print(f"  {len(new_image_paths)}/{len(candidate_paths)}", end="\r")
    print()
    image_paths.extend(new_image_paths)

else:
    print("No new images to index.")

if not image_paths:
    print("No images could be indexed.")
    exit(1)

image_features = torch.cat(image_features_list, dim=0)

if candidate_paths:
    torch.save(
        {"image_paths": image_paths, "image_features": image_features}, index_file
    )
    print(f"Index saved to '{index_file}'.")

print(f"Total indexed images: {len(image_paths)}.")

while True:
    search_criteria = input("Search terms or 'x': ")
    if search_criteria == "x":
        break

    if search_criteria == "":
        continue

    search_amount = 100
    search_amount_text = input("Amount (100): ")
    if search_amount_text != "":
        try:
            search_amount = int(search_amount_text)
        except Exception:
            pass

    with torch.no_grad():
        tokens = tokenizer([search_criteria]).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarities = (image_features @ text_features.cpu().T).squeeze(1)
    top_k = min(search_amount, len(image_paths))
    top_indices = similarities.topk(top_k).indices.tolist()
    result = [image_paths[i] for i in top_indices]

    print(str(len(result)) + " ranked images")

    title = (
        "Searching for '"
        + search_criteria
        + "' resulting in "
        + str(len(result))
        + " pictures of max "
        + str(search_amount)
    )
    file_name = str(index_file.parent / (title + ".html"))
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("<h2>" + title + "</h2>")
        for item in result:
            f.write(build_markup(item))

    print("Created " + file_name)

print("Thanks for using image search. We hope to see you again soon.")
