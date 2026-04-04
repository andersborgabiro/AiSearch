# https://github.com/mlfoundations/open_clip
# https://openai.com/index/clip/
# https://pytorch.org/get-started/locally/

import html
import os
import pathlib
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

import open_clip
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
BATCH_SIZE = 64


def build_markup(path):
    path_url = pathlib.Path(path).as_uri()
    markup = "<div style='text-align: center;'>"
    markup += "<a href='" + html.escape(path_url) + "' target='_blank'>"
    markup += (
        "<img src='"
        + html.escape(path_url)
        + "' style='max-width: 320px; max-height: 320px; display: block; margin: 0 auto;'/>"
    )
    markup += "<br/>"
    markup += html.escape(path)
    markup += "</a>"
    markup += "</div>\n"
    return markup


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image Search")
        self.resizable(True, True)
        self.minsize(640, 480)

        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.device = None
        self.image_paths = []
        self.image_features = None
        self.index_file = None

        self._q = queue.Queue()
        self._build_ui()
        self._load_model_async()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- top frame: path selection + load button ---
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Image path:").pack(side="left")
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var, width=55).pack(side="left", padx=(4, 2))
        ttk.Button(top, text="Browse…", command=self._browse).pack(side="left", padx=(0, 4))
        self.load_btn = ttk.Button(top, text="Load / Index", command=self._load_index_async, state="disabled")
        self.load_btn.pack(side="left")

        # --- middle frame: search controls ---
        mid = ttk.Frame(self)
        mid.pack(fill="x", **pad)

        ttk.Label(mid, text="Query:").pack(side="left")
        self.query_var = tk.StringVar()
        query_entry = ttk.Entry(mid, textvariable=self.query_var, width=45)
        query_entry.pack(side="left", padx=(4, 2))
        query_entry.bind("<Return>", lambda _: self._search())

        ttk.Label(mid, text="Max results:").pack(side="left", padx=(8, 2))
        self.amount_var = tk.StringVar(value="100")
        ttk.Spinbox(mid, from_=1, to=10000, textvariable=self.amount_var, width=6).pack(side="left", padx=(0, 4))

        self.search_btn = ttk.Button(mid, text="Search", command=self._search, state="disabled")
        self.search_btn.pack(side="left")

        # --- status bar ---
        self.status_var = tk.StringVar(value="Loading CLIP model…")
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken").pack(
            fill="x", side="bottom", ipady=2, padx=2, pady=(0, 2)
        )

        # --- results area ---
        results_frame = ttk.Frame(self)
        results_frame.pack(fill="both", expand=True, **pad)

        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side="right", fill="y")

        self.result_list = tk.Listbox(
            results_frame,
            yscrollcommand=scrollbar.set,
            selectmode="browse",
            activestyle="none",
            font=("Courier New", 9),
        )
        self.result_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.result_list.yview)
        self.result_list.bind("<Double-Button-1>", self._open_selected)

        ttk.Label(self, text="Double-click a result to open it in the default viewer.", anchor="w").pack(
            fill="x", padx=8, pady=(0, 2)
        )

    # --------------------------------------------------------- model loading --

    def _load_model_async(self):
        threading.Thread(target=self._load_model_worker, daemon=True).start()
        self.after(100, self._poll_queue)

    def _load_model_worker(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "ViT-L-14"
        self._q.put(("status", f"Loading CLIP model (ViT-L-14) on {device}…"))
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", force_quick_gelu=True
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        self._q.put(("model_ready", (model, preprocess, tokenizer, device)))

    # -------------------------------------------------------- index loading ---

    def _browse(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self.path_var.set(folder)

    def _load_index_async(self):
        path = self.path_var.get().strip()
        if not path:
            self.status_var.set("Please enter or browse to an image folder.")
            return
        self.load_btn.config(state="disabled")
        self.search_btn.config(state="disabled")
        self.result_list.delete(0, "end")
        threading.Thread(target=self._load_index_worker, args=(path,), daemon=True).start()

    def _load_index_worker(self, path):
        index_file = Path(path) / "aisearch_index.pt"

        image_paths = []
        image_features_list = []

        if index_file.exists():
            self._q.put(("status", f"Loading existing index…"))
            saved = torch.load(index_file, weights_only=False)
            image_paths = saved["image_paths"]
            image_features_list = [saved["image_features"]]
            self._q.put(("status", f"Loaded {len(image_paths)} indexed images. Scanning for new files…"))

        indexed_set = set(image_paths)
        all_candidates = [
            str(p) for p in Path(path).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        new_candidates = [p for p in all_candidates if p not in indexed_set]

        if new_candidates:
            self._q.put(("status", f"Encoding {len(new_candidates)} new images…"))
            new_image_paths = []
            with torch.no_grad():
                for i in range(0, len(new_candidates), BATCH_SIZE):
                    batch_paths = new_candidates[i : i + BATCH_SIZE]
                    images, valid = [], []
                    for p in batch_paths:
                        try:
                            img = self.preprocess(Image.open(p).convert("RGB"))
                            images.append(img)
                            valid.append(p)
                        except Exception:
                            pass
                    if not images:
                        continue
                    batch = torch.stack(images).to(self.device)
                    feats = self.model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    image_features_list.append(feats.cpu())
                    new_image_paths.extend(valid)
                    self._q.put(("status", f"Encoded {len(new_image_paths)}/{len(new_candidates)}…"))
            image_paths.extend(new_image_paths)

        if not image_paths:
            self._q.put(("index_error", "No images could be indexed."))
            return

        image_features = torch.cat(image_features_list, dim=0)

        if new_candidates:
            torch.save({"image_paths": image_paths, "image_features": image_features}, index_file)

        self._q.put(("index_ready", (image_paths, image_features, index_file)))

    # -------------------------------------------------------------- search ---

    def _search(self):
        if self.image_features is None:
            return
        query = self.query_var.get().strip()
        if not query:
            return

        try:
            amount = int(self.amount_var.get())
        except ValueError:
            amount = 100

        self.search_btn.config(state="disabled")
        threading.Thread(target=self._search_worker, args=(query, amount), daemon=True).start()

    def _search_worker(self, query, amount):
        with torch.no_grad():
            tokens = self.tokenizer([query]).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (self.image_features @ text_features.cpu().T).squeeze(1)
        top_k = min(amount, len(self.image_paths))
        top_indices = similarities.topk(top_k).indices.tolist()
        result = [self.image_paths[i] for i in top_indices]

        title = (
            f"Searching for '{query}' resulting in {len(result)} pictures of max {amount}"
        )
        file_name = str(self.index_file.parent / (title + ".html"))
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"<h2>{html.escape(title)}</h2>")
            for item in result:
                f.write(build_markup(item))

        self._q.put(("search_done", (result, file_name, title)))

    # ----------------------------------------------- open file on dbl-click --

    def _open_selected(self, _event):
        sel = self.result_list.curselection()
        if not sel:
            return
        # strip the leading rank prefix "  1. "
        line = self.result_list.get(sel[0])
        path = line.split(". ", 1)[-1].strip()
        if Path(path).exists():
            os.startfile(path)

    # ---------------------------------------------------- queue poll loop ---

    def _poll_queue(self):
        try:
            while True:
                msg, data = self._q.get_nowait()

                if msg == "status":
                    self.status_var.set(data)

                elif msg == "model_ready":
                    self.model, self.preprocess, self.tokenizer, self.device = data
                    self.status_var.set("Model ready. Select a folder and click Load / Index.")
                    self.load_btn.config(state="normal")

                elif msg == "index_ready":
                    self.image_paths, self.image_features, self.index_file = data
                    self.status_var.set(f"Ready — {len(self.image_paths)} images indexed.")
                    self.load_btn.config(state="normal")
                    self.search_btn.config(state="normal")

                elif msg == "index_error":
                    self.status_var.set(data)
                    self.load_btn.config(state="normal")

                elif msg == "search_done":
                    result, file_name, title = data
                    self.result_list.delete(0, "end")
                    for i, p in enumerate(result, 1):
                        self.result_list.insert("end", f"{i:4}. {p}")
                    self.status_var.set(f"{len(result)} results — HTML saved to {file_name}")
                    self.search_btn.config(state="normal")

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)


if __name__ == "__main__":
    app = App()
    app.mainloop()
