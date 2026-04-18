# AiSearch
AI Search - search images using terms

This is a simple search application written in Python for finding pictures based on what objects they contain. It’s based on OpenCLIP.

## Description ##

An index file is stored directly under the given search root. When images are added it will index those too.

It indexes JPEG, PNG, GIF, BMP, WEBP and TIFF files recursively. It doesn't support any RAW format, nor video.

Indexing will take a long time the first time. Progress is displayed. Only changes will be indexed from then on.

It creates an HTML page that lists all search hits with links to full size pictures. It's stored under the search root.

Each search creates a new web page that includes the search term in the file name for easy reference.

## Variants

* aisearch.py uses a command line interface
* aisearch_gui.py uses a Tkinter GUI interface

## Installation ##

Run "pip install open_clip_torch" to get the image search module.
Run "pip uninstall torch torchvision -y" if you have the CPU version of Pytorch to avoid conflict.
Run "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126" (or later; see https://pytorch.org/) to get GPU acceleration.

## Use ##

* Go to where this document and aisearch.py were stored.
* Run “python aisearch.py”.
* Image path: The path to the root of the picture archive. The created index and generated HTML files are stored here.
* Search terms: A phrase describing the looked-for object. Write "x" for exit.
* Amount (100): How many search hits you want.
* It asks for new search terms until you enter "x".

## Warnings ##

If you see the following it's perfectly OK:

*Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.*

## References ##

* https://github.com/mlfoundations/open_clip
* https://openai.com/index/clip/
* https://pytorch.org/get-started/locally/
