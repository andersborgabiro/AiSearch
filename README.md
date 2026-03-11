# AiSearch
AI Search - search images using descriptions

This is a simple search application written in Python for finding pictures based on what objects they contain. It’s based on Manuel Fay’s ImageSearcher library. The library is rather old, so it might not be optimal for very new types of objects.

## Description ##

It creates a local web page that lists all search hits with links to full size pictures.

Each search creates a new web page that includes the search term in the file name for easy reference.

It searches only JPEG files.

Indexing will take a long time the first time. Only changes will be indexed from then on.

The index is stored in the root of the given start folder. It will index all image files under that folder recursively. When images are added it will index those too.

## Installation ##

* Run pip install image-searcher

## Use ##

* Go to where this document and aisearch.py was stored.
* Run “python aisearch.py”.
* Image path: The path to the root of the picture archive. The created index is stored here.
* Search terms: A phrase describing the looked-for object. Write "x" for exit.
* Amount (100): How many search hits you want.
* It loops for more searches.

## References ##

* https://github.com/ManuelFay/ImageSearcher
* https://openai.com/index/clip/
