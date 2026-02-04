import urllib.request
import ssl
import certifi
import os

# Run this python script in order to create and add the book files that will be used to train and test our model.

os.makedirs("data", exist_ok=True)

ssl_context = ssl.create_default_context(cafile=certifi.where())

urls = {
    "the_sun_also_rises.txt": "https://www.gutenberg.org/cache/epub/67138/pg67138.txt",
    "ulysses.txt": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    "pride_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
}

for name, url in urls.items():
    with urllib.request.urlopen(url, context=ssl_context) as r:
        with open(f"data/{name}", "wb") as f:
            f.write(r.read())
