# AI Painter

An algorithm that reconstructs one image *(the shape image)* using patches from another image *(the appearance image)*. It produces images with an unusual, mosaic-like texture, very different from the digital appearance of traditional neural style transfer.

This project is the basis of an [art piece](https://www.behance.net/gallery/127536973/Seasons) that won the St Edmund's College, Cambridge Art Competition (Christmas Card Category).

Shape Image | Appearance Image | Result Image
:-:|:-:|:-:
![vermeer.png](https://github.com/Dan-Andrei-Iliescu/ai-painting/blob/master/data/vermeer.png?raw=true)  |  ![gauguin.png](https://github.com/Dan-Andrei-Iliescu/ai-painting/blob/master/data/gauguin.png?raw=true) | ![portrait.png](https://github.com/Dan-Andrei-Iliescu/ai-painting/blob/master/results/portrait.png?raw=true)

## Instructions

1. Clone the repo.
2. Install [Poetry](https://python-poetry.org), a Python virtualenv/dependency management system). Then install the dependencies using the command `$ poetry install`.
3. Run the following command:
```
$ poetry run python painter.py \
    --shape_img vermeer.png \
    --app_img gauguin.png \
    --res_img portrait.png \
    --num_patches 1024 \
    --num_trials 8
```

- The argument `--num_patches` controls how many patches from the *appearance image* are used to reconstruct the *shape img*. A higher number produces a more detailed reconstruction, while a lower number renders larger patches.
- The argument `--num_trials` controls the maximum number of trials spent trying to find a matching patch. A higher number produces more accurate patch matching, while a lower number increases the variability in the patches.