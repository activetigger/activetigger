"""
Shared image-processing helpers used by image-project tasks.

Centralises corrupt-image pre-flight and thumbnail generation so the
project-creation and add-evalset paths apply the same cleaning rules.
"""

from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

# Match the global cap set in tasks/train_image.py so verify() rejects bombs
# the same way regardless of which entry point sees the image first.
Image.MAX_IMAGE_PIXELS = 64_000_000

MAX_ZIP_BYTES = 8 * 1024 * 1024 * 1024
MAX_IMAGE_BYTES = 100 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}


def is_readable_image(path: str | Path) -> bool:
    """
    True iff PIL can open the file and verify() succeeds.

    Catches the same exception set as the original inline check in
    TrainImage.__check_data so behaviour is unchanged for callers.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (
        FileNotFoundError,
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        return False


def filter_readable_images(
    paths: Iterable[str | Path],
    stop_check: Callable[[], None] | None = None,
) -> list[bool]:
    """
    Return a list of readability flags aligned with `paths`.

    `stop_check` is called every 100 items so the caller can raise on a
    user-requested cancellation without checking inside is_readable_image.
    """
    flags: list[bool] = []
    for i, p in enumerate(paths):
        if stop_check is not None and i % 100 == 0:
            stop_check()
        flags.append(is_readable_image(p))
    return flags


def generate_thumbnail(src: Path, dest: Path, max_size: int = 256) -> bool:
    """
    Write a `max_size`-px JPEG thumbnail of `src` to `dest`.

    Returns True on success, False if Pillow could not process the file.
    EXIF rotation is honored and non-RGB/L modes are converted to RGB to
    avoid JPEG encode errors on palette / RGBA inputs.
    """
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            im.save(dest, "JPEG", quality=80, optimize=True)
        return True
    except Exception as ex:
        print(f"thumbnail generation failed for {src}: {ex}")
        return False
