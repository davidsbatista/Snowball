import re
from typing import Generator, TextIO

tags_regex = re.compile("<[A-Z]+>[^<]+</[A-Z]+>", re.U)


def blocks(files: TextIO, size: int = 65536) -> Generator[str, None, None]:
    """Read the file block-wise."""
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer


def clean_tags(sentence: str) -> str:
    """Remove tags from a sentence."""
    return re.sub(tags_regex, "", sentence)
