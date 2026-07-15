"""Lossless hybrid UTF-8 tokenizer.

IDs 0–255 represent raw bytes and guarantee fallback support for arbitrary
UTF-8 text. IDs 256+ represent frequent non-ASCII characters learned only
from the provided training corpus.
"""

import json
from pathlib import Path


class HybridUTF8Tokenizer:
    def __init__(self, chars):
        if len(chars) != len(set(chars)):
            raise ValueError("Tokenizer vocabulary contains duplicate characters")

        self.id_to_char = list(chars)
        self.char_to_id = {
            char: 256 + index
            for index, char in enumerate(self.id_to_char)
        }
        self.vocab_size = 256 + len(self.id_to_char)

    def encode(self, text):
        ids = []

        for char in text:
            token_id = self.char_to_id.get(char)

            if token_id is not None:
                ids.append(token_id)
            else:
                ids.extend(char.encode("utf-8"))

        return ids

    def decode(self, ids):
        raw = bytearray()

        for token_id in ids:
            token_id = int(token_id)

            if 0 <= token_id < 256:
                raw.append(token_id)
            elif 256 <= token_id < self.vocab_size:
                char = self.id_to_char[token_id - 256]
                raw.extend(char.encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {token_id}")

        return raw.decode("utf-8")

    def save(self, path):
        payload = {
            "type": "hybrid_utf8_char",
            "chars": self.id_to_char,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load(path=None):
    vocab_path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("tokenizer_vocab.json")
    )

    payload = json.loads(vocab_path.read_text(encoding="utf-8"))

    if payload.get("type") != "hybrid_utf8_char":
        raise ValueError("Unsupported tokenizer vocabulary format")

    return HybridUTF8Tokenizer(payload["chars"])