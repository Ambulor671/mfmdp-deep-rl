# -*- coding: utf-8 -*-
"""
Minimal I/O utilities and CSV logging.
"""

import csv
import os
from typing import Any, Dict, Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_eval_row(csv_path: str, row: Dict[str, Any]) -> None:
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


class CSVLogger:
    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(fieldnames))
        self._writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
