# mcp_utils/file_manager.py
import os
import uuid
from pathlib import Path

class FileManager:
    def __init__(self, base_dir="artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_content(self, filename_hint: str, content: str) -> str:
        # Use uuid to avoid collisions
        ext = ""
        if "." in filename_hint:
            ext = filename_hint.split(".")[-1]
        fname = f"{filename_hint.rstrip('.')}-{uuid.uuid4().hex[:8]}"
        if ext:
            fname = f"{fname}.{ext}"
        fp = self.base_dir / fname
        fp.write_text(content, encoding="utf-8")
        return str(fp.resolve())
