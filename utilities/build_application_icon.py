"""Build a multi-resolution Windows ICO from the existing, unchanged logo."""
from pathlib import Path
import struct

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage, QPainter


def build_icon(source, destination):
    image = QImage(str(source))
    if image.isNull():
        raise ValueError(f"Cannot read logo: {source}")
    entries = []
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        scaled = image.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        square = QImage(size, size, QImage.Format.Format_ARGB32)
        square.fill(Qt.GlobalColor.transparent)
        painter = QPainter(square)
        painter.drawImage((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
        painter.end()
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not square.save(buffer, "PNG"):
            raise RuntimeError("Could not encode icon image")
        entries.append((size, bytes(data)))
    offset = 6 + 16 * len(entries)
    directory = []
    for size, data in entries:
        directory.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    Path(destination).write_bytes(struct.pack("<HHH", 0, 1, len(entries)) + b"".join(directory)
                                 + b"".join(data for _, data in entries))


if __name__ == "__main__":
    from app.paths import LOGO_PATH, WINDOWS_ICON_PATH
    build_icon(LOGO_PATH, WINDOWS_ICON_PATH)
