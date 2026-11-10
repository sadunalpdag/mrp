"""Bounded standalone ZIP exports for the standard Telegram Bot API.

The Bot API accepts at most 50 MB per uploaded document. We stop at 49
decimal MB (one decimal MB margin), including ZIP headers and indexes.
Every part is a valid ZIP; very large individual members are split into
numbered byte-perfect pieces with a manifest to explain reconstruction.
"""
import json
import os
import shutil
import zipfile
import zlib

TELEGRAM_DOCUMENT_LIMIT_BYTES = 50_000_000
REPORT_PART_LIMIT_BYTES = TELEGRAM_DOCUMENT_LIMIT_BYTES - 1_000_000
LARGE_MEMBER_CHUNK_BYTES = 8_000_000


def report_zip_parts(path, limit=REPORT_PART_LIMIT_BYTES,
                     chunk_bytes=LARGE_MEMBER_CHUNK_BYTES):
    """Return ordered paths; never create a file over *limit* bytes.

    Small existing ZIPs are returned untouched. Large files are re-packaged
    from original ZIP members, preserving each unsplit member byte-for-byte.
    Members too large to fit a part are split into *.partNNNN byte chunks.
    Append these chunks in numeric order to restore the exact original file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) <= limit:
        return [path]
    if limit < 100_000 or chunk_bytes <= 0:
        raise ValueError("INVALID_REPORT_PART_SIZE")

    stem, _ = os.path.splitext(path)
    reserve = min(256_000, max(16_000, limit // 12))
    paths = []
    written = []
    mapping = []
    current = None
    current_count = 0

    def begin():
        nonlocal current, current_count
        filename = f"{stem}.part{len(paths)+1:03d}.zip"
        tmp = filename + ".tmp"
        written.append(tmp)
        paths.append(filename)
        current = zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                                  compresslevel=6, allowZip64=True)
        current_count = 0

    def push(name, size_hint, source, data=None):
        nonlocal current, current_count
        # Reserve the ZIP footer, central-directory entries and manifest.
        budget = limit - reserve
        if size_hint + 1024 + 2*len(name.encode("utf-8")) >= budget:
            raise ValueError("REPORT_MEMBER_TOO_LARGE")
        if current is None:
            begin()
        estimated = (current.fp.tell() + size_hint + 2048 +
                     150*(current_count+1) + 2*len(name.encode("utf-8")))
        if current_count and estimated > budget:
            current.close()
            begin()
        if data is None:
            with current.open(name, "w", force_zip64=True) as output:
                shutil.copyfileobj(source, output, 1024*1024)
        else:
            current.writestr(name, data)
        current_count += 1
        # Guaranteed by the reserved margin unless compression estimate of
        # a source entry was drastically incorrect, which is checked below.
        if current.fp.tell() + 150*current_count >= budget:
            current.close()
            current = None

    try:
        with zipfile.ZipFile(path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                if (member.compress_size + 2048 + 2*len(name.encode("utf-8"))
                        < limit - reserve - 1024):
                    with archive.open(member) as source:
                        push(name, member.compress_size, source)
                    mapping.append({"name": name, "files": [name]})
                else:
                    pieces = []
                    with archive.open(member) as source:
                        number = 0
                        while True:
                            data = source.read(min(chunk_bytes, limit//4))
                            if not data:
                                break
                            number += 1
                            partname = name + f".part{number:04d}"
                            # zlib's 6-byte wrapper is absent in raw ZIP deflate.
                            compressed_hint = len(zlib.compress(data, level=6))
                            push(partname, compressed_hint, None, data)
                            pieces.append(partname)
                    mapping.append({"name": name, "files": pieces})
            if current is not None:
                current.close()
                current = None

        if not paths:
            raise ValueError("EMPTY_REPORT_ARCHIVE")
        instruction = (
            "Her dosya bagimsiz bir ZIP arsividir; sirayla indirin.\n"
            "Normal dosyalar acildiginda dogrudan kullanilabilir.\n"
            "Parcalanmis buyuk bir kaynak icin ayni adi tasiyan .part0001, "
            ".part0002 ... dosyalarinin BAYTLARINI sirayla birlestirin.\n"
            "Orijinal adlar ve parca listesi asagidaki JSON'dadir.\n\n"
            + json.dumps({"version": 1, "source": os.path.basename(path),
                          "zip_parts": [os.path.basename(p) for p in paths],
                          "original_files": mapping}, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(paths[0]+".tmp", "a",
                             compression=zipfile.ZIP_DEFLATED) as first:
            first.writestr("MULTIPART_README.txt", instruction)
        for tmp, dest in zip(written, paths):
            if os.path.getsize(tmp) > limit:
                raise ValueError("REPORT_PART_LIMIT_EXCEEDED")
            # Readable ZIP before making it visible to the sender.
            with zipfile.ZipFile(tmp) as check:
                if check.testzip() is not None:
                    raise ValueError("CORRUPT_REPORT_PART")
        for tmp, dest in zip(written, paths):
            os.replace(tmp, dest)
        return paths
    finally:
        if current is not None:
            current.close()
        for tmp in written:
            if os.path.isfile(tmp):
                os.unlink(tmp)
