import os, zipfile

src = "./opt"

for name in os.listdir(src):
    full = os.path.join(src, name)
    if not os.path.isfile(full):
        continue
    if not name.endswith(".json"):
        continue

    zip_path = full + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(full, arcname=name)