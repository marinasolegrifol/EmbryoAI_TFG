import os
from PIL import Image

ROOT = r"C:\Users\oriol\Desktop\excels"

bad = 0
good = 0

valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

for root, dirs, files in os.walk(ROOT):

    for f in files:

        if not f.lower().endswith(valid_ext):
            continue

        path = os.path.join(root, f)

        try:
            with Image.open(path) as img:
                img.load()            # <- CLAVE
                img.convert("RGB")    # <- CLAVE

            good += 1

        except Exception as e:
            print(f"CORRUPTA: {path}")
            bad += 1

            try:
                os.remove(path)
            except:
                pass

print("\n...NETEJA COMPLETADA...")
print("Bones : ", good)
print("Corruptes eliminades : ", bad)