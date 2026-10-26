from PIL import Image, ImageChops
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = "."
OUTPUT_FOLDER = "resized"

# All images will have EXACTLY the same size
TARGET_WIDTH = 1600
TARGET_HEIGHT = 900

# White background
BACKGROUND = (255, 255, 255)

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# PROCESS ALL PNG/JPG/JPEG IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"Found {len(image_files)} images.")

for filename in image_files:

    input_path = os.path.join(INPUT_FOLDER, filename)

    try:
        img = Image.open(input_path).convert("RGB")

        # ----------------------------------------------------
        # STEP 1: REMOVE EXCESS WHITE BORDER
        # ----------------------------------------------------

        bg = Image.new("RGB", img.size, BACKGROUND)

        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()

        if bbox:
            img = img.crop(bbox)

        # ----------------------------------------------------
        # STEP 2: RESIZE WHILE KEEPING ORIGINAL PROPORTION
        # ----------------------------------------------------

        img.thumbnail(
            (TARGET_WIDTH, TARGET_HEIGHT),
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # STEP 3: CREATE SAME SIZE CANVAS
        # ----------------------------------------------------

        canvas = Image.new(
            "RGB",
            (TARGET_WIDTH, TARGET_HEIGHT),
            BACKGROUND
        )

        # ----------------------------------------------------
        # STEP 4: CENTER IMAGE
        # ----------------------------------------------------

        x = (TARGET_WIDTH - img.width) // 2
        y = (TARGET_HEIGHT - img.height) // 2

        canvas.paste(img, (x, y))

        # ----------------------------------------------------
        # STEP 5: SAVE
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename.rsplit(".", 1)[0] + ".png"
        )

        canvas.save(
            output_path,
            "PNG",
            dpi=(300, 300)
        )

        print(f"Done: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\n====================================")
print("ALL IMAGES HAVE BEEN RESIZED")
print("====================================")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Size: {TARGET_WIDTH} x {TARGET_HEIGHT}")