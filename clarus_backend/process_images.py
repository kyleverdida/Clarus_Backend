from PIL import Image, ImageEnhance, ImageFilter

for i in range(1, 5):
    input_path = f"images/fundus{i}.jpg"
    output_path = f"images/fundus{i}_processed.jpg"

    img = Image.open(input_path)
    img = ImageEnhance.Brightness(img).enhance(1.5)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = img.filter(ImageFilter.SHARPEN)
    img.save(output_path)

    print(f"Saved processed image: {output_path}")