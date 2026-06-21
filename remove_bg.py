from rembg import remove
from PIL import Image
import sys

input_path = "/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/static/optimized/imagejoueur.png"
output_path = "/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/static/optimized/imagejoueur_cut.png"

try:
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    output_image.save(output_path)
    print("Background removed successfully!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
