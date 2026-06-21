from rembg import remove
from PIL import Image

img = Image.open("/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/static/optimized/ll.png")
out = remove(img)
out.save("/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/static/optimized/ll.png")
print("Logo background removed!")
