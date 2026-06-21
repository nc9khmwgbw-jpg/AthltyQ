from PIL import Image
import numpy as np

img = Image.open("/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/static/optimized/ll.png").convert("RGBA")
data = np.array(img)

# The background color is approximately #69879A = RGB(105, 135, 154)
# We'll remove pixels close to this color and make them transparent
# But KEEP white text and orange text

bg_r, bg_g, bg_b = 105, 135, 154
tolerance = 60  # How close a pixel must be to the bg color to be removed

# Calculate distance from background color for each pixel
r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
dist = np.sqrt((r.astype(float) - bg_r)**2 + (g.astype(float) - bg_g)**2 + (b.astype(float) - bg_b)**2)

# Make background pixels transparent
# Pixels close to bg color -> transparent
# Pixels far from bg color (white text, orange text) -> fully opaque
is_bg = dist < tolerance
data[is_bg, 3] = 0  # Make background transparent

# Make non-background text pixels fully opaque
data[~is_bg, 3] = 255

result = Image.fromarray(data)
result.save("/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/static/optimized/logo_athlytiq.png")
print("Logo fixed! Background removed, text kept opaque.")

# Verify
verify = np.array(result)
opaque_count = np.sum(verify[:,:,3] == 255)
total = verify.shape[0] * verify.shape[1]
print(f"Opaque pixels now: {opaque_count} ({opaque_count/total*100:.1f}%)")
