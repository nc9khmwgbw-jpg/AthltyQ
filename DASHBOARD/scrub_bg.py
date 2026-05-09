import os
from PIL import Image

def gentle_transparency(input_filename, threshold=20):
    static_path = "static/"
    filepath = os.path.join(static_path, input_filename)
    
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    img = Image.open(filepath).convert("RGBA")
    
    datas = img.getdata()
    newData = []
    for item in datas:
        # Seuil très bas : on ne vire que ce qui est VRAIMENT noir pur
        if item[0] < threshold and item[1] < threshold and item[2] < threshold:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(filepath, "PNG")
    print(f"✅ Gentle cleaning done for {input_filename}")

# On nettoie les nouveaux v2
targets = ["player-action.png", "player-optimal.png", "player-injured.png"]
for t in targets:
    gentle_transparency(t)