import os
from rembg import remove

STATIC_PATH = "DASHBOARD/static/"

def clean_image(input_name, output_name):
    input_path = os.path.join(STATIC_PATH, input_name)
    output_path = os.path.join(STATIC_PATH, output_name)
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    print(f"🔄 AI Processing (rembg) for {input_name}...")
    with open(input_path, "rb") as f:
        input_data = f.read()
    
    # Apply background removal
    output_data = remove(input_data)
    
    with open(output_path, "wb") as f:
        f.write(output_data)
    print(f"✅ Success! Saved to: {output_path}")

if __name__ == "__main__":
    clean_image("tactical-core.png", "tactical-core-clean.png")
    clean_image("performance-core.png", "performance-core-clean.png")
