import os
from rembg import remove, new_session

# Assure-toi de lancer le script depuis la racine de ton projet
STATIC_PATH = "static/"

def clean_player_image(filename):
    input_path = os.path.join(STATIC_PATH, filename)
    output_path = os.path.join(STATIC_PATH, filename.replace(".png", "-clean.png"))
    
    if not os.path.exists(input_path):
        print(f"❌ Erreur : {input_path} introuvable.")
        return

    print(f"🔄 Traitement IA (Mode Humain) de {filename}...")
    with open(input_path, "rb") as f:
        input_data = f.read()
    
    # On charge le modèle spécialisé pour les humains
    session = new_session("u2net_human_seg")
    
    # On applique les paramètres pour adoucir les bords et garder les vêtements
    output_data = remove(
        input_data, 
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=20,
        post_process_mask=True
    )
    
    with open(output_path, "wb") as f:
        f.write(output_data)
    print(f"✅ Terminé ! Sauvegardé sous : {output_path}")

players_to_clean = ["player-action.png", "player-optimal.png", "player-injured.png"]

for player in players_to_clean:
    clean_player_image(player)