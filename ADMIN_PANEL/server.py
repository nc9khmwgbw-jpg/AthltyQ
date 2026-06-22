import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="AthlytIQ Control Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PIPELINE_STEPS = {
    "step_1": {
        "name": "Scraping SofaScore (Auto-détection)",
        "cmd": ".venv/bin/python DATA_PIPELINE/SCRAPPING/main.py --mode auto"
    },
    "step_2": {
        "name": "Scraping Transfermarkt (Blessures)",
        "cmd": ".venv/bin/python DATA_PIPELINE/SCRAPPING/main.py --source 2 --league ALL"
    },
    "step_3": {
        "name": "Nettoyage et Fusion (Master Dataset)",
        "cmd": ".venv/bin/python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py"
    },
    "step_4": {
        "name": "Audit de Santé",
        "cmd": ".venv/bin/python DATA_PIPELINE/MAINTENANCE/reconciler.py"
    },
    "step_5": {
        "name": "Feature Engineering (Fatigue & Risques)",
        "cmd": ".venv/bin/python LM/models/feature_engineering.py"
    },
    "step_6": {
        "name": "Classification 9 Postes",
        "cmd": ".venv/bin/python LM2/build_positions.py"
    },
    "step_7": {
        "name": "Préparation Benchmark Scouting",
        "cmd": ".venv/bin/python LM2/benchmark/setup_scouting_data.py"
    },
    "step_8": {
        "name": "Apprentissage Poids Similarité",
        "cmd": ".venv/bin/python LM2/benchmark/train_weights.py"
    },
    "step_9": {
        "name": "Clustering des Archétypes",
        "cmd": ".venv/bin/python LM2/benchmark/train_clusters.py"
    },
    "step_10": {
        "name": "Préparation Benchmark IA",
        "cmd": ".venv/bin/python LM/models/benchmark/setup_data.py"
    },
    "step_11": {
        "name": "Entraînement Algorithmes (Poly, LGBM, RF)",
        "cmd": ".venv/bin/python LM/models/benchmark/Polynomial_Regression/train_poly.py > reports/training_poly.txt 2>&1 && .venv/bin/python LM/models/benchmark/LightGBM/train_lgbm.py > reports/training_lgbm.txt 2>&1 && .venv/bin/python LM/models/benchmark/Random_Forest/train_rf.py > reports/training_rf.txt 2>&1"
    },
    "step_12": {
        "name": "Évaluation et Visualisation",
        "cmd": ".venv/bin/python LM/models/benchmark/generate_visuals.py && .venv/bin/python LM2/benchmark/generate_scouting_visuals.py"
    }
}

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

async def run_pipeline_steps(websocket: WebSocket, selected_steps: list, context: dict):
    try:
        total_steps = len(selected_steps)
        for i, step_id in enumerate(selected_steps):
            if step_id not in PIPELINE_STEPS:
                continue
                
            step_info = PIPELINE_STEPS[step_id]
            cmd = step_info["cmd"]
            
            await websocket.send_text(f"\n[SYSTEM] === Démarrage de l'étape [{i+1}/{total_steps}] : {step_info['name']} ===")
            
            # Use create_subprocess_shell because cmd might contain '&&' and redirections
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            context["process"] = process

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                await websocket.send_text(text)
                
            await process.wait()
            context["process"] = None
            
            if process.returncode != 0:
                # Si le processus a été interrompu manuellement, returncode est généralement -15 (SIGTERM)
                if process.returncode == -15:
                    await websocket.send_text(f"[ERROR] ❌ Étape interrompue par l'utilisateur.")
                else:
                    await websocket.send_text(f"[ERROR] ❌ L'étape a échoué avec le code {process.returncode}.")
                break
                
        else:
            await websocket.send_text("\n[SYSTEM] ✅ Pipeline terminé avec succès !")

    except asyncio.CancelledError:
        # Tâche annulée via websocket (Stop)
        if context["process"] and context["process"].returncode is None:
            context["process"].terminate()
            await websocket.send_text("\n[SYSTEM] 🛑 SIGNAL D'ARRÊT ENVOYÉ. Fin forcée du processus.")

@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    await websocket.accept()
    context = {"process": None}
    run_task = None
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "start":
                if run_task and not run_task.done():
                    await websocket.send_text("[ERROR] ⚠️ Un pipeline est déjà en cours d'exécution.")
                    continue
                
                selected_steps = data.get("steps", [])
                if not selected_steps:
                    await websocket.send_text("[ERROR] ⚠️ Aucune étape sélectionnée.")
                    continue
                    
                # Lancement en arrière-plan pour garder le websocket réactif (permet le "stop")
                run_task = asyncio.create_task(run_pipeline_steps(websocket, selected_steps, context))
                
            elif action == "stop":
                if context["process"] and context["process"].returncode is None:
                    context["process"].terminate()
                    await websocket.send_text("[SYSTEM] 🛑 Arrêt d'urgence déclenché par l'utilisateur.")
                if run_task and not run_task.done():
                    run_task.cancel()
                    
    except WebSocketDisconnect:
        print("Client déconnecté.")
        if context["process"] and context["process"].returncode is None:
            context["process"].terminate()
        if run_task and not run_task.done():
            run_task.cancel()
    except Exception as e:
        print(f"Erreur WebSocket: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8082, reload=True)
