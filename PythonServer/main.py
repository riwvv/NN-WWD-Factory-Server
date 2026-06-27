from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title="NN-WWD-Factory Audio Generator",
    version="1.0.0"
)
app.include_router(router)

# --- ЭТОТ БЛОК НУЖНО ДОБАВИТЬ ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)