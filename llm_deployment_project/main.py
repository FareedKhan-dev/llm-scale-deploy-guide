import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from optimized_llm_server import OptimizedLLM

load_dotenv()

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    full_response: str

app = FastAPI(
    title="Optimized 3B LLM Inference API",
    description="An API to serve a highly optimized LLM."
)

@app.on_event("startup")
def load_model():
    print("--- Server is starting up, loading models... ---")
    main_model_id = os.getenv("MAIN_MODEL_ID", "stabilityai/stablelm-3b-4e1t")
    assistant_model_id = os.getenv("ASSISTANT_MODEL_ID", "EleutherAI/pythia-160m-deduped")

    app.state.llm = OptimizedLLM(
        model_id=main_model_id,
        assistant_model_id=assistant_model_id
    )
    print("--- Models loaded and ready to serve requests. ---")

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "message": "Optimized LLM Server is running."}

@app.post("/generate", response_model=GenerationResponse, summary="Generate Text")
def generate_text(request: GenerationRequest):
    if not hasattr(app.state, 'llm') or app.state.llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or still initializing.")

    try:
        full_response = app.state.llm.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens
        )

        generated_text = full_response.replace(request.prompt, "").strip()

        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            full_response=full_response
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))