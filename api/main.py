from fastapi import FastAPI, HTTPException, UploadFile, File
from api.schemas import PredictRequest, PredictResponse, BatchPredictResponse
from api.predict import predict_single, predict_batch
import traceback

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service for predicting customer churn using trained ML models.",
    version="1.0.0",
)

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        result = predict_single(request)
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(exc)}\n{tb}")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(file: UploadFile = File(...)):

    try:
        results = await predict_batch(file)
        return {"results": results}
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(exc)}\n{tb}")