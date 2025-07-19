from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from src.cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

templates = Jinja2Templates(directory="app/templates")

router = APIRouter(
    # prefix="/prediction",
    tags=["Prediction"]
)

@router.post("/predict", response_class=HTMLResponse)
async def make_prediction(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to make predictions on an uploaded image file.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    # content = await file.read()
    # save the image 
    try:
        file_location = f"app/static/temp/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        pipeline = PredictionPipeline(content)
        result = pipeline.predict()
        print(file.filename, result)
        return templates.TemplateResponse("result.html", {
            "request": request,
            "filename": file.filename,
            "prediction": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
   