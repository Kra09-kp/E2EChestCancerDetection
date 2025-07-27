from fastapi import FastAPI,Request, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routers import predict
from cnnClassifier import logger
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting the Chest Cancer Detection API")
    yield
    try:
        if os.path.exists("app/static/temp/"):
            for root, dirs, files in os.walk("app/static/temp/", topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir("app/static/temp/")
        logger.info("Temporary files and folders cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files and folders: {str(e)}")
    logger.info("Chest Cancer Detection API has ended properly")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

logger.info("Chest Cancer Detection API has started")   

@app.get("/")
async def root():
    return {"message": "Welcome to the Chest Cancer Detection API"}

    
@app.get("/train")
async def train():
    os.system("python main.py")
    return {"message": "Training has been initiated successfully"}

@app.get("/dvc")
async def dvc():
    os.system("dvc repro")
    return {"message": "DVC pipeline has been executed successfully"}

@app.get("/about")
async def about():
    return {"message": "This API is designed for Chest Cancer Detection using CNN.\
            It allows users to upload images and receive predictions. Created by Kirti Pogra"}

@app.get("/home", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.include_router(predict.router) 
