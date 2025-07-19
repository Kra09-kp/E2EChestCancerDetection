from fastapi import FastAPI,Request, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routers import predict
from cnnClassifier import logger
from contextlib import asynccontextmanager
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

logger.info("Chest Cancer Detection API has started")   

@app.get("/")
async def root():
    return {"message": "Welcome to the Chest Cancer Detection API"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting the Chest Cancer Detection API")
    yield
    os.rmdir("app/static/temp/") if os.path.exists("app/static/temp/") else None  
    logger.info("Chest Cancer Detection API has ended properly")
    
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
