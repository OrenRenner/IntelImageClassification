from fastapi import FastAPI, UploadFile, Depends, HTTPException
from fastapi.param_functions import File

from PIL import Image
import io
import os

from classificator.run import Classificator

app = FastAPI(
    title="Intel Image Classification API",
    description="API for Intel Image Classification",
    version="0.0.1",
)


async def upload_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        return pil_image
    except:
        raise HTTPException(status_code=400, detail="Invalid Image File")


@app.post("/recognize/",
          tags=["Recognize"])
async def recognize(
    image: Image.Image = Depends(upload_image)
):
    try:
        return {"class": Classificator("./models/transfer_model.pt")(image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
