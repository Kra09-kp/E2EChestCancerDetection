import torch
from PIL import Image
from torchvision.transforms import v2 as T
import os
from io import BytesIO
from cnnClassifier import logger

model = torch.load(os.path.join("artifacts", "train_model", "best_model.pth"))
# model = torch.load(os.path.join("Model","final_model.pth"))
model.eval()

logger.info("Model loaded successfully")

transform = T.Compose([
            T.Resize((224, 224)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

class_map = {
    0: "adenocarcinoma",
    1: "large.cell.carcinoma",
    2: "normal",
    3: "squamous.cell.carcinoma"
}



class PredictionPipeline:
    def __init__(self,content:bytes ):
        self.content = content

    def predict(self):
        """ Predict the class of the image.
        Returns:
            list: A list containing the predicted class and confidence score.
        """
        # load image
         
        test_img = Image.open(BytesIO(self.content)).convert("RGB")
        test_img = transform(test_img).unsqueeze(0)  #add batch dimension
        logger.info("Image loaded and transformed successfully")
        # make prediction
        try: 

            with torch.no_grad():
                output = model(test_img)
                _, predicted = torch.max(output, 1)
                class_index = predicted.item()
                confidence = output[0][class_index].item()
                logger.info(f"Prediction made successfully: {class_index} with confidence {confidence}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e


        return [{
            "image": class_map.get(int(class_index), "unknown"),
            "confidence": round(confidence, 4)
        }]





    
