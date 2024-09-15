from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io


app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = torch.jit.load("scripted-plant-disease-model.pth") 
    model.eval()  

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  

@app.get("/predict_local")
async def predict_local():
    try:
        image_path = "./test/AppleCedarRust2.JPG"  
        img_tensor = transform_image(image_path)
        
        with torch.no_grad():
            output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        
        return {"predicted_class": predicted.item()}
    except Exception as e:
        print(e)
        return {"error": str(e)}
                                                            
