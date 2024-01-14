import gradio as gr
from torchvision import transforms
from PIL.Image import Image
from arquitectura import Arquitectura
from torch.nn import functional as F
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
cartas = ['ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades', 'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades', 'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades', 'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades', 'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades', 'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades', 'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades', 'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades']
red = Arquitectura(numclasses=len(cartas)).to(device)
red.load_state_dict(torch.load("modelo.pth"))
def inferencia(imagen: Image):
    return dict(zip(cartas, [F.softmax(red.forward(t(imagen).unsqueeze(0).to(device)), dim=1)[0][i].item() for i in range(len(cartas))]))

gr.Interface(fn=inferencia, inputs=gr.Image(image_mode="RGB", type="pil"), outputs=gr.Label(num_top_classes=5)).launch(inbrowser=True)