import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from arquitectura import Arquitectura

device = "cuda" if torch.cuda.is_available() else "cpu"

# Transformaciones de las imagenes
transformaciones = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Cargamos el dataset

dataset = datasets.ImageFolder(root="Dataset/train", transform=transformaciones)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instanciamos el modelo

modelo = Arquitectura(numclasses=len(dataset.classes)).to(device)

# Definimos la funcion de perdida y el optimizador

criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Entrenamiento

precisiones = []

for epoch in range(35):
    modelo.load_state_dict(torch.load("modelo.pth"))
    correctas = 0
    total = 0
    for imagenes, etiquetas in tqdm(loader):
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)
        salidas = modelo.forward(imagenes)
        optimizador.zero_grad()
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
        _, predicciones = torch.max(salidas, 1)
        total += etiquetas.size(0)
        correctas += (predicciones == etiquetas).sum().item()
    precision = correctas / total
    precisiones.append(precision)
    print(f"Epoca: {epoch + 1} Perdida: {perdida.item()} Precision: {precision}")
    if precision == max(precisiones):
        torch.save(modelo.state_dict(), "modelo.pth")
        print("Mejor modelo guardado")
