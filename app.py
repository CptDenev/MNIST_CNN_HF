import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# model architecture (CNN same as pth)
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# --- Load ---
model = MNISTCNN()
model.load_state_dict(torch.load("mnist_cnn_final.pth", map_location="cpu", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

def predict(img):
    if img is None:
        return "upload an image with a digit (0-9)"
    x = transform(Image.fromarray(img)).unsqueeze(0)  # [1,1,28,28]
    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(dim=1).item()
    conf = torch.softmax(logits, dim=1)[0][pred].item()
    return f"prediction : **{pred}** (confidence {conf:.1%})"

# --- UI Gradio ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="digit image"),
    outputs=gr.Textbox(label="result"),
    title="MNIST CNN — 12 Mo, CPU only",
    description="upload a 28×28 px image (or above, we'll resize it)",
)

demo.launch(server_name="0.0.0.0", server_port=7860)