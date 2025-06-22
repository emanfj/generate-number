import os
import torch
import streamlit as st
from torchvision.transforms import ToPILImage

latent_dim = 100
n_classes  = 10
checkpoint_path = "cgan_checkpoints/generator_epoch100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(torch.nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.label_emb = torch.nn.Embedding(n_classes, embed_dim)
        input_dim = latent_dim + embed_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(1024, 28*28),
            torch.nn.Tanh()
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([noise, emb], dim=1)
        img = self.model(x).view(-1, 1, 28, 28)
        return img

# use the new cache_resource decorator for loading heavy objects
@st.cache_resource
def load_generator():
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gen.eval()
    return gen

generator = load_generator()
to_pil = ToPILImage()

st.title("MNIST cGAN demo")
st.write("choose a digit below and click generate")

digit = st.slider("digit", 0, 9, 0)
if st.button("generate 5 images"):
    # sample noise + labels
    z      = torch.randn(5, latent_dim, device=device)
    labels = torch.full((5,), digit, dtype=torch.long, device=device)

    with torch.no_grad():
        imgs = generator(z, labels)       # in [-1,1]
        imgs = (imgs + 1) * 0.5           # scale to [0,1]

    # convert each to a PIL image on CPU
    pil_imgs = [ to_pil(img.cpu()) for img in imgs ]

    # display 5 separate images with 5 matching captions
    st.image(
        pil_imgs,
        caption=[str(digit)] * 5,
        width=56
    )

st.sidebar.write("run `streamlit run d:/meti-ai/app.py`")
