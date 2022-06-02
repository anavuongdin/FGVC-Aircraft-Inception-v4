import streamlit as st
import os
from PIL import Image
import torch
from timm.models.inception_v4 import inception_v4
from torchvision import transforms
import torchvision.transforms.functional as fn


# Setup Inception-v4 model
model = inception_v4(num_classes=105)
checkpoint = torch.load('/content/drive/MyDrive/FGVC-Aircraft/checkpoint-72.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

with open('/content/FGVC-Aircraft-Inception-v4/data/fgvc-aircraft-2013b/data/variants.txt', 'rb') as f:
  classes = list(map(lambda x: x.decode("utf-8").strip('\n'), f.readlines()))

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
  st.title("Fine-grained aircraft classification")

  menu = ["Image","Dataset","DocumentFiles","About"]
  choice = st.sidebar.selectbox("Menu",menu)

  if choice == "Image":
	  image_file = st.file_uploader("Upload Images",
			type=["png","jpg","jpeg"])

  if image_file is not None:
			    # TO See details
          file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
          st.write(file_details)
          img = load_image(image_file)
          convert_tensor = transforms.ToTensor()
          resized_img = fn.resize(img, size=[299, 299])

          inp = convert_tensor(resized_img).unsqueeze(0)

          output = model(inp)
          prob = output[0].view(-1)
          label = prob.argmax()
          labels = list(prob.topk(5).indices)

          st.image(img, width=250)
			  
			  # Saving upload
          with open(os.path.join("/content/",image_file.name),"wb") as f:
            f.write((image_file).getbuffer())
			  
        # Give prediction

          st.success("Variant of the aircraft is: {}".format(classes[label.item()]))

  elif choice == "Dataset":
	  st.subheader("Dataset")

  elif choice == "DocumentFiles":
	  st.subheader("DocumentFiles")
  
main()