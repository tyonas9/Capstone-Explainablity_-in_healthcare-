import streamlit as st
from PIL import ImageFile
import openslide

def main():
    st.title("Image upload")

    menu = ["Image"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Image", type=["svs"])
        if image_file is not None:
            # see details
            st.write(type(image_file))



if __name__ == '__main__':
	main()



