import streamlit as st
import os

def main():
    st.title("Image upload")

    menu = ["Image"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Upload Images")
        image_file = st.file_uploader("Upload an Image", type=["svs"])
        if image_file is not None:
            # see image details
            file_details = {"FileName":image_file.name, "FileType":image_file.type}

            #Saving image file in a WSI folder
            with open(os.path.join("WSI",image_file.name),"wb") as f:
                f.write(image_file.getbuffer())
            
            st.success("WSI image has been uploaded")

if __name__ == '__main__':
	main()



