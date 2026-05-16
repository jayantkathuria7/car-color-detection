import streamlit as st
from utils.pipeline import handle_image, handle_video
import io
from PIL import Image
# def display_image(image, title='Image'):
#     plt.figure(figsize=(6, 6))
#     if len(image.shape) == 4:
#         image = image[0]
#     if image.shape[-1] == 1:
#         plt.imshow(image.squeeze(), cmap='gray')
#     else:
#         plt.imshow(image)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()


def main():
    st.title("Car Color Detection")

    with open("assets/project_notes.md", "r") as f:
        content = f.read()

    with st.expander("ℹ️ Customization Rules & Project Notes"):
        st.markdown(content)

    st.write(
        "Upload an image or video to analyze traffic. The app will predict car colors, count cars, and detect people.")

    upload_option = st.radio("Select Upload Type:", ("Image", "Video"))

    
    if upload_option == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_image:
            output_image_arr, counts = handle_image(uploaded_image)
            output_image = Image.fromarray(output_image_arr)
            dynamic_width = st.slider("Select image width in pixels", min_value=100, max_value=1200, value=600)
            st.write(f"Number of cars detected: {counts['car_count']} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Number of other vehicles detected: {counts['other_vehicle_count']}")
            st.write(f"Number of males detected: {counts['male_count']} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Number of females detected: {counts['female_count']}")
            st.image(output_image, caption='output', width=dynamic_width)
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            buf.seek(0) 
            st.download_button(
                label="Download Image",
                data=buf,
                file_name="generated_image.png",
                mime="image/png"
            )      
              
    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type="mp4")
        if uploaded_video:
            output_video_path = handle_video(uploaded_video)
            st.video(output_video_path)
            with open(output_video_path,'rb') as file:
                video_bytes = file.read()
            st.download_button(
                label="Download Video",
                data=video_bytes,
                file_name="generated_video.mp4",
                mime="video/mp4"
            ) 
        
if __name__ == "__main__":
    main()
