import torch 
from model import UNet3D
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import streamlit as st
import tempfile
import os

def get_default_device():
    gpu_available = torch.cuda.is_available()
    return torch.device("cuda" if gpu_available else "cpu"), gpu_available

def load_model():
    device, gpu_available = get_default_device()
    model = UNet3D(in_channels=3, out_channels=4)
    model.load_state_dict(torch.load("ckpt.tar", map_location= device)["model"])
    model.to(device)
    model.eval()
    
    return model, device

def predict(model: UNet3D, device, img_tensor: torch.Tensor):
    with torch.no_grad():
        predict = model(img_tensor.to(device).float())

    predict = predict.cpu().argmax(dim = 1).numpy()
    return predict

def load_img_file(img_path: str):
    img_data = np.load(img_path)
    # print(f"Image data shape: {img_data.shape}")
    transformer = transforms.Normalize(mean=[0.5], std=[0.5])
    image = torch.from_numpy(img_data).permute(3,2,0,1).float().unsqueeze(0)
    transformed_image = transformer(image)
    print(transformed_image.shape)
    return transformed_image

def main():    
    st.title("3D U-Net Segmentation")
    st.write("Upload ảnh .npy để thực hiện phân đoạn 3D")
    
    # Upload file
    uploaded_file = st.file_uploader("Chọn file .npy", type=['npy'])
    
    
    if uploaded_file is not None:
        try:
            with st.spinner("Đang xu ly file..."):
                transformed_image = load_img_file(uploaded_file)

                if transformed_image is not None:
                    # Load model
                    with st.spinner("Đang tải model..."):
                        model, device = load_model()
                    st.success("Model đã được tải!")
                    
                    # Dự đoán
                    prediction = predict(model, device, transformed_image)
                    
                    st.success("Hoàn thành dự đoán!")
                    
                    # Hiển thị kết quả
                    st.subheader("Kết quả")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Ảnh gốc**")
                        # Hiển thị slice giữa của ảnh 3D
                        mid_slice = transformed_image.shape[2] // 2
                        fig1, ax1 = plt.subplots(figsize=(6, 6))
                        ax1.imshow(transformed_image[0, 0, mid_slice, :], cmap='gray')
                        ax1.set_title(f"Slice {mid_slice}")
                        ax1.axis('off')
                        st.pyplot(fig1)
                    
                    with col2:
                        st.write("**Kết quả phân đoạn**")
                        fig2, ax2 = plt.subplots(figsize=(6, 6))
                        ax2.imshow(prediction[0, mid_slice, :, :], cmap='viridis')
                        ax2.set_title(f"Prediction Slice {mid_slice}")
                        ax2.axis('off')
                        st.pyplot(fig2)
                
                # # Slider để xem các slice khác
                # st.subheader("Xem các slice khác")
                # slice_idx = st.slider("Chọn slice", 0, img_data.shape[0]-1, mid_slice)
                
                # col3, col4 = st.columns(2)
                
                # with col3:
                #     fig3, ax3 = plt.subplots(figsize=(6, 6))
                #     if img_data.ndim == 4:
                #         ax3.imshow(img_data[slice_idx, :, :, 0], cmap='gray')
                #     else:
                #         ax3.imshow(img_data[slice_idx, :, :], cmap='gray')
                #     ax3.set_title(f"Original - Slice {slice_idx}")
                #     ax3.axis('off')
                #     st.pyplot(fig3)
                
                # with col4:
                #     fig4, ax4 = plt.subplots(figsize=(6, 6))
                #     ax4.imshow(prediction[0, slice_idx, :, :], cmap='viridis')
                #     ax4.set_title(f"Segmentation - Slice {slice_idx}")
                #     ax4.axis('off')
                #     st.pyplot(fig4)
                
              
      
                
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()