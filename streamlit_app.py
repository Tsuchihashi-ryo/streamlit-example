import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from PIL import Image
import numpy as np
import torch
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")


@st.cache_resource
def get_predictor():
  sam_checkpoint = "sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  if torch.cuda.is_available():
      device = 'cuda'
  else:
      device = 'cpu'

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)

  predictor = SamPredictor(sam)
  return predictor

def get_predictor_():
  return None

predictor = get_predictor()



def get_show_image(cv2_img):
  # グレースケール化
  gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

  # 2値化
  ret,thresh = cv2.threshold(gray, 244, 255, cv2.THRESH_BINARY)

  thresh = cv2.bitwise_not(thresh)

  # 輪郭検出
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


  # 輪郭を囲む矩形を取得
  x,y,w,h = cv2.boundingRect(np.concatenate(contours))

  # 画像をトリミング
  cropped_img = cv2_img[y:y+h, x:x+w]

  return cropped_img

def scale_to_height(img, height):
    """高さが指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))

    return dst

def scale_box(img, width, height):
    """指定した大きさに収まるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)

    dst = cv2.resize(img, dsize=(nw, nh))
    return dst

def adjust(img, alpha=1.0, beta=0.0):
    # 積和演算を行う。
    dst =  img * alpha  + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 1, 254).astype(np.uint8)


@st.cache_data
def get_image(uploaded_file,width,height):
  image = np.array(Image.open(uploaded_file))
  image = scale_box(image, width, height)

  
  image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
  image_gray = adjust(image_gray,1.0,100)
  

  predictor.set_image(image)

  return image,image_gray


@st.cache_data
def get_masks(x,y):
  input_point = np.array([[x, y]])
  input_label = np.array([1])
  
  
  masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
  )

  mask_images = [get_show_image(np.where(mask[:, :, np.newaxis] == 0, 255, image)) for mask in masks]
  return masks,mask_images


@st.cache_data
def get_dummy_masks(x,y,r):
  masks = []
  for i in range(3):
    # 全要素がFalseの配列を作成
    arr = np.zeros((x, y), dtype=bool)

    # 中心座標を計算
    c = np.random.randint(-100,100)
    center_x = x // 2 + c
    center_y = y // 2 + c

    circle = 50*(i+1)

    # 中心にTrueを配置
    arr[center_x-circle:center_x+circle, center_y-circle:center_y+circle] = True
    masks.append(arr)

  mask_images = [get_show_image(np.where(mask[:, :, np.newaxis] == 0, 255, image)) for mask in masks]

  return masks,mask_images


def clickable_image(img,height):
  return streamlit_image_coordinates(Image.fromarray(img),height=height)


def next_command(mask_list,masks,mask_images,option,value):
  st.session_state.value = value

  if option == "Option 1":
    mask_list.append((masks[0],mask_images[0]))

  elif option == "Option 2":
    mask_list.append((masks[1],mask_images[1]))

  elif option == "Option 3":
    mask_list.append((masks[2],mask_images[2]))

  elif option == "Miss":
    pass

  else:pass

def delete_command():
  st.session_state.mask_list = [k for i,k in enumerate(st.session_state.mask_list) if i in selector]


def get_output(image,image_gray):
  image_mask = [mask for mask,_ in st.session_state.mask_list]
  all_mask = np.any(np.array(image_mask),axis=0)
  counter_mask = np.where(all_mask[:, :, np.newaxis] == 0, 255, 0).reshape(all_mask.shape[:2]).astype("uint8")
  contours,_ = cv2.findContours(counter_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


  output_image = np.where(all_mask[:, :, np.newaxis] == 0, image_gray, image)

  cv2.drawContours(output_image, contours, -1, (0,255,0), 3)
  return output_image


#######################################################################################


image_size = 1024 #st.slider("Select image height (small to LOW spec, large to HIGH spec...)", min_value=200, max_value=1600, value=800, step=200)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

selectTab, maskTab, outputTab = st.tabs(["Select", "Mask", "Output"])

placeholder = selectTab.empty()

if "value" not in st.session_state:
  st.session_state.value = False

if "mask_list" not in st.session_state:
  st.session_state.mask_list = []

if uploaded_file is not None:
  # 画像の読み込み
  image, image_gray = get_image(uploaded_file,image_size,image_size)
  height,width = image.shape[:2]


  if "image" not in st.session_state:
    st.session_state.image = uploaded_file
  else:
    if st.session_state.image == uploaded_file:
      pass
    else:
      st.session_state.image = uploaded_file
      st.session_state.mask_list = []


  if len(st.session_state.mask_list):
    with maskTab:
      COL_n = 8
      cols = st.columns(COL_n)
      for i,(_,img) in enumerate(st.session_state.mask_list):
        cols[i%(COL_n-1)].image(img,caption=i,use_column_width=True)
      selector = cols[-1].multiselect(
        'Delete Select',
        list(range(len(st.session_state.mask_list))),
        list(range(len(st.session_state.mask_list))),
        on_change=delete_command)
      cols[-1].button("Reload",on_click=delete_command)
          
    output_image = get_output(image,image_gray)
    

    with outputTab:
      col1,col2 = st.columns([8,2])
      col1.image(output_image,use_column_width=True)


      
      if col2.button("Create Image"):
        cv2.imwrite("output.jpg",gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        with open("output.jpg", "rb") as file:
          btn = col2.download_button(
                  label="Download image",
                  data=file,
                  file_name="output.jpg")


  
  with placeholder.container():
    value = clickable_image(image,height)

    if not value:
      st.stop()

#################################################################

  if st.session_state.value != value:
    
    placeholder.empty()

    x = value["x"]
    y = value["y"]

    masks,mask_images = get_masks(x,y)

    #masks,mask_images = get_dummy_masks(*image.shape[:2],x)


    with placeholder.container():
      cols = st.columns(4)
      for i,(col,m_img) in enumerate(zip(cols[:3],mask_images)):
        col.image(Image.fromarray(m_img), caption=f'Option {i+1}',use_column_width=True)
      
      
      option = cols[-1].radio('Which mask?',('Option 1', 'Option 2', 'Option 3',"Miss"),)
        
      cols[-1].button("Add Select",on_click=lambda :next_command(st.session_state.mask_list,masks,mask_images,option,value))
