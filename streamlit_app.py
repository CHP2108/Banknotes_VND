import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import base64

menu=['Home','Play game','Obb or Even']
money=None
choice = st.sidebar.selectbox('HY CASINO MENU:', menu)
main_bg = "Media/1190722.jpg"
main_bg_ext = "jpg"

side_bg = "Media/ab.jpg"
side_bg_ext = "jpg"

st.markdown(
f"""
<style>
.reportview-container {{
    background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
}}
.sidebar-content {{
    background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
}}
</style>
""",
unsafe_allow_html=True)
st.audio('Media\musiz.mp3',format="audio/wav",start_time=35)
#Load your model and check create the class_names list
Model_Path = 'Model\Result\model_ckpt.h5'
class_names = [1000, 10000, 100000, 2000, 20000, 200000, 5000, 50000, 500000]
model = tf.keras.models.load_model(Model_Path)
if choice =='Home':
    
    

    name = st.text_input("What's your name?")
    if name:
        st.header('Hello ' + name )
        st.title("Welcome to HY Casino!")
        st.header("Do you want to play a game?")
        col1, col2 = st.columns(2)
        with col1:
            c1=st.button('Yes')
        with col2:
            c2=st.button('Very Yes')
        if (c1==True or c2==True):
            st.write('Please go to play game page')  
elif choice == 'Play game':
    st.title("How much money you want to bet?")
    cap = cv2.VideoCapture(0)  # device 0
    run = st.checkbox('Show Webcam')
    capture_button = st.button('Capture')

    captured_image = np.array(None)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while run:
        ret, frame = cap.read()        
        # Display Webcam
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ) #Convert color
        FRAME_WINDOW.image(frame)

        if capture_button:      
            captured_image = frame
            break

    cap.release()

    if  captured_image.all() != None:
        st.image(captured_image)
        st.write('Image is capture:')

        #Resize the Image according with your model
        captured_image = cv2.resize(captured_image,(224,224))
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(captured_image, axis=0)
        #Check the img_array here
        # st.write(img_array)
        prediction = model.predict(img_array)
        index=prediction[0].argmax()
        st.write(f'You bet {class_names[index]}  VND to play "Obb or Even"')
        money =class_names[index]
        # Initialization
        if 'key' not in st.session_state:
            st.session_state['money'] = class_names[index]
        
else: 
        money=st.session_state['money']
        choiz = st.radio("Obb or Even?",['Stop','Obb','Even'])
        if choiz=='Obb':
            with st.spinner("Training ongoing"):
                number = np.random.random_integers(100)
                st.header(number)
                if number%2==0:
                    result='Even'
                else:
                    result='Obb'
                if choiz==result:
                    st.balloons()
                    st.session_state['money']*=2
                    choiz='Stop'
                else: 
                    st.session_state['money']=0
                    st.write('You lose!')
                    pl=False
        elif choiz=='Even':
            with st.spinner("Training ongoing"):
                number = np.random.random_integers(100)
                st.header(number)
                if number%2==0:
                    result='Even'
                else:
                    result='Obb'
                if choiz==result:
                    st.balloons()
                    st.session_state['money']*=2
                    choiz='Stop'
                else: 
                    st.session_state['money']=0
                    st.write('You lose!')   
                    pl=False       
        st.write('You have: ',st.session_state['money'])  
        
#     image_upload = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
#     st.image(image_upload)
#     if image_upload !=None:
#         image_upload = cv2.resize(image_upload,(299,299))
#         #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
#         img_array  = np.expand_dims(image_upload, axis=0)
#         #Check the img_array here
#         # st.write(img_array)
#         prediction = model.predict(img_array)
#         index=prediction[0].argmax()
#         st.write(index)
#         st.write(class_names[index])
        
        # Preprocess your prediction , How are we going to get the label name out from the prediction
        # Now it's your turn to solve the rest of the code