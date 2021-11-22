import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import base64

menu=['Home','Play game','Obb or Even']
money=None
st.sidebar.image('casino_pos.gif')
choice = st.sidebar.selectbox('HY CASINO MENU:', menu)
main_bg = "moneyback.jpeg"
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
st.audio('Media/musiz.mp3',format="audio/wav",start_time=35)
#Load your model and check create the class_names list
Model_Path = 'model_save.h5'
class_names = [1000, 10000, 100000, 2000, 20000, 200000, 5000, 50000, 500000]
model = tf.keras.models.load_model(Model_Path)
        # Initialization
if 'money' not in st.session_state:
    st.session_state['money'] = 0
if choice =='Home':
    st.image('panda_girl.gif')  
    st.header("ENJOY THE CASINO NIGHT!")
    
    st.header('  ')
    st.code('feeling bored so I am going to the casino today!')
    

    name = st.text_input("What's your name?")
    if name:
        st.header('Hello ' + name )
        st.title("Ladies and Gentlement!")
        st.title("Welcome to HY Casino!")
        st.slider('How excited are you?')
        st.header("Do you want to play a game?")
        col1, col2 = st.columns(2)
        with col1:
            c1=st.button('Yes')
            
        with col2:
            c2=st.button('Very Yes')
        if (c1==True or c2==True):
            st.write('Please go to play game page')  
elif choice == 'Play game':
    st.image('giphy_betmoney.gif')
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
        st.write(prediction[0].max())
        if prediction[0].max() >= 0.6:
            st.write(f'You bet {class_names[index]}  VND to play "Obb or Even"')
        else:
            st.write("Something wrong!!!")
        money =class_names[index]
        st.session_state['money'] = class_names[index]
        
else: 
        st.image('mel-gibson-stupid.gif')
        money=st.session_state['money']
        st.header('ODD OR EVEN?')
        choiz = st.radio("Do it! Do it now! Don't wait!!!!!!!!!!!!!",['Stop','Odd','Even'])
        if choiz=='Odd':
            with st.spinner("Training ongoing"):
                number = np.random.random_integers(100)
                st.header(number)
                if number%2==0:
                    result='Even'
                else:
                    result='Odd'
                if choiz==result:
                    st.write('Congratulations! You win!')
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
                    result='Odd'
                if choiz==result:
                    st.balloons()
                    st.session_state['money']*=2
                    choiz='Stop'
                else: 
                    st.session_state['money']=0
                    st.write('You lose!')   
                    pl=False       
        st.write('You have: ',st.session_state['money'])  
        