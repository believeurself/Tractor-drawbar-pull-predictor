import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.sidebar.title('Tractor Drawbar Pull Predictor')
st.sidebar.image('https://media.gettyimages.com/photos/an-indian-farmer-ploughs-a-rice-paddy-field-with-a-tractor-on-the-of-picture-id977884630?k=20&m=977884630&s=612x612&w=0&h=VdScWaPjeipEA8r6NRROfGH-_3EIy0wiAdw8c7lynjo=')
st.sidebar.header('Predict drawbar pull for your tractor at 15% wheel slip')
st.sidebar.image('https://media.gettyimages.com/photos/indian-employees-stand-near-tractors-at-a-new-mahindra-mahindra-at-picture-id163198690?k=20&m=163198690&s=612x612&w=0&h=m3YiF8hZH0C6IsiQX9bQSz-eXaBJkd5xzyUDWmQvukE=')

# pto power
pto_power = st.number_input('PTO power in kW')

# tire width
tire_width = st.number_input('Tire sectional width in m')

# tire dia
tire_diameter = st.number_input('Tire overall diameter in m')

# drawbar height
drawbar_height = st.number_input('Drawbar Height in m')

# wheelbase
wheel_base = st.number_input('Wheel base in m')

# total mass
total_mass = st.number_input('Total mass in kN')

# velocity
velocity = st.number_input('Velocity in km/h')

# feat = [pto_power, tire_width, tire_diameter, drawbar_height, wheel_base, total_mass, velocity]

if st.button('Predict Drawbar Pull'):
    query = np.array([[pto_power, tire_width, tire_diameter, drawbar_height, wheel_base, total_mass, velocity]])
    query = scaler.transform(query)
#     query = query.reshape(1,7)
#     feat = np.array(feat)
#     feat = feat.reshape(1,len(feat))
    st.header("The predicted maximum Drawbar Pull at 15% wheel slip for this configeration is " + str(np.round(model.predict(query)[0], 2)) + " kN")