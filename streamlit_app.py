import streamlit as st
from st_pages import Page, show_pages

show_pages(
    [
        Page("streamlit_app.py", "Home", "🏠"),
        Page("./pages/1_Exploratory_Data_Analysis.py", "Exploratory Data Analysis", "🗺️"),
        Page("./pages/2_Model_Prediction.py", "Model Prediction", "🔮"),
    ]
)


if __name__ == "__main__":
    st.header("Cian Flat Cost Prediction Project 🦜")
    st.divider()
    st.subheader("👈 In the tab panel you can see several options to click.")
    st.divider()
    st.subheader(
        "To check out the EDA and try to make a prediction with machine learning models, please, follow the according tab."
    )
    st.divider()
    st.subheader("❗ Note, that all the graphics are interactive, you can round the data and zoom in/out.")
