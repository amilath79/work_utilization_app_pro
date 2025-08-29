import streamlit as st

def load_brand_css():
    """Load Forlagssystem.se brand styling with high priority"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600&family=League+Spartan:wght@300&display=swap');
    
    /* Force override Streamlit default styles */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Brand Colors */
    :root {
        --fs-blue: #0073AE;
        --fs-dark-blue: #0F2436;
        --fs-light-green: #A2BF00;
        --fs-pink: #E780A3;
    }
    
    /* Main title and headers - Montserrat SemiBold, UPPERCASE */
    h1, h2, h3, .stTitle, .main h1, .main h2, .main h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        color: var(--fs-blue) !important;
    }
    
    /* Centered title specifically */
    .centered-title {
        text-align: center !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        color: var(--fs-blue) !important;
        margin: 20px 0 !important;
    }
    
    /* Body text - League Spartan Light */
    p, .stText, .stMarkdown, .main p {
        font-family: 'League Spartan', sans-serif !important;
        font-weight: 300 !important;
        color: var(--fs-dark-blue) !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--fs-blue) !important;
        color: white !important;
        border: none !important;
        font-family: 'League Spartan', sans-serif !important;
    }
    
    .stButton button:hover {
        background-color: var(--fs-dark-blue) !important;
    }
    
    </style>
    """, unsafe_allow_html=True)