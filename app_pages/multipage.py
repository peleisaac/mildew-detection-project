import streamlit as st

class MultiPage:
    def __init__(self, app_name="SmartLeaf Dashboard"):
        self.pages = []
        self.app_name = app_name

        # Set Streamlit page configuration
        st.set_page_config(
            page_title=self.app_name,
            page_icon="🌿",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def add_page(self, title, func):
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select a page:", 
            self.pages, 
            format_func=lambda p: p["title"]
        )
        page["function"]()
