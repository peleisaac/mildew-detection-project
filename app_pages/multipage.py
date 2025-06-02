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
            initial_sidebar_state="expanded",
        )

    def add_page(self, title, func):
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.title("Navigation")

        # Extract just the titles for the radio options
        page_titles = [page["title"] for page in self.pages]

        # Get the selected page title
        selected_title = st.sidebar.radio("Select a page:", page_titles)

        # Find and execute the corresponding function
        for page in self.pages:
            if page["title"] == selected_title:
                page["function"]()
                break
