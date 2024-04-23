import httpx
import streamlit as st

from sources.settings import Settings

settings = Settings()


def show_training():
    st.write("## Training")
    file = st.file_uploader("Upload CSV", type=("csv",))

    if st.button("Training"):
        if file is None:
            return

        response = httpx.post(
            settings.get_url("training"),
            files={"file": file},
        )
        if response.is_success:
            st.write("Training success")
            return

        st.write(response.content.decode())


def show_model():
    st.write("## Prediction")
    message = st.text_area("Message")

    if st.button("Predict"):
        response = httpx.post(
            settings.get_url("predict"),
            json={"message": message},
        )
        if response.is_success:
            st.write(response.json()["emotion"])
            return

        st.write(response.content.decode())


def show_generation():
    st.write("## Generation")
    emotion = st.selectbox("Emotion", ("positive", "negative"))

    if st.button("Generate"):
        response = httpx.get(settings.get_url(f"model?emotion={emotion}"))

        if response.is_success:
            st.write(response.json()["message"])
            return

        st.write(response.content.decode())


def main():
    st.write("# IA tweet plane company")

    show_training()
    show_model()
    show_generation()


if __name__ == "__main__":
    main()
