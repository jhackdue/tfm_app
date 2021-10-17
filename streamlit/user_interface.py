import streamlit as st
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests

img = Image.open("img/imf_logo.png")
st.beta_set_page_config(page_title="TFM-APP", page_icon=img)


# interact with FastAPI endpoint
backend = 'http://localhost:8008/qas/'


def process(context: str, question: str, server_url: str):

    m = MultipartEncoder(
        fields={'context': context, 'question': question}
        )
    r = requests.post(server_url,
                      data=m,
                      params=m.fields,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)
    return r


def main():
    # construct UI layout
    st.title('TFM: Question Answering')

    st.write("Modelo de Question Answering. Visita `url:8008/docs` para la descripción de FastAPI.")

    user_input_context = st.text_area("Contexto:")
    user_input_question = st.text_area("Pregunta:")

    if st.button('¡Obtener respuesta!'):

        if user_input_context and user_input_question:
            result = process(user_input_context, user_input_question, backend)
            res = result.content
            st.write(f'Respuesta:    {res.decode("utf-8") }')

        elif user_input_context:
            # handle case with no image
            st.write("Introduce una pregunta")

        elif user_input_context:
            # handle case with no image
            st.write("Introduce el contexto de la pregunta")

        else:
            # handle case with no image
            st.write("Insertar contexto y pregunta")


if __name__ == "__main__":
    main()
