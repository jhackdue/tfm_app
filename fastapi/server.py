from typing import Optional
from fastapi import FastAPI, Query
import api_utils.utils.read_and_write as rw
import api_utils.predict.predict_utils as pu

logger = rw.crear_logger("tfm-app.log")

# Cargar app
app = FastAPI(title="Question Answering",
              description=''' El objetivo es encontrar el espacio de texto en el párrafo 
              que responde a la pregunta planteada.''',
              version="0.1.0")


@app.get("/")
def read_root():
    """
    Mensaje de bienvenida
    :return: Devuelve mensaje de bienvenida
    """
    return {"message": "Mensaje de bienvenida desde la API"}


@app.post("/qas/")
async def get_qas(context: str = Query(..., min_length=3),
                  question: str = Query(..., min_length=3),
                  use_pipeline: Optional[bool] = True):
    """
    Obtienes las respuestas a las preguntas dentro del contexto dado.

    :param context: Contexto donde encontrar una respuesta
    :param question: Pregunta para el modelo
    :param use_pipeline: Flag booleano para usar o no un modelo de HugginFace
    :return: Respuesta predicha a partir de la pregunta y contexto
    """
    logger.debug("Ejecutar modelo...")

    if context and question:
        result = pu.predict(question, context, use_pipeline=use_pipeline)
        logger.debug("modelo ejecutado...")

        return result["answer"]
    return {"items": "Null"}
