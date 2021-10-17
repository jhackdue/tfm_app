from typing import Optional
from fastapi import FastAPI, Query

import json
import os

import api_utils.utils.read_and_write as rw
import api_utils.utils.preprocesado as pp
import api_utils.train.train_utils as tu
import api_utils.predict.predict_utils as pu


# Cargar ficheros de configuración y preprocesado
ABS_PATH = "./"
DATA_DIR = os.path.join(ABS_PATH, "api_utils/data/")
MODELS_DIR = os.path.join(DATA_DIR, "models/")

logger = rw.crear_logger("tfm-app.log")
config_file = rw.cargar_config(logger)

vocab_BERT_path = rw.comprobar_fichero_existe(os.path.join(MODELS_DIR, "vocab.txt"), logger)
config_BERT_path = rw.comprobar_fichero_existe(os.path.join(MODELS_DIR, "config.json"), logger)

tokenizador = pp.obtener_tokenizador(vocab=vocab_BERT_path, lowercase=False)

with open(config_BERT_path, 'r') as json_file:
    config_BERT = json.load(json_file)
config_BERT.update(config_file["train"]["preprocess"])


# Cargar app
app = FastAPI(title="Question Answering",
              description=''' El objetivo es encontrar el espacio de texto en el párrafo 
              que responde a la pregunta planteada.''',
              version="0.1.0")

modelo = tu.cargar_modelo(logger=logger, config=config_BERT)


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
                  use_pipeline: Optional[bool] = False):
    """
    Get question answering

    :param context: Contexto donde encontrar una respuesta
    :param question: Pregunta para el modelo
    :param use_pipeline: Flag booleano para usar o no un modelo de HugginFace
    :return: Respuesta predicha a partir de la pregunta y contexto
    """
    logger.debug("Ejecutar modelo...")

    if context and question:
        result = pu.predict(question, context, modelo=modelo, tokenizer=tokenizador)
        logger.debug("modelo ejecutado...")

        if use_pipeline:
            return result["answer"]
        else:
            return result
    return {"items": "Null"}
