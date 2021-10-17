# -*- coding: utf-8 -*-

from pathlib import Path
import logging
import os
import yaml
import pandas as pd
import json


ABS_PATH = "./api_utils/"
LOGS_PATH = os.path.join(ABS_PATH, "data/logs/")
CONFIG_PATH = os.path.join(ABS_PATH, "config/config.yml")


def cargar_config(logger=None):
    """
    Cargamos el fichero de configuración de la aplicación.

    :param logger: Fichero donde almacenamos los logs de la app

    :return config Fichero de configuración
    """
    with open(CONFIG_PATH, "r") as yml_file:
        try:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)
            if logger is not None:
                logger.info("Fichero de configuración de la aplicación cargado")

        except Exception as e:
            if logger is not None:
                logger.error(e)
            raise e

    return config


def comprobar_fichero_existe(file_path, logger=None):
    """
    Verifica que el fichero que se pasa existe.
    Si existe, se devuelve la ruta al fichero, si no,
    se crea el directorio padre donde se almacenará
    :param file_path: Ruta al fichero a comprobar
    :param logger: Fichero donde almacenamos los logs de la app

    :return file_path: Ruta al fichero a comprobar
    """

    file_path_name = Path(file_path).name
    file_path_parent = Path(file_path).parent.absolute()

    if Path(file_path).is_file():
        if logger is not None:
            logger.info(f"El fichero {file_path_name} existe en {file_path_parent}")

    else:
        if logger is not None:
            logger.info(f"El fichero {file_path_name} no existe. "
                        f"Se verificará/creará el directorio padre: {file_path_parent}")
        crear_directorio(file_path_parent)

    return file_path


def crear_directorio(path, logger=None):
    """
    Verifica que existe el directorio que se pasa como argumento, si no, se crea.
    :param path: Ruta al directorio
    :param logger: Fichero donde almacenamos los logs de la app

    :return path: Ruta al directorio
    """

    try:
        os.makedirs(path, exist_ok=True)

    except Exception as e:
        if logger is not None:
            logger.error(e)

        raise e

    return path


def crear_logger(log_name, directory=None, config=None):
    """
    Verifica que existe el fichero de logs que se pasa como argumento, si no, se crea.
    :param log_name: Nombre que recibe el fichero de logs a crear/verificar
    :param directory: Ruta donde se almacenará el fichero de logs.
    :param config: Fichero de configuración de la aplicación.

    :return logger: Fichero donde almacenamos los logs de la app
    """

    config = cargar_config() if config is None else config

    if log_name not in logging.Logger.manager.loggerDict:

        logger = logging.getLogger(log_name)

        logger.setLevel(config['logs'].get("level"))

        format_str = config['logs'].get("format")
        formatter = logging.Formatter(format_str)

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = 0

        directory = LOGS_PATH if directory is None else directory

        crear_directorio(directory)
        handler = logging.FileHandler(os.path.join(directory, log_name))
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.info(f"El fichero log {log_name} ha sido creado")

    logger = logging.getLogger(log_name)

    return logger


def json_to_dataframe(file):
    """
    Código basado en https://www.kaggle.com/jagannathpatta/reading-json-data-getting-dataframe
    :param file: Ruta al fichero json con datos para el modelo

    :return final_df: Dataframe formado con los valores del fichero json.
    """

    f = open(file, "r")
    data = json.loads(f.read())
    iid = []
    tit = []
    con = []
    que = []
    ans_st = []
    txt = []

    for i in range(len(data['data'])):
        title = data['data'][i]['title']
        for p in range(len(data['data'][i]['paragraphs'])):
            context = data['data'][i]['paragraphs'][p]['context']
            for q in range(len(data['data'][i]['paragraphs'][p]['qas'])):
                question = data['data'][i]['paragraphs'][p]['qas'][q]['question']
                id_q = data['data'][i]['paragraphs'][p]['qas'][q]['id']
                for a in range(len(data['data'][i]['paragraphs'][p]['qas'][q]['answers'])):
                    ans_start = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['answer_start']
                    text = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['text']

                    tit.append(title)
                    con.append(context)
                    que.append(question)
                    iid.append(id_q)
                    ans_st.append(ans_start)
                    txt.append(text)

    new_df = pd.DataFrame(columns=['Id', 'Title', 'Context', 'Question', 'Ans_start', 'Text'])
    new_df.Id = iid
    new_df.Title = tit
    new_df.Context = con
    new_df.Question = que
    new_df.Ans_start = ans_st
    new_df.Text = txt

    final_df = new_df.drop_duplicates(keep='first')
    return final_df