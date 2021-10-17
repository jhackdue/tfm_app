# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
import sys
sys.path.append("..")
import api_utils.utils.squadExample as sE  # noqa: E402


ABS_PATH = "./api_utils/"
CACHE_PATH = os.path.join(ABS_PATH, "data/cache/")
CONFIG_PATH = os.path.join(ABS_PATH, "config/config.yml")


def obtener_tokenizador(*args, **kargs):
    """
    Utilizamos los mismos parámetros que la función BertWordPieceTokenizer,
    que a partir de un vocabulario, ya nos permite tokenizar nuestros textos
    :param args, kargs: Parámetros por defecto de la función Tokenizadora de HuggingFace

    :return BertWordPieceTokenizer: Tokenizador de la función Tokenizadora de HuggingFace
    """

    return BertWordPieceTokenizer(*args, **kargs)


def iniciar_squad_example(tokenizer, config, idqa, question, context, ans_start, text):
    """
    Función para inicializar el objeto SquadExample y preprocesarlo.

    :param tokenizer: Tokenizador usado para el entrenamiento
    :param config: Fichero de configuración sobre el modelo
    :param idqa: Id de la respuesta.
    :param question: Texto con una pregunta
    :param context: Contexto sobre el que se hace la pregunta
    :param ans_start: Inicio del caracter donde se encuentra la respuesta
    :param text: Texto con la respuesta a la pregunta

    :return result: Objeto de la clase SquadExample.
    """

    result = sE.SquadExample(tokenizer, config, idqa, question, context, ans_start, text, [text])
    result.preprocess()

    return result


def crear_squad_example(raw_df, tokenizer, config):
    """
    Función para obtener objetos de la clase SquadExample a partir de los datos en un dataframe.

    :param raw_df: Dataframe con los registros del dataset SQuAD v2 en español.
    :param tokenizer: Tokenizador usado para el entrenamiento
    :param config: Fichero de configuración sobre el modelo

    :return squad_examples: Lista con los objetos de la clase SquadExample
    """
    squad_examples = raw_df.apply(lambda x: iniciar_squad_example(tokenizer, config, x["Id"],
                                                                  x["Question"], x["Context"],
                                                                  x["Ans_start"], x["Text"]), axis=1)
    squad_examples = squad_examples.to_list()

    return squad_examples


def crear_inputs_targets(squad_examples):
    """
    Función para crear los inputs del modelo a partir de los objetos SquadExample.

    :param squad_examples: Lista con los objetos de la clase SquadExample

    :return x, y, dataset_tf, errors: x_features, y_targets, dataset con registros en forma de tensores y
                                      errores del preprocesado de objetos de la clase SquadExample.
    """

    dataset_dict = {"input_ids": [],
                    "token_type_ids": [],
                    "attention_mask": [],
                    "start_token_idx": [],
                    "end_token_idx": []}

    errors = []

    for item in squad_examples:
        if not item.skip:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
        else:
            errors.append(getattr(item, "idqa"))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]

    dataset_tf = tf.data.Dataset.from_tensor_slices(
        ({key: tf.cast(dataset_dict[key], dtype="int32") for key in ["input_ids", "attention_mask", "token_type_ids"]},
         {key: tf.cast(dataset_dict[key], dtype="int32") for key in ['start_token_idx', 'end_token_idx']}))

    return x, y, dataset_tf, errors


def transformar_datos_squad(raw_df, tokenizer, config, logger=None, name_data=None):
    """
    Función para transformar las filas de datos del SQuAD-es-v2 en objetos SquadExample.

    :param raw_df: Dataframe con los registros del dataset SQuAD v2 en español.
    :param tokenizer: Tokenizador usado para el entrenamiento
    :param config: Fichero de configuración sobre el modelo
    :param logger: Fichero donde almacenamos los logs de la app
    :param name_data: Informar si los datos son de entrenamiento o de validación.

    :return x, y, dataset_tf, sq_objects, errores: x_features, y_targets, dataset in tensor formats
    """
    try:
        squad_objects = crear_squad_example(raw_df, tokenizer, config)
        if logger is not None:
            name_data = "datos" if logger is None else name_data
            logger.info(f"Transformado el conjunto de {name_data} al formato SquadExample")

        x, y, dataset, errores = crear_inputs_targets(squad_objects)

        if logger is not None:
            name_data = "datos" if logger is None else name_data
            logger.info(f"Se ha conseguido obtener los inputs y los targets para el conjunto de {name_data}. "
                        f"Se han creado  {len(dataset)} puntos")

        return x, y, dataset, squad_objects, errores

    except Exception as e:
        if logger is not None:
            logger.error(e)


def input_formato_keras(dataset, config, file_cache=None, logger=None, shuffle_buffer=1000):
    """
    Realiza la transformación del dataset en un conjunto de batches. Teniendo ahora el formato:
    ({tensor_inputids, tensor_attention_mask, tensor_token_type_ids}, (tensor_token_start_idx, tensor_token_end_idx))

    :param dataset: dataset in tensor formats
    :param config: Fichero de configuración sobre el modelo
    :param file_cache: Nombre del fichero donde se cacheará el dataset. "datacached.*" por defecto.
    :param logger: Fichero donde almacenamos los logs de la app
    :param shuffle_buffer: Ver https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle

    :return: Dataset en el formato descrito
    """
    if file_cache is None:
        file_cache = os.path.join(CACHE_PATH, "datacached")
    else:
        file_cache = os.path.join(CACHE_PATH, file_cache)

    ds_format = dataset.cache(filename=file_cache).shuffle(shuffle_buffer).batch(config["batch_size"])
    ds_format = ds_format.map(lambda x, y: (x, (y["start_token_idx"], y["end_token_idx"])))

    return ds_format
