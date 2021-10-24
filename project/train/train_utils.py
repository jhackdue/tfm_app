# -*- coding: utf-8 -*-

import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from transformers import TFBertForQuestionAnswering
import sys
sys.path.append("..")
import train.exactMatch as eM  # noqa: E402
import utils.read_and_write as rw  # noqa: E402

ABS_PATH = "..\\"
CHK_PATH = os.path.join(ABS_PATH, "data\\checkpoints\\")
LOGS_PATH = os.path.join(ABS_PATH, "data\\logs\\tensorboard\\")
CACHE_PATH = os.path.join(ABS_PATH, "data\\cache\\")
MODEL_PATH = os.path.join(ABS_PATH, "data\\models\\")


def freeze_model(modelo, logger=None):
    """
    Cambia el atributo de la primera capa del modelo. Pasando a ser no entrenable.

    :param modelo: Modelo de keras inicializado
    :param logger: Fichero de logs de la aplicación

    :return: Modelo con la primera capa congelada
    """
    modelo.layers[0].trainable = False

    if logger is not None:
        logger.info(f"Se ha congelado la primera capa del modelo: {modelo.layers[0].name}")

    return modelo


def crear_modelo(pre_model=None, logger=None, config=None, freeze=True):
    """
    Realiza un finetuning sobre un modelo pre-entrenado. Congela la primera capa (modelo pre-entrenado)
    y deja las últimas capas compiladas para entrenamiento.

    :param pre_model: Nombre del modelo en HuggungFace para usarlo como base.
    :param logger: Fichero de logs de la aplicación
    :param config: Fichero de configuración del modelo.
    :param freeze: Booleano para congelar la capa base.

    :return: Modelo compilado
    """

    # Modelo
    model = TFBertForQuestionAnswering.from_pretrained(pre_model)  # "bert-base-multilingual-cased"

    if logger is not None:
        logger.info(f"Se ha realizado la carga del modelo para QA a partir del modelo preentrenado {pre_model}.")

    if freeze:
        model = freeze_model(model, logger=logger)

    # Optimizer y loss
    lr = config["lr"] if config is not None else 0.005
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.AUTO,
                                                         name='sparse_categorical_crossentropy')

    model.compile(optimizer=optimizer, loss=loss)

    if logger is not None:
        logger.info(f"Se ha compilado el modelo con Adam(lr={lr}) como Optimizador y SparseCategoricalCrossentropy "
                    f"como función de pérdida.")

    return model


def obtener_modelo(pre_model=None, logger=None, config=None, freeze=True):
    """
    Crea el modelo dentro de la GPU, marcado como dispositivo 0.

    :param pre_model: Nombre del modelo en HuggungFace para usarlo como base.
                    Por defecto: "dccuchile/bert-base-spanish-wwm-cased"
    :param logger: Fichero de logs de la aplicación
    :param config: Fichero de configuración del modelo.
    :param freeze: Booleano para congelar la capa base.

    :return: Modelo compilado y el optimizador usado.
    """

    pre_model = "dccuchile/bert-base-spanish-wwm-cased" if pre_model is None else pre_model
    with tf.device('/GPU:0'):
        model = crear_modelo(pre_model=pre_model, logger=logger, config=config, freeze=freeze)

        if logger is not None:
            model.summary(print_fn=logger.info)

    return model


def generar_callbacks(x_eval, y_eval, squad_objects, logger=None, chkpts_path=None, update_freq=500):
    """
    Genera callbacks para el entrenamiento del modelo. Parada del entrenamiento en caso de que en dos epochs seguida
    la función de pérdida no se minimice. Checkpoints para guardar el mejor modelo en cada época. Pantalla de control
    en el formato Tensorboard. Nuestra función ExactMatch, que nos devuelve el accuracy basada lo parecido de las
    respuestas al set de preguntas en SQuAD_v2_esp.

    :param x_eval: Arrays de evaluación
    :param y_eval: Arrays output que indica las posiciones de inicio y fin en el texto donde están las respuestas.
    :param squad_objects: Lista con los objetos squadExample de evaluación
    :param logger: Fichero de logs de la aplicación
    :param chkpts_path: Ruta al fichero de checkpoints donde se almacenarán los resultados del mejor modelo
    :param update_freq: Cada cuántos batches se actualizará el tensorboard. Mientras sea más pequeño, más tardará el
                        entrenamiento

    :return: Lista de los callbacks generados
    """

    # EarlyStopping
    earlystop = tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss', verbose=1)

    # ReduceLR
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3, monitor='loss', verbose=1, factor=0.1, min_lr=0)

    # CheckPoint

    if chkpts_path is None:
        chkpts_path = os.path.join(CHK_PATH, "model.{epoch:02d}-{loss:.2f}.h5")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=chkpts_path, monitor='loss', verbose=1,
                                                    save_best_only=True, save_weights_only=False,
                                                    mode='min', save_freq='epoch')
    # Tensorboard
    logdir = os.path.join(LOGS_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq=update_freq)

    # ExactMatch
    exact_match = eM.ExactMatch(x_eval, y_eval, squad_objects, logger)

    # Total Callbacks
    callbacks = [earlystop, reduce_lr, checkpoint, tensorboard, exact_match]

    return callbacks


def entrenar_modelo(modelo, dataset, config, callbacks, model_name=None, logger=None):
    """
    Entrenaremos el modelo con los datos de entranamiento, los parámetros de configuración y la lista de
    callbacks útiles para el entrenamiento.

    :param modelo: Modelo de Question Answering a entrenar
    :param dataset: Datos en formato de tensores
    :param config: fichero de configuración del modelo
    :param callbacks: lista de callbacks para entrenamiento
    :param model_name: Nombre donde se guardará el modelo
    :param logger: Fichero de logs de la aplicación

    :return: Modelo entrenado (History object)
    """

    epochs = config["nb_epoch"]
    if logger is not None:
        logger.info(f"Se realizará el entrenamiento con {epochs} epochs")

    try:
        mod_hist = modelo.fit(dataset, epochs=epochs, callbacks=callbacks)

        if logger is not None:
            logger.info(f"Entrenamiento finalizado!")

        model_name = "qa_model_squad_v2_esp" if model_name is None else model_name
        model_path = os.path.join(MODEL_PATH, model_name)
        model_path = rw.crear_directorio(model_path, logger)

        pesos_path = os.path.join(model_path, f"{model_name}.h5")

        if logger is not None:
            logger.info(f"Guardando pesos...")
        modelo.save_weights(pesos_path)
        if logger is not None:
            logger.info(f"Se han guardado los pesos en: {pesos_path}")
            logger.info("Guardando Modelo Json...")

        json_path = os.path.join(model_path, f"{model_name}.json")
        with open(json_path, "w") as json_file:
            json_file.write(modelo.to_json())

        if logger is not None:
            logger.info(f"Se ha guardado el modelo json en: {json_path}")

        return mod_hist

    except Exception as e:
        if logger is not None:
            logger.error(e)


def cargar_modelo(model_name=None, pre_model=None, logger=None, config=None, pipeline=False):
    """
    Cargamos el modelo para realizar una predicción.

    :param model_name: Nombre del modelo. "qa_model_squad_v2_esp" por defecto.
    :param pre_model: Nombre del modelo en HuggungFace para usarlo como base.
                    Por defecto: "dccuchile/bert-base-spanish-wwm-cased"
    :param logger: Fichero de logs de la aplicación
    :param config: Fichero de configuración del modelo.
    :param pipeline: Si estamos cargando un modelo ya o no un modelo entrenado.
    :return: modelo cargado
    """

    model_name = "qa_model_squad_v2_esp" if model_name is None else model_name
    model_path = os.path.join(MODEL_PATH, model_name)

    if not pipeline:
        model_path = os.path.join(model_path, f"{model_name}.h5")
        modelo = obtener_modelo(pre_model=pre_model, logger=logger, config=config, freeze=False)
        modelo.load_weights(model_path)

    else:
        modelo = load_model(model_path)

    if logger is not None:
        logger.info(f"Se ha cargado el modelo {model_name} con éxito")

    return modelo
