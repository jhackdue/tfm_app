# -*- coding: utf-8 -*-

from tensorflow import keras
import string
import re
import numpy as np


def normalizar_texto(txt):
    """
    Minimiza el texto. Quita signos de puntuación y determinantes artículos en español.
    :param txt: Texto que se quiere normalizar
    :return: Texto normalizado
    """
    text = txt.lower()

    # Remove punctuations
    exclude = set(string.punctuation+'¿¡')
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(un|una|el|la|los)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class ExactMatch(keras.callbacks.Callback):
    """
    Cada objeto `SquadExample` contiene offsets de cada caracter para cada token en su párrafo de entrada. Los usamos
    para recuperar el espacio de texto correspondiente a los tokens entre nuestros tokens de inicio y fin predichos.
    Todas las respuestas ground-truth también están presentes en cada objeto `SquadExample`.
    Calculamos el porcentaje de puntos de datos en los que el espacio de texto obtenido a partir de las predicciones
    del modelo coincide con una de las respuestas reales.
    """

    def __init__(self, x_eval, y_eval, eval_squad_examples, logger):
        super().__init__()
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.eval_squad_examples = eval_squad_examples
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in self.eval_squad_examples if not _.skip]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalizar_texto(pred_ans)
            normalized_true_ans = [normalizar_texto(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1

        acc = count / len(self.y_eval[0])

        if self.logger is not None:
            self.logger.info(f"\nepoch={epoch+1}, exact match score={acc:.2f}")

        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")
