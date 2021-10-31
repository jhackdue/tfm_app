import numpy as np
import requests


def is_whitespace(c):
    """
    Indica si un cadena de caracteres se corresponde con un espacio en blanco / separador o no.

    :param c: caracter sobre el cual indicar si es un espacio en blanco o no

    :return Booleano sobre si es un espacio en blanco
    """

    if (c in [" ", "\t", "\r", "\n"]) or (ord(c) == 0x202F):
        return True
    return False


def whitespace_split(text):
    """
    Toma el texto y devuelve una lista de "palabras" separadas segun los
    espacios en blanco / separadores anteriores.

    :param text: Texto que queremos separar por los espacios en blanco

    :return Una lista de palabras en el texto, obviando palabras en blanco
    """
    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens


def create_input_dict(question, context, tokenizer, max_len):
    """
    A partir de una pregunta y un contexto como cadenas, devuelve un diccionario
    con los 3 elementos necesarios para el modelo. También devuelve las context_words,
    context_tok a la correspondencia de ids de context_word y la longitud de
    question_tok que necesitaremos más adelante.

    :param question: Pregunta para el modelo
    :param context: Contexto donde encontrar una respuesta
    :param tokenizer: Tokenizador utilizado en el modelo
    :param max_len: Tamaño de la secuencia máxima de tokens en el contexto

    :returns diccionario input para el modelo, tokens del contexto,
            índices de los tokens y longitud de los tokens de la pregunta.
    """

    import tensorflow as tf

    question_tok = tokenizer.encode(question)

    context_words = whitespace_split(context)

    input_tok = tokenizer.encode(question, context).tokens
    input_ids = tokenizer.encode(question, context).ids
    input_mask = tokenizer.encode(question, context).attention_mask
    input_type_ids = tokenizer.encode(question, context).type_ids

    input_tok += ["[PAD]"]*(max_len-len(input_tok))
    input_ids += [0]*(max_len-len(input_ids))
    input_mask += [0]*(max_len-len(input_mask))
    input_type_ids += [0]*(max_len-len(input_type_ids))

    input_dict = {"input_ids": tf.expand_dims(tf.cast(input_ids, tf.int32), 0),
                  "attention_mask": tf.expand_dims(tf.cast(input_mask, tf.int32), 0),
                  "token_type_ids": tf.expand_dims(tf.cast(input_type_ids, tf.int32), 0)}

    return input_dict, context_words, len(question_tok)


def predict(question, context, modelo=None, tokenizer=None, max_len=384, use_pipeline=False):
    """
    Dado un modelo, una pregunta, un contexto, un tokenizador y un tamaño máximo de secuencia,
    se obtiene la predicción del modelo. Con el flag use_pipeline, podemos usar un modelo
    cargado en HuggingFace ya preentrenado para esta tarea.

    :param question: Pregunta para el modelo
    :param context: Contexto donde encontrar una respuesta
    :param modelo: Modelo con el cual predecir
    :param tokenizer: Tokenizador utilizado en el modelo
    :param max_len: Tamaño de la secuencia máxima de tokens en el contexto
    :param use_pipeline: Flag booleano para usar o no un modelo de HugginFace
    :return: Respuesta predicha a partir de la pregunta y contexto
    """

    if not use_pipeline:

        # Formateamos datos de entrada
        my_input_dict, my_context_words, question_tok_len = create_input_dict(question, context, tokenizer, max_len)

        # Inferencia del modelo

        start_logits, end_logits = modelo(my_input_dict, training=False)

        start_logits_context = start_logits.numpy()[0, question_tok_len + 1:]
        end_logits_context = end_logits.numpy()[0, question_tok_len + 1:]

        pair_scores = np.ones((len(start_logits_context), len(end_logits_context))) * (-1E10)
        for i in range(len(start_logits_context) - 1):
            for j in range(i, len(end_logits_context)):
                pair_scores[i, j] = start_logits_context[i] + end_logits_context[j]
        pair_scores_argmax = np.argmax(pair_scores)

        start_word_id = pair_scores_argmax // len(start_logits_context)
        end_word_id = pair_scores_argmax % len(end_logits_context)

        predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id + 1])

        response = {"start": start_word_id, "end": end_word_id, "answer": predicted_answer}

    else:
        modelo = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        api_token = "api_FicAVZIfHMXrTbDkxCvjpCybuZBkHJaYRx"
        api_url = f"https://api-inference.huggingface.co/models/{modelo}"
        headers = {"Authorization": f"Bearer {api_token}"}
        error = True

        while error:
            response = requests.post(api_url,
                                     headers=headers,
                                     json={"inputs": {"question": question, "context": context}})

            response = response.json()

            if "error" not in list(response.keys()):
                error = False

    return response
