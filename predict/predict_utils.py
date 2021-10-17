import numpy as np
import tensorflow as tf
from transformers import pipeline


def is_whitespace(c):
    """
    Indica si un cadena de caracteres se corresponde con un espacio en blanco / separador o no.
    """

    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_split(text):
    """
    Toma el texto y devuelve una lista de "palabras" separadas segun los
    espacios en blanco / separadores anteriores.
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

# def tokenize_context(text_words, tokenizer):
#     '''
#     Toma una lista de palabras (devueltas por whitespace_split()) y tokeniza cada
#     palabra una por una. También almacena, para cada nuevo token, la palabra original
#     del parámetro text_words.
#     '''
#     text_tok = []
#     tok_to_word_id = []
#     for word_id, word in enumerate(text_words):
#         word_tok = tokenizer.token_to_id(word)
#         text_tok += word_tok
#         tok_to_word_id += [word_id]*len(word_tok)
#     return text_tok, tok_to_word_id

# def get_ids(tokens, tokenizer):
#     return tokenizer.convert_tokens_to_ids(tokens)

# def get_mask(tokens):
#     return np.char.not_equal(tokens, "[PAD]").astype(int)

# def get_segments(tokens):
#     seg_ids = []
#     current_seg_id = 0
#     for tok in tokens:
#         seg_ids.append(current_seg_id)
#         if tok == "[SEP]":
#             current_seg_id = 1-current_seg_id  # Convierte 1 en 0 y viceversa
#     return seg_ids


def create_input_dict(question, context, tokenizer, max_len):
    """
    Take a question and a context as strings and return a dictionary with the 3
    elements needed for the model. Also return the context_words, the
    context_tok to context_word ids correspondance and the length of
    question_tok that we will need later.
    """
    question_tok = tokenizer.encode(question)

    context_words = whitespace_split(context)  # Se puede reemplazar por un .split()
    # context_tok, context_tok_to_word_id = tokenize_context(context_words)
    # context_tok = tokenizer.encode(context).tokens
    context_tok_to_word_id = tokenizer.encode(context).ids

    input_tok = tokenizer.encode(question, context).tokens
    # input_tok = question_tok + ["[SEP]"] + context_tok + ["[SEP]"]
    input_ids = tokenizer.encode(question, context).ids
    input_mask = tokenizer.encode(question, context).attention_mask
    input_type_ids = tokenizer.encode(question, context).type_ids

    input_tok += ["[PAD]"]*(max_len-len(input_tok))
    input_ids += [0]*(max_len-len(input_ids))
    input_mask += [0]*(max_len-len(input_mask))
    input_type_ids += [0]*(max_len-len(input_type_ids))

    input_dict = {}
    # ["input_ids", "attention_mask", "token_type_ids"]
    input_dict["input_ids"] = tf.expand_dims(tf.cast(input_ids, tf.int32), 0)
    input_dict["attention_mask"] = tf.expand_dims(tf.cast(input_mask, tf.int32), 0)
    input_dict["token_type_ids"] = tf.expand_dims(tf.cast(input_type_ids, tf.int32), 0)

    return input_dict, context_words, context_tok_to_word_id, len(question_tok)


def predict(question, context, modelo=None, tokenizer=None, max_len=384, use_pipeline=False):

    if not use_pipeline:
        # Formateamos datos de entrada
        my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = create_input_dict(question,
                                                                                                      context,
                                                                                                      tokenizer,
                                                                                                      max_len)

        # Inferencia del modelo

        start_logits, end_logits = modelo(my_input_dict, training=False)

        start_logits_context = start_logits.numpy()[0, question_tok_len + 1:]
        end_logits_context = end_logits.numpy()[0, question_tok_len + 1:]

        # start_word_id = np.argmax(start_logits_context)
        # end_word_id = np.argmax(end_logits_context)

        pair_scores = np.ones((len(start_logits_context), len(end_logits_context))) * (-1E10)
        for i in range(len(start_logits_context - 1)):
            for j in range(i, len(end_logits_context)):
                pair_scores[i, j] = start_logits_context[i] + end_logits_context[j]
        pair_scores_argmax = np.argmax(pair_scores)

        start_word_id = pair_scores_argmax // len(start_logits_context)
        end_word_id = pair_scores_argmax % len(end_logits_context)

        predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id + 1])

    else:
        qa = pipeline('question-answering',
                      model=modelo,
                      tokenizer=tokenizer)

        predicted_answer = qa(context=context, question=question)

    return predicted_answer
