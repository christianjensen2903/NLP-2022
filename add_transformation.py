def add_label(row):
    row['is_answerable'] = int(len(row["annotations"]['answer_start']) != 0)
    return row

def add_tokenization(row, languages):
    language = row['language']
    tokenizer = languages[language]['tokenizer']
    row['tokenized_question'] = tokenizer(row['question_text'].lower())
    #row['tokenized_context'] = tokenizer(row['document_plaintext'].lower())
    return row

def add_transformation(row, languages):
    row = add_label(row)
    row = add_tokenization(row, languages)
    return row