#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import string
import random
import fasttext
import fasttext.util
import resiliparse
import resiliparse.parse.encoding
import resiliparse.extract.html2text
import nltk
from nltk.tokenize import word_tokenize
from typing import Any
import hashlib
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from typing import List, Union

nltk.download('punkt_tab')



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    
    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    html_string = html_bytes.decode(encoding)
    plain_text = resiliparse.extract.html2text.extract_plain_text(html_string)

    return plain_text

    


    raise NotImplementedError


def run_identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("/Users/tiffanyloe/Desktop/ECE 491B/s2025-assignment2-data/lid.176.bin")
    
    cleaned_text = text.replace("\n", " ").strip()

    chunks = [cleaned_text[i:i+2000] for i in range(0, len(cleaned_text), 2000)]

    predictions = []
    confidences = []

    for chunk in chunks:
        prediction = model.predict(chunk)[0][0].replace("__label__", "")
        confidence = model.predict(chunk)[1][0]

        if prediction == 'eng':
            prediction = 'en'
        elif prediction == 'cmn':
            prediction = 'zh'

        predictions.append(prediction)
        confidences.append(confidence)

    most_common_lang = max(set(predictions), key=predictions.count)
    avg_confidence = sum(confidences) / len(confidences)

    return most_common_lang, avg_confidence
    
    raise NotImplementedError


def run_mask_emails(text: str) -> tuple[str, int]:
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    matches = re.findall(email_pattern, text)
    
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)
    
    return masked_text, len(matches)
    raise NotImplementedError


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_pattern = r'''
        (?:(?:\+1\s*)?    
        (?:\(?\d{3}\)?    
        [\s\-\.]?         
        \d{3}             
        [\s\-\.]?         
        \d{4}))           
    '''

    matches = re.findall(phone_pattern, text, re.VERBOSE)
    
    masked_text = re.sub(phone_pattern, "|||PHONE_NUMBER|||", text, flags=re.VERBOSE)
    
    return masked_text, len(matches)
    
    raise NotImplementedError


def run_mask_ips(text: str) -> tuple[str, int]:
    ip_pattern = r'''
        \b                   
        (?:                  
            (?:25[0-5]|      
             2[0-4][0-9]|    
             1[0-9]{2}|      
             [1-9][0-9]?|    
             0)              
            \.               
        ){3}                 
        (?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]?|0)  
        \b                   
    '''

    matches = re.findall(ip_pattern, text, re.VERBOSE)
    
    masked_text = re.sub(ip_pattern, "|||IP_ADDRESS|||", text, flags=re.VERBOSE)
    
    return masked_text, len(matches)
    raise NotImplementedError


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")
    
    cleaned_text = text.strip().replace("\n", " ")

    prediction = model.predict(cleaned_text)

    predicted_label = prediction[0][0] 
    confidence = prediction[1][0]       

    if predicted_label == '__label__nsfw':
        label = "nsfw"
    elif predicted_label == '__label__non-nsfw':
        label = "non-nsfw"  
    else:
        label = "unknown"

    return label, confidence
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("jigsaw_fasttext_bigrams_hatespeech_final.bin")

    cleaned_text = text.strip().replace("\n", " ")

    prediction = model.predict(cleaned_text)

    predicted_label = prediction[0][0] 
    confidence = prediction[1][0]       

    if predicted_label == '__label__toxic':
        label = "toxic"
    elif predicted_label == '__label__non-toxic':
        label = "non-toxic"  
    else:
        label = "unknown"

    return label, confidence

    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("quality_classifier.bin")
    
    cleaned_text = text.replace("\n", " ").strip()  
    prediction = model.predict(cleaned_text)
    predicted_label = prediction[0][0]  
    confidence = prediction[1][0]

    if predicted_label == "__label__high":
        label = "wiki"
    elif predicted_label == "__label__low":
        label = "cc"
    else:
        label = "unknown"
    
    return label, confidence

    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    words = word_tokenize(text)
    
    if not (50 <= len(words) <= 100000):
        return False

    word_lengths = [len(word) for word in words]
    mean_word_length = sum(word_lengths) / len(words)
    if not (3 <= mean_word_length <= 10):
        return False

    lines = text.split('\n')
    ellipsis_count = sum(1 for line in lines if line.strip().endswith("..."))
    if ellipsis_count / len(lines) > 0.3:
        return False

    words_with_alpha = sum(1 for word in words if any(char.isalpha() for char in word))
    if words_with_alpha / len(words) < 0.8:
        return False

    return True
    raise NotImplementedError


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    line_counts = defaultdict(int)

    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                hashed_line = hashlib.md5(line.strip().encode()).hexdigest()
                line_counts[hashed_line] += 1

    os.makedirs(output_directory, exist_ok=True)

    for file_path in input_files:
        input_filename = os.path.basename(file_path)
        output_path = os.path.join(output_directory, input_filename)

        with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                hashed_line = hashlib.md5(line.strip().encode()).hexdigest()
                if line_counts[hashed_line] == 1:
                    outfile.write(line)

    return None
    raise NotImplementedError


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_ngrams(text: str, n: int) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_hashes)

    minhashes = {}
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = normalize_text(file.read())
            doc_ngrams = get_ngrams(text, ngrams)

            minhash = MinHash(num_perm=num_hashes)
            for ngram in doc_ngrams:
                minhash.update(ngram.encode('utf-8'))

            minhashes[file_path] = minhash
            lsh.insert(file_path, minhash)

    clusters = defaultdict(set)
    for file_path, minhash in minhashes.items():
        candidates = lsh.query(minhash)
        for candidate in candidates:
            clusters[file_path].add(candidate)

    seen_documents = set()
    os.makedirs(output_directory, exist_ok=True)

    for file_path, duplicates in clusters.items():
        if file_path not in seen_documents:
            selected_document = random.choice(list(duplicates))
            seen_documents.update(duplicates)  

            output_path = os.path.join(output_directory, os.path.basename(selected_document))
            with open(selected_document, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(infile.read())

    return None

    raise NotImplementedError
