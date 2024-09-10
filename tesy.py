import os
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU instead.")


# Function to query GPT-2 for ideology
def query_gpt2(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    input_text = f"Analyze the ideology (Pro-Israeli, Pro-Palestine, Neutral) expressed in this sentence: '{text}'"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=10,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id,
        temperature=1.0,
        top_p=0.85,
        do_sample=True
    )

    output_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    print(f"Generated Outputs: {output_texts}")

    possible_ideologies = ["Pro-Israeli", "Pro-Palestine", "Neutral", "Neutral, leans Pro-Israeli", "Neutral, leans Pro-Palestine"]

    for output_text in output_texts:
        for ideology in possible_ideologies:
            if ideology in output_text:
                return ideology

    return output_texts[0]


# Function to read the corpus and labels from a text file with labels in parentheses
def load_corpus_and_labels(file_path):
    corpus = []
    labels = []
    label_map = {
        "Pro-Israeli": 0,
        "Pro-Palestine": 1
    }

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"^(.*?):\s+(.*)$", line.strip())
            if match:
                label, text = match.groups()
                if label in label_map:
                    labels.append(label_map[label])  # Convert labels to integers
                    corpus.append(text)

    return corpus, labels


# Preprocess the text for topic modeling
def preprocess_for_topic_modeling(corpus):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    term_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, term_matrix


# Apply LDA for topic extraction
def apply_topic_modeling(corpus, num_topics=5):
    vectorizer, term_matrix = preprocess_for_topic_modeling(corpus)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(term_matrix)
    return lda, vectorizer


# Display the top words per topic
def display_topics(model, feature_names, num_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
        print(f"Topic {topic_idx}: {topics[-1]}")
    return topics


# Create a custom Dataset for fine-tuning
class IdeologyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True,
                                return_tensors="pt")
        inputs = {key: val.squeeze(0).to(device) for key, val in inputs.items()}  # Move inputs to GPU
        inputs['labels'] = torch.tensor(label, dtype=torch.long).to(device)  # Ensure label is a long integer tensor
        return inputs


# Function to identify the ideology from a given text using your fine-tuned BERT
def identify_ideology_bert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    model.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    alignment_prob = torch.nn.functional.softmax(logits, dim=1).max().item()

    ideologies = ["Pro-Israeli", "Pro-Palestine"]
    return logits, predicted_class_id, ideologies[predicted_class_id], alignment_prob


# Function to compare with baseline BERT model and validate using GPT-2
def compare_with_baseline(text, model, tokenizer, baseline_model):
    # Predict with the fine-tuned BERT model
    _, _, my_model_ideology, _ = identify_ideology_bert(text, model, tokenizer)

    # Predict with GPT-2
    gpt2_ideology = query_gpt2(text)

    print(f"My Model Ideology: {my_model_ideology}")
    print(f"GPT-2 Ideology: {gpt2_ideology}")

    if my_model_ideology != gpt2_ideology:
        print("Discrepancy found between GPT-2 and my model. Using baseline prediction.")

        # Predict with the baseline model
        _, _, baseline_ideology, _ = identify_ideology_bert(text, baseline_model, tokenizer)
        return baseline_ideology

    return my_model_ideology


def learn_word_correlations(text, model, tokenizer, fixed_size=256):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    model.to(device)

    # Get model outputs, including the attention weights
    outputs = model(**inputs, output_attentions=True)
    logits = outputs.logits

    # Access the attention weights from all layers
    attention_weights = outputs.attentions  # List of attention matrices from all layers

    # Average across layers and heads, and flatten
    avg_attention = np.mean([att.mean(dim=1).cpu().detach().numpy() for att in attention_weights], axis=0).flatten()

    # Fix the size of the correlation vector by padding or truncating
    if len(avg_attention) > fixed_size:
        word_correlations = avg_attention[:fixed_size]
    else:
        word_correlations = np.pad(avg_attention, (0, max(0, fixed_size - len(avg_attention))), 'constant')

    return logits.argmax().item(), word_correlations


def fine_tune_gpt2(train_dataset):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    return model, tokenizer


# Function to evaluate the model performance
def evaluate_model(eval_texts, eval_labels, model, tokenizer):
    model.eval()
    predictions = []
    true_labels = []
    for text, label in zip(eval_texts, eval_labels):
        _, predicted_class_id, _, _ = identify_ideology_bert(text, model, tokenizer)
        predictions.append(predicted_class_id)
        true_labels.append(label)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")

    conf_matrix = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Pro-Israeli", "Pro-Palestine"])
    disp.plot(cmap='Blues')
    plt.show()


def is_novel(position, archive, threshold=0.5):
    position_array = np.array(position, dtype=float)  # Ensure position is a flat, fixed-size array
    for archived_array in archive:
        archived_array = np.array(archived_array, dtype=float)
        if np.linalg.norm(position_array - archived_array) < threshold:
            return False
    return True


# Dynamic Learning with Exploration and GPT-2 Feedback
def dynamic_learning_with_gpt2(corpus, labels, model, tokenizer, baseline_model, train_dataset, eval_dataset, epochs=5, exploration_threshold=0.50, correction_threshold=10):
    archive = defaultdict(list)

    total_samples = 0
    match_count = 0  # Count matches with baseline
    gpt_mismatch_count = 0  # Count mismatches between GPT-2 predictions and themselves
    pro_israeli_streak = 0  # Track consecutive pro-Israeli predictions

    training_args = TrainingArguments(
        output_dir='./results_full',
        num_train_epochs=1,  # More epochs for deeper training
        per_device_train_batch_size=4,  # Lower batch size if needed for more precise training
        gradient_accumulation_steps=32,  # Larger accumulation steps
        fp16=True,  # Mixed precision for speed
        save_steps=1000,  # Save less frequently
        logging_steps=500,  # Less frequent logging
        evaluation_strategy="epoch",  # Same evaluation strategy
        learning_rate=5e-5  # More conservative learning rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    for epoch in range(epochs):
        print(f"Dynamic Learning Epoch {epoch + 1}/{epochs}")

        for text, label in zip(corpus, labels):
            total_samples += 1

            # Predict ideology using word correlations
            predicted_class_id, word_correlations = learn_word_correlations(text, model, tokenizer)
            ideology_label = ["Pro-Israeli", "Pro-Palestine"][predicted_class_id]

            # Get baseline model prediction
            baseline_prediction = compare_with_baseline(text, model, tokenizer, baseline_model)

            # Check if the model's prediction matches the baseline
            if baseline_prediction == ideology_label:
                match_count += 1
            else:
                print(f"Discrepancy between baseline and model for: {text}")

            # Get GPT-2 predictions
            gpt_prediction_1 = query_gpt2(text)
            gpt_prediction_2 = query_gpt2(text)  # Get a second prediction for consistency check

            # Check if GPT-2 is consistent with itself
            if gpt_prediction_1 != gpt_prediction_2:
                gpt_mismatch_count += 1
                print(f"GPT-2 has inconsistent predictions for: {text}")
                print(f"GPT-2 Prediction 1: {gpt_prediction_1}, GPT-2 Prediction 2: {gpt_prediction_2}")

            # Manual intervention if too many Pro-Israeli predictions in a row
            if gpt_prediction_1 == "Pro-Israeli":
                pro_israeli_streak += 1
            else:
                pro_israeli_streak = 0  # Reset streak on different prediction

            # Inject manual intervention if Pro-Israeli streak exceeds threshold
            if pro_israeli_streak >= correction_threshold:
                print(f"Manual intervention triggered after {pro_israeli_streak} Pro-Israeli predictions.")
                correction_text = random.choice(["The humanitarian situation in Gaza must be addressed.", "Palestinian statehood is a fundamental right."])
                correction_label = 1  # Pro-Palestine
                corpus.append(correction_text)
                labels.append(correction_label)
                pro_israeli_streak = 0  # Reset streak after correction

                # Fine-tune GPT-2 with the corrected example
                print(f"Fine-tuning GPT-2 with manual correction: {correction_text}")
                fine_tune_gpt2(IdeologyDataset([correction_text], [correction_label], tokenizer))

            # If the final ideology differs from the model's current prediction, update the model
            final_ideology = baseline_prediction  # Can be weighted by feedback from baseline and GPT-2
            if final_ideology != ideology_label:
                print(f"Updating model based on feedback for: {text}")
                corpus.append(text)
                labels.append(label)

                # Split data and retrain model
                train_texts, eval_texts, train_labels, eval_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)
                train_dataset = IdeologyDataset(train_texts, train_labels, tokenizer)
                trainer.train_dataset = train_dataset
                trainer.train()

    # Calculate and print accuracy based on how many times the predictions matched the baseline
    accuracy = (match_count / total_samples) * 100
    print(f"Model accuracy based on matching baseline: {accuracy:.2f}%")

    # Print mismatch rate for GPT-2
    gpt_mismatch_rate = (gpt_mismatch_count / total_samples) * 100
    print(f"GPT-2 mismatch rate: {gpt_mismatch_rate:.2f}%")
    print("Dynamic learning with GPT-2 feedback and manual intervention completed.")


if __name__ == "__main__":
    # Load the corpus and labels from the text file
    corpus, labels = load_corpus_and_labels('E:/Ideological embeddings/Ideological embeddings/shuffled_transcript_variations.txt')


    print(f"Loaded {len(corpus)} samples.")

    # Load pre-trained BERT model and tokenizer from Hugging Face
    print("Loading BERT model and tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Load the baseline model
    baseline_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Split data into training and evaluation sets
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)

    train_dataset = IdeologyDataset(train_texts, train_labels, tokenizer)
    eval_dataset = IdeologyDataset(eval_texts, eval_labels, tokenizer)

    # Fine-tune the BERT model
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Reduced number of epochs
        per_device_train_batch_size=2,  # Adjusted batch size
        per_device_eval_batch_size=2,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        fp16=True,
        save_steps=500,  # Save model every 500 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        dataloader_num_workers=4,  # Increase number of workers for data loading
    )

    checkpoint = "./results/checkpoint-20652"  # Adjust the path to your latest checkpoint

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Resuming training from checkpoint...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    print("Model and tokenizer saved.")

    # Apply topic modeling and display the topics
    lda_model, vectorizer = apply_topic_modeling(corpus, num_topics=5)
    display_topics(lda_model, vectorizer.get_feature_names_out(), num_top_words=10)

    # Execute dynamic learning with GPT-2 feedback
    dynamic_learning_with_gpt2(corpus, labels, model, tokenizer, baseline_model, train_dataset, eval_dataset)
