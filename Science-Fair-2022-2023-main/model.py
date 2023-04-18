from transformers import ASTFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import sys
import librosa

filename = '1-100038-A-14.wav'
# y, s = librosa.load('test.wav', sr=16000 ) 
y, sr = librosa.load(filename, sr = 16000 )
sampling_rate = sr

sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

inputs = feature_extractor(y, sampling_rate=sampling_rate, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)