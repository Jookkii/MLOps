import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

# 1. Pobranie przykładowego obrazka z datasetu MNIST (testowego)
# Zrobimy to, aby mieć pewność, że wysyłamy "prawdziwą" cyfrę
test_ds = datasets.MNIST("./data", train=False, download=True)
sample_idx = np.random.randint(len(test_ds)) # Losujemy obrazek
image, label = test_ds[sample_idx]

# Konwertujemy obrazek PIL na tablicę numpy (BentoML oczekuje tablicy)
image_array = np.array(image).astype(np.uint8)

# 2. Wyświetlenie obrazka przed wysłaniem
print(f"Wybrana cyfra z datasetu: {label}")
plt.imshow(image_array, cmap='gray')
plt.title(f"Oryginalny obrazek (Label: {label})")
plt.axis('off')
plt.show(block=False) # Nie blokujemy skryptu wyświetlaniem okna
plt.pause(2) # Dajemy 2 sekundy na podejrzenie obrazka

# 3. Przygotowanie i wysłanie requesta
print("Wysyłanie requesta do localhost:3000/classify...")

try:
    # BentoML przyjmuje JSON-a, więc zamieniamy tablicę numpy na listę Pythona
    payload = image_array.tolist()
    
    response = requests.post(
        "http://localhost:3000/classify",
        json={"input_series": image_array.tolist()}, # Nazwa klucza musi pasować do argumentu w service.py
        timeout=10
    )

    # 4. Odczytanie odpowiedzi
    if response.status_code == 200:
        prediction_logits = response.json()
        
        # Wynik z modelu to zazwyczaj logits lub softmax (10 wartości)
        # Wybieramy indeks z najwyższą wartością
        predicted_digit = np.argmax(prediction_logits)
        
        print("\n" + "="*30)
        print(f"ODPOWIEDŹ SERWERA:")
        print(f"Przewidziana cyfra: {predicted_digit}")
        print(f"Prawdziwa cyfra:    {label}")
        print("="*30)
        
        if predicted_digit == label:
            print("Sukces! Model trafił.")
        else:
            print("Model się pomylił, ale przy 96% accuracy to się zdarza.")
            
    else:
        print(f"Błąd serwera! Kod statusu: {response.status_code}")
        print(f"Treść błędu: {response.text}")

except requests.exceptions.ConnectionError:
    print("Błąd: Nie można połączyć się z serwerem. Upewnij się, że 'bentoml serve' działa!")