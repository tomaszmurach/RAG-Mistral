# RAG with Mistral 7B and FAISS

Projekt przedstawia prostą implementację systemu **RAG (Retrieval-Augmented Generation)** z wykorzystaniem modelu **Mistral-7B-Instruct-v0.3**, embeddingów **multilingual-e5-base** oraz wyszukiwarki wektorowej **FAISS**.  
Celem projektu jest udzielanie odpowiedzi wyłącznie na podstawie dostarczonego dokumentu źródłowego, bez korzystania z wiedzy spoza kontekstu.

## Opis projektu

W projekcie zastosowano klasyczną architekturę RAG:

**dokument -> chunking -> embeddingi -> FAISS -> retrieval -> prompt -> generacja odpowiedzi**

System działa na przykładowym dokumencie opisującym **procedurę reagowania na incydenty bezpieczeństwa IT w organizacji**.  
Po zadaniu pytania użytkownika:

1. pytanie zamieniane jest na embedding,
2. FAISS wyszukuje najbardziej podobne fragmenty dokumentu,
3. znalezione fragmenty trafiają do promptu,
4. model Mistral generuje krótką odpowiedź wyłącznie na podstawie odzyskanego kontekstu.

Dodatkowo system:
- pozwala sterować liczbą zwracanych fragmentów (`k`),
- umożliwia ustawienie progu podobieństwa (`score_threshold`),
- obsługuje parametr `temperature`,
- odmawia odpowiedzi, jeśli w dokumencie nie ma potrzebnych informacji.

## Wykorzystane technologie

- Python
- Google Colab
- Hugging Face Transformers
- Mistral-7B-Instruct-v0.3
- Sentence Transformers
- FAISS
- bitsandbytes
- accelerate
- NumPy
- PyTorch

## Model i komponenty

### Model językowy
- `mistralai/Mistral-7B-Instruct-v0.3`

### Model embeddingów
- `intfloat/multilingual-e5-base`

### Wyszukiwanie wektorowe
- `FAISS IndexFlatIP`

### Kwantyzacja
- 4-bit quantization przy użyciu `bitsandbytes`, co zmniejsza zużycie pamięci VRAM i ułatwia uruchomienie modelu w środowisku Google Colab z GPU.

## Struktura działania

### 1. Dokument źródłowy
Dokument zawiera procedurę obsługi incydentów bezpieczeństwa IT, podzieloną na sekcje, takie jak:
- cel procedury,
- definicja incydentu,
- przykłady incydentów,
- role i odpowiedzialności,
- zgłaszanie, klasyfikacja, eskalacja,
- działania naprawcze i dokumentacja.

### 2. Chunking
Dokument dzielony jest na mniejsze fragmenty tekstu.  
W tym projekcie każda linia lub para **nagłówek + treść** stanowi osobny chunk.

### 3. Embeddingi
Każdy chunk zostaje zamieniony na wektor numeryczny przy pomocy modelu `multilingual-e5-base`.

### 4. Indeks FAISS
Embeddingi trafiają do indeksu FAISS, który umożliwia szybkie wyszukiwanie najbardziej podobnych fragmentów dla zadanego pytania.

### 5. Retrieval
Dla pytania użytkownika tworzony jest embedding, a następnie FAISS zwraca najbardziej trafne fragmenty dokumentu.

### 6. Generacja odpowiedzi
Znaleziony kontekst trafia do promptu modelu Mistral, który generuje krótką odpowiedź zgodną z treścią dokumentu.

## Najważniejsze funkcje

### `chunk_text(text)`
Dzieli dokument na chunki, łącząc nagłówki z odpowiadającą im treścią.

### `retrieve_context(query, k=3, score_threshold=None, return_meta=False)`
Wyszukuje najbardziej podobne fragmenty dokumentu na podstawie embeddingu pytania.

Parametry:
- `k` – liczba zwracanych fragmentów,
- `score_threshold` – minimalny próg podobieństwa,
- `return_meta` – zwraca dodatkowe informacje diagnostyczne.

### `ask_bot(question, k=3, score_threshold=None, temperature=0.01, show_meta=False)`
Główna funkcja systemu RAG.  
Łączy retrieval i generację odpowiedzi w jednym przepływie:

**pytanie -> wyszukiwanie kontekstu -> budowa promptu -> odpowiedź**

## Cechy rozwiązania

- odpowiedzi generowane są wyłącznie na podstawie dokumentu,
- brak inferencji spoza kontekstu,
- możliwość testowania wpływu parametrów `k`, `score_threshold` i `temperature`,
- obsługa sytuacji, w której dokument nie zawiera odpowiedzi,
- możliwość wyświetlania metadanych FAISS do analizy działania systemu.

## Wymagania

Projekt był uruchamiany w **Google Colab** z dostępem do **GPU**.

Instalacja bibliotek:

```bash
pip install -U bitsandbytes sentence-transformers faiss-cpu transformers accelerate
```

Widzę też, że wcześniej rozwaliło się formatowanie przez zagnieżdżone code blocki. Poniżej masz wersję bez zewnętrznego bloku, już normalnie do skopiowania linia po linii:

## Uruchomienie

1. Otwórz notebook w Google Colab.  
2. Włącz środowisko GPU.  
3. Zainstaluj wymagane biblioteki.  
4. Załaduj model Mistral oraz model embeddingów.  
5. Przygotuj dokument źródłowy.  
6. Podziel dokument na chunki:

```python
chunks = chunk_text(document)
```
7. Wygeneruj embeddingi chunków i zbuduj indeks FAISS.
8. Zadawaj pytania przy pomocy funkcji:

```python
ask_bot("Jak należy zgłosić incydent według dokumentu?")
```

## Przykładowe testy

W projekcie przetestowano kilka scenariuszy:

### 1. Poprawne działanie RAG

Pytania dotyczące informacji zawartych w dokumencie, np.:

- Jaki jest cel procedury reagowania na incydenty?
- Jak należy zgłosić incydent według dokumentu?

### 2. Odmowa odpowiedzi spoza dokumentu

Przykład:

- W jakiej temperaturze wrze woda?

Oczekiwany wynik:

`Brak informacji w dokumencie.`

### 3. Wpływ `score_threshold`

Pokazanie, jak próg podobieństwa wpływa na liczbę zachowanych fragmentów kontekstu.

### 4. Wpływ `k`

Porównanie odpowiedzi dla różnych wartości liczby pobieranych chunków.

### 5. Wpływ `temperature`

Sprawdzenie, jak temperatura wpływa na styl generowanej odpowiedzi.

## Ograniczenia

- Projekt działa na jednym, ręcznie zdefiniowanym dokumencie tekstowym.
- Chunking jest prosty i oparty na liniach tekstu.
- System nie korzysta z dodatkowej bazy danych dokumentów.
- Jakość odpowiedzi zależy od jakości chunkingu i retrievalu.
- Do uruchomienia modelu Mistral zalecane jest środowisko z GPU.

## Możliwe rozszerzenia

- obsługa wielu dokumentów,
- zapis dokumentów do osobnej bazy wiedzy,
- bardziej zaawansowany chunking,
- dodanie interfejsu użytkownika,
- porównanie kilku modeli językowych,
- ewaluacja jakości odpowiedzi na większym zbiorze pytań.

## Autor

Projekt przygotowany jako implementacja systemu **RAG** z użyciem modelu **Mistral**, embeddingów semantycznych oraz indeksowania wektorowego **FAISS**.
