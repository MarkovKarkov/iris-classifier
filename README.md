# Iris Classifier

Questo progetto implementa un semplice classificatore di fiori Iris usando Python e scikit-learn.   
L'obiettivo è prevedere la specie del fiore (setosa, versicolor, virginica) sulla base di misure come lunghezza e larghezza di sepalo e petalo.

## 📂 Struttura del progetto

- `main.py` — Script principale per caricare i dati, addestrare il modello e visualizzare i risultati.
- `requirements.txt` — Librerie Python necessarie.
- `.gitignore` — File e cartelle ignorati da Git.
- `.env.example` (opzionale) — Esempio di file per variabili d’ambiente (se usi API o chiavi segrete).

## 🔧 Requisiti

Assicurati di avere Python 3.7+ installato.  
Installa le dipendenze in un ambiente virtuale:

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate su Windows
pip install -r requirements.txt
