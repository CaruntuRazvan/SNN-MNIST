# SNN-MNIST - Spiking Neural Network pentru clasificare MNIST

- **`train.py`**: Antrenează modelul SNN pe MNIST și salvează greutățile.  
- **`app.py`**: Preia greutățile și rulează o interfață Streamlit unde poți testa predicțiile.

### Acuratețea
Codul actual obține o acuratețe de aproximativ **96.5%** pe setul de date MNIST. Acest rezultat este obținut prin implementarea celor mai recente direcții sugerate pe Google Colab, care includ:
- **Weight decay** pentru regularizare.
- **Time window** mai mare (50) pentru o performanță mai bună.

### Rulare
1. Execută `train.py` pentru antrenare.  
2. Rulează `streamlit run app.py` pentru a vedea interfața.
