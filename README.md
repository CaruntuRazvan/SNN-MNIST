# SNN-MNIST - Spiking Neural Network pentru clasificare MNIST

- **`train.py`**: Antrenează modelul SNN pe MNIST și salvează greutățile.  
- **`app.py`**: Preia greutățile și rulează o interfață Streamlit unde poți testa predicțiile.

### Acuratețea
Codul actual obține o acuratețe de aproximativ **96.5%** pe setul de date MNIST. Acest rezultat este obținut prin implementarea celor mai recente direcții sugerate pe Google Colab, care includ:
- **Weight decay** pentru regularizare.
- **Time window** mai mare (50) pentru o performanță mai bună.

### Funcționalități în `app.py`
În `app.py`, poți:
1. Testa și vizualiza predicțiile modelului pentru **10 imagini random** din setul de date MNIST.
2. Testa acuratețea modelului pentru **100 de imagini random** și afișa doar imaginile greșite, împreună cu predicția modelului și eticheta corectă.

### Rulare
1. Execută `train.py` pentru antrenare.  
2. Rulează `streamlit run app.py` pentru a vedea interfața.
