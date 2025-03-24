# Numerical Algorithms Project

## Opis projektu

Projekt zaliczeniowy z przedmiotu **Algorytmy numeryczne**. Skrypt wykonuje analizę danych, wizualizację w formie 2D i 3D oraz obliczenia numeryczne, takie jak interpolacja, aproksymacja, obliczanie całek i pochodnych.

## Funkcje skryptu

### Wizualizacja danych
- **Mapa 2D** – przedstawia dane z mapowaniem koloru dla współrzędnej **Z**.
- **Powierzchnia 3D** – wizualizuje dane jako powierzchnię z mapowaniem koloru dla współrzędnej **Z**.

### Statystyki
- Oblicza **średnią**, **medianę** i **odchylenie standardowe** z podziałem na współrzędne **Y**.

### Interpolacja dla współrzędnej **0.8**
- **Metoda Lagrange'a** – interpolacja wielomianowa.
- **Metoda trygonometryczna** – interpolacja za pomocą funkcji trygonometrycznych.

### Aproksymacja dla współrzędnej **0.8**
- **Aproksymacja liniowa** – przybliżenie danych za pomocą funkcji liniowej.
- **Aproksymacja kwadratowa** – przybliżenie danych za pomocą funkcji kwadratowej.

### Obliczenia numeryczne
- **Pole powierzchni funkcji** – oblicza pole pod zadaną powierzchnią.
- **Całki z funkcji interpolacyjnych i aproksymacyjnych:**
  - **Metoda prostokątów**
  - **Metoda trapezów**

### Pochodne i monotoniczność
- **Pochodne cząstkowe** – oblicza pochodne cząstkowe dla współrzędnej **0.3**.
- **Monotoniczność** – określa, czy funkcja jest rosnąca czy malejąca w danym zakresie.

### Wizualizacja wyników
- Prezentuje graficznie:
  - Wszystkie funkcje interpolacyjne i aproksymacyjne.
  - Pochodne cząstkowe.
  - Monotoniczność funkcji.

## Wymagania
- Python 3.x
- Biblioteki: `numpy`, `matplotlib`, `scipy`
