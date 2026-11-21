# Zadanie domowe 1: różniczkowanie automatyczne za pomocą klasy `Dual`

Projekt `uv` z klasą `Dual` (różniczkowanie automatyczne w trybie w przód) z ćwiczenia
`ex04` z lab02, uzupełnioną o potęgę o wykładniku rzeczywistym, `c ** x`, funkcje
elementarne `tan`, `log`, `sqrt`, ścisłą funkcję `lift` oraz pochodne cząstkowe
(`gradient`). Dzielenie, potęgę o wykładniku całkowitym i `log` z zadania 7 `ex04` można
przenieść.

## Pliki

- `dual.py`: szkielet do uzupełnienia. Każda funkcja i metoda oznaczona `# TODO` zgłasza
  `NotImplementedError`; jej dokumentacja podaje regułę do zaimplementowania. Części
  wykonane na zajęciach w `ex04` można przenieść.
- `test_dual.py`: komplet testów. Punkty, w których sprawdzane są pochodne, są losowane
  z ziarna `HW01_SEED` (domyślnie 0), więc testy można powtórzyć w innych punktach.
  Testów nie należy zmieniać.
- `pyproject.toml`: definicja projektu (`pytest` jako zależność deweloperska).

## Polecenia

```bash
uv sync                 # środowisko projektu z pytest
uv run pytest -x        # testy w kolejności R1–R7, zatrzymanie na pierwszym niepowodzeniu
uv run pytest           # wszystkie testy; wynik końcowy: same "passed"
HW01_SEED=7 uv run pytest   # te same testy w innych punktach
```

Sposób pracy jak w TDD: uruchomić testy (wszystkie zgłaszają `NotImplementedError`),
uzupełnić pierwszą funkcję, uruchomić ponownie, i tak do końca. Wyrocznią dla pochodnych
funkcji złożonych (R6, R7) jest schemat różnicowy centralny z krokiem $10^{-5}$ i tolerancją
względną $10^{-6}$ (bezwzględną $10^{-8}$ przy pochodnej bliskiej zeru).

## Oddanie

Archiwum zip z katalogiem projektu bez `.venv` i `.pytest_cache` (wystarczą `dual.py`,
`test_dual.py`, `pyproject.toml`), przez Moodle, przed spotkaniem 3.
