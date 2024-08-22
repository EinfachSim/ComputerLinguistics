# Zwischen den Zeilen lesen: Können domänenspezifische Embeddings Hinweise auf politische Überzeugungen und Vorurteile geben?
Dieses Projekt beinhaltet die verwendeten Skripte zur Erstellung und Analyse der Korpora aus dem Report.
Kurze Erläuterung zu den Dateien:
- preprocess.ipynb: In diesem Notebook findet das Preprocessing statt. Es führt die im Report genannten Schritte durch und speichert die fertigen Daten in .txt Dateien.
- fasttext.ipynb: Hierin wurden die Modelle trainiert.
- experimentation_neighborhoods.ipynb: Dieses Notebook beinhaltet einige Experimente mit den trainierten Modellen und t-SNE Plots.
- exclusive_word_uses.ipynb: In diesem Notebook werden die Worte ermittelt, die in nur einem der beiden Vokabulare verwendet werden.
- alignment_azarbonyad_et_al.ipynb implementiert die Aligment Methode aus Azarbonyad et al. (2017) in der Variante, wie sie im Report genannt wird.