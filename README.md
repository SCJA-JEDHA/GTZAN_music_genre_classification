# GTZAN_music_genre_classification
Music Genre Classification project for Jedha Demodays with Sandra, Cyril, John and Adrien.

<p>Lien du dataset : https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data</p>
<p></p>Lien de l'architecture : https://excalidraw.com/#json=FsKQX4mxm74kp6hbglEDG,SABUtoTNIVD_Gp8jmtJFPw</p>

### 1. Métadonnées et durée
**length** : Le nombre total d'échantillons (samples) dans le fichier audio.
C'est une mesure brute de la durée (30 secondes à 22050 Hz donnent environ 661 500 échantillons).

### 2. Caractéristiques de "Chroma" (Contenu Harmonique)
**chroma_stft_mean / chroma_stft_var** : Ces valeurs calculent la moyenne et la variance d'un chromagramme obtenu par transformée de Fourier à court terme (STFT).
Elles représentent l'intensité des 12 demi-tons de l'octave musicale. C'est l'indicateur principal pour identifier la tonalité ou la progression d'accords.

### 3. Énergie et Volume
**rms_mean / rms_var** : La valeur efficace (Root Mean Square) est la mesure directe de la puissance ou du volume sonore.
La moyenne indique le volume global, tandis que la variance indique si le morceau a une dynamique stable ou s'il y a de grands écarts de volume.

### 4. Forme et Brillance du Spectre
**spectral_centroid_mean / spectral_centroid_var** : Indique le "centre de gravité" du spectre.
Une valeur élevée correspond à un son brillant (beaucoup de hautes fréquences), tandis qu'une valeur faible indique un son plus sourd/sombre.

**spectral_bandwidth_mean / spectral_bandwidth_var** : Mesure l'étendue de la plage de fréquences occupée par le signal.

**rolloff_mean / rolloff_var** : La fréquence de coupure spectrale est le seuil sous lequel se trouve un certain pourcentage (généralement 85 % ou 95 %) de l'énergie.
Elle permet de distinguer les sons riches en hautes fréquences des autres.

### 5. Rugosité et Bruit
**zero_crossing_rate_mean / zero_crossing_rate_var** : Le taux de passage par zéro indique la fréquence à laquelle le signal change de signe.
Des valeurs élevées sont caractéristiques des bruits blancs ou des sons percussifs.

### 6. Séparation HPSS (Harmonique-Percussive)
Ces colonnes proviennent de la technique Harmonic-Percussive Source Separation (HPSS) utilisée par Librosa :
**harmony_mean / harmony_var** : Représente la composante tonale du son (les notes tenues et mélodiques).
**perceptr_mean / perceptr_var** : Souvent noté pour percussive, il représente la composante transitoire (les attaques, les percussions, les bruits brusques).

### 7. Rythme
**tempo** : L'estimation de la vitesse de la musique en battements par minute (BPM).

### 8. Coefficients Cepstraux (MFCC)
**mfcc1_mean à mfcc20_var** : Les 20 coefficients cepstraux sur l'échelle de Mel sont les caractéristiques les plus importantes pour la classification.
Les premiers coefficients (ordres inférieurs) représentent la forme globale de l'enveloppe spectrale et l'identité de l'instrument ou de la voix.
Les coefficients supérieurs capturent des détails spectraux plus fins.

Le dataset fournit la moyenne et la variance pour chacun de ces 20 coefficients sur toute la durée du segment.

Pourquoi Mean et Var ? Pour chaque fenêtre de temps (frame), une valeur est extraite. Comme le morceau dure 30 secondes, le dataset résume ces milliers de valeurs en deux chiffres : la moyenne (l'état général) et la variance (comment cette caractéristique change au cours du temps).

