# GTZAN_music_genre_classification
Music Genre Classification projet for Jedha Demodays Project with Sandra, Cyril, John and Adrien.

1. Caractéristiques Temporelles (Temps et Amplitude)
Ces caractéristiques sont analysées directement à partir de la forme d'onde du signal.
#length : Représente la durée du segment audio, généralement exprimée en secondes ou en nombre d'échantillons ;
#rms_mean / rms_var : Le Root Mean Square (valeur efficace) est une mesure de la puissance ou du volume sonore
Il décrit l'évolution temporelle de l'énergie et permet de percevoir les variations d'amplitude dans la forme d'onde.

#zero_crossing_rate_mean / zero_crossing_rate_var : Le taux de passage par zéro (ZCR) indique la fréquence à laquelle le signal change de signe (passe du positif au négatif et vice-versa).
Une valeur élevée de ZCR est souvent associée à un bruit ou à des sons non périodiques (comme les sons percussifs), tandis qu'une valeur faible indique un signal plus périodique et tonal

2. Caractéristiques Spectrales (Fréquences)
Ces caractéristiques sont obtenues en convertissant le signal du domaine temporel vers le domaine fréquentiel (généralement via une transformée de Fourier - STFT)
.
spectral_centroid_mean / spectral_centroid_var : Indique le "centre de masse" du spectre de fréquences
. Il est étroitement lié à la brillance perçue d'un son : plus le centroïde est élevé, plus le son est perçu comme "clair" ou "brillant"
.
spectral_bandwidth_mean / spectral_bandwidth_var : Représente la plage de fréquences couverte par le signal, pondérée par son spectre
.
rolloff_mean / rolloff_var : La fréquence de coupure spectrale (spectral roll-off) est la fréquence en dessous de laquelle se trouve un certain pourcentage (généralement 85 % ou 95 %) de l'énergie spectrale totale
. Cela aide à distinguer les sons avec beaucoup de hautes fréquences de ceux qui sont plus sourds.
3. Caractéristiques de Timbre et d'Harmonie
chroma_stft_mean / chroma_stft_var : Les caractéristiques de chroma projettent l'ensemble du spectre de fréquences sur 12 bacs représentant les 12 demi-tons de l'octave musicale
. C'est un outil puissant pour analyser le contenu harmonique et la progression des accords d'un morceau
.
harmony_mean / harmony_var et perceptr_mean / perceptr_var : Ces colonnes proviennent probablement d'une technique de séparation source-filtre (HPSS - Harmonic-Percussive Source Separation)
.
harmony représente la composante tonale/harmonique du son (les notes tenues)
.
perceptr (souvent pour percussive) représente la composante transitoire/percussive (les attaques brusques comme les coups de batterie)
.
mfcc1_mean à mfcc20_var : Les coefficients cepstraux sur l'échelle de Mel (Mel-Frequency Cepstral Coefficients) sont des représentations compressées de l'enveloppe spectrale
. Ils imitent la manière dont le système auditif humain perçoit les fréquences (échelle de Mel non linéaire)
. Les premiers coefficients (les ordres inférieurs) capturent les caractéristiques principales de l'instrument ou de la voix, tandis que les suivants apportent des détails spectraux plus fins
.
4. Rythme
tempo : Indique la vitesse de la pulsation de la musique, mesurée en battements par minute (BPM)
. Il est extrait en analysant la périodicité des événements de l'audio
.
