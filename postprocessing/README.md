# DISCLAIMER SU CARTELLA POSTPROCESSING

Praticamente l'unico algoritmo che funzioan decentemente è HDBSCAN_uniqueFrag_high_att_frags_analysis

Usa HDBSCAN che è una versione migliroata di DBSCAN basato su clustering gerarchico.

Inoltre uso solamente ECFP4 come fingerprint.

Trovi tutti i dati in cartelle e file contrassegnati da uf (unique fingerprint) iniziali.

Uso configurazione con 

    n_bits_fingerprint=1024          
    min_cluster_size=3 

