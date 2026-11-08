> **État au moment de l'audit.** Ce rapport est conservé comme trace de l'audit initial.
> Les bloquants P0-1 à P0-4 et la majorité des points P1 ont été corrigés depuis
> (voir le README, `deepfake/`, `scripts/` et l'historique git). Les constats restent
> exacts pour la version d'origine du code.

# Revue du projet Deepfake — ce qu'il faut améliorer

Revue faite en inspectant `app.py`, `deepfake-resnet18-etudiant2.ipynb`, les artefacts de
`result_etudiant1/` et `result_etudiant2/`, le split `split_par_video.csv`, et en **exécutant
les deux modèles livrés** sur les 20 échantillons du dépôt.

Verdict court : **le protocole d'entraînement du notebook est sain** (split par vidéo vérifié,
test jamais touché, sélection du checkpoint sur la validation). Les problèmes sérieux sont
ailleurs : **l'artefact CNN livré est non déterministe**, **l'application ne fait pas de
détection de visage**, et **les chiffres du README ne sont pas reproductibles par le notebook
actuel**. Ce sont les trois points à traiter avant toute soutenance.

---

## 0. Ce qui est déjà bien (à garder)

- Split stratifié **par vidéo** avec un contrôle programmatique de fuite (`groupby("video")["split"].nunique()`), plus vérification que le test n'a jamais servi à sélectionner le checkpoint.
- Vérifications de chemins/colonnes/labels en début de notebook : erreurs explicites plutôt que silencieuses.
- Lecture des images à la demande (`Dataset.__getitem__`) au lieu de tout charger en RAM.
- Prétraitement d'inférence soigné : EXIF, canal alpha sur fond blanc, conversion RGB.
- Équilibre des classes correct (pas besoin de pondération) : 50,2 % de FAKE sur 101 032 images — train 80 798 / val 10 078 / test 10 155.
- Interface : `prefers-reduced-motion`, `prefers-contrast`, focus visibles, latence affichée, endpoint REST, mode démo.
- README structuré et honnête sur la provenance des chiffres archivés.

---

## 1. Bloquants (P0) — à corriger avant de présenter le projet

### P0-1. Le modèle CNN livré est **non déterministe** : la même image donne un verdict différent

`cnn_simple.keras` contient les couches d'augmentation dans son graphe :

```
InputLayer → Rescaling → RandomFlip → RandomRotation → RandomBrightness
           → RandomContrast → Conv2D → ... → Dropout → Dense
```

Elles **s'appliquent aussi à l'inférence**. Vérifié en appelant 12 fois `cnn_model.predict()`
sur les mêmes 20 images, sans aucune modification du code :

| Constat mesuré | Valeur |
|---|---|
| Images dont le **verdict change** entre deux appels identiques | **6 / 20** |
| Écart-type moyen de la probabilité (12 appels) | 0,105 |
| Exemple extrême — `08_REAL_00105_face_5985.jpg` | probabilité de FAKE entre **0,04 et 0,98** |
| Exemple — `19_FAKE_id41_id40_0002_face_34111.jpg` | entre 0,04 et 0,85 |

Conséquences :

1. La démo live n'est pas reproductible : le bandeau « consensus / divergence » dépend du tirage aléatoire du request.
2. Les métriques CNN archivées (89,77 %) ont été calculées **sur des versions augmentées** des images de test, pas sur le modèle tel qu'il tourne. Comparer un modèle bruité à un ResNet-18 déterministe n'est pas une comparaison loyale.
3. En reconstruisant le graphe **sans** les couches aléatoires et sans Dropout (poids identiques), le CNN obtient **12/20** sur les échantillons livrés, au lieu de 19/20 avec le bruit. La performance réelle du CNN est donc très probablement inférieure au chiffre annoncé — c'est exactement ce qu'il faut mesurer avant de rédiger le rapport.

Correctif (dans le notebook **et** dans `app.py`) :

```python
import keras
from keras import layers as L

RANDOM_LAYERS = (L.RandomFlip, L.RandomRotation, L.RandomZoom,
                 L.RandomBrightness, L.RandomContrast, L.Dropout)

def strip_random_layers(model):
    """Graphe d'inférence déterministe, mêmes poids, sans augmentation ni dropout."""
    x = keras.Input(model.input_shape[1:])
    y = x
    for layer in model.layers[1:]:
        if isinstance(layer, RANDOM_LAYERS):
            continue
        y = layer(y)
    return keras.Model(x, y, name="cnn_simple_inference")

inference = strip_random_layers(cnn_model)
inference.save("cnn_simple_inference.keras")

# Garde-fou obligatoire : deux appels identiques → mêmes sorties
b = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files[:8]])
assert np.allclose(inference.predict(b, verbose=0), inference.predict(b, verbose=0)), \
    "Le modèle d'inférence n'est pas déterministe"
```

Mieux encore : **sortir l'augmentation du modèle**. Deux architectures propres :

- `model` = architecture pure (Rescaling → Conv → … → Dense), entraîné avec un `tf.data` qui applique l'augmentation en amont ;
- à la fin, on sauvegarde `model` tel quel. Plus rien à nettoyer, plus de risque de régression.

Et dans `app.py`, faire un **self-test au démarrage** :

```python
probe = np.zeros((2, 128, 128, 3), dtype=np.float32)
if not np.allclose(cnn_model.predict(probe, verbose=0), cnn_model.predict(probe, verbose=0)):
    raise RuntimeError("CNN non déterministe : artefact invalide, réexportez-le.")
```

### P0-2. Aucune détection ni recadrage de visage dans l'application

Le modèle a été entraîné sur des **crops de visage 128 × 128** (c'est ce que contient
`echantillons_20_photos/`). L'application, elle, accepte n'importe quelle photo ou vidéo et se
contente de la **compresser en 128 × 128**. Sur un portrait en pied, un plan large ou une photo
de groupe, le réseau ne voit jamais le visage : la prédiction est du bruit, avec une confiance
affichée qui reste élevée. C'est le décalage entraînement / production le plus visible pour un
correcteur.

Correctifs, par ordre de coût :

1. **Minimum** : détecter les visages (MediaPipe Face Detection, OpenCV DNN, ou RetinaFace) et ne pas répondre si aucun visage n'est trouvé. Recadrer avec une marge (~20 % autour de la bbox), puis redimensionner en 128 × 128.
2. **Multi-visages** : inférer sur chaque visage, agréger (moyenne ou max des probabilités), et afficher les bbox avec le verdict par visage. C'est aussi beaucoup plus convaincant en démo.
3. **Vidéo** : l'actuel `extract_frame_from_video` prend la frame du milieu. Mieux : échantillonner N frames (ex. 1 s d'intervalle), agréger les scores, et afficher la timeline. Sinon, retirer l'option vidéo et l'assumer dans le README (aujourd'hui elle est annoncée mais donne un résultat non fiable).
4. Si le périmètre du projet interdit la détection : **le dire explicitement** dans l'UI (« Image attendue : crop de visage 128 × 128 ») et refuser les autres formats, au lieu de produire un verdict silencieusement faux.

### P0-3. Les chiffres du README ne sont pas reproductibles par le notebook livré

Preuves concrètes :

- `result_etudiant2/resnet18_history.csv` a les colonnes `epoch, loss, accuracy, precision, recall, auc, val_*` ; le notebook actuel écrit `phase, epoch, train_loss, train_accuracy, train_auc, val_*`. **Ce CSV ne peut pas être produit par ce notebook.**
- Le README annonce `result_etudiant1/cnn_simple.keras` : le fichier est **absent** du dossier (et supprimé dans l'arbre de travail).
- Le README annonce `resnet18_erreurs.png` : le notebook ne contient **aucun** code qui génère cette figure.
- Le README annonce 93,79 % d'accuracy et 97,5 % de recall FAKE ; sur les 20 échantillons livrés, le ResNet-18 tourne à **15/20** (5 faux classés réels, 0 fausse alerte) et le CNN déterministe à **12/20**. n = 20, donc ce n'est pas une preuve, mais cela montre que **le chemin de déploiement n'a jamais été évalué**.

À faire : soit relancer le notebook sur le dataset complet et **remplacer** les chiffres + régénérer tous les artefacts, soit marquer explicitement dans le README « chiffres d'une version antérieure du code, non reproductibles » (c'est déjà partiellement dit, mais le tableau de résultats reste présenté comme les résultats du projet). Un tableau de résultats non reproductible est le premier point qu'un jury attaque.

### P0-4. Évaluation au niveau *frame* uniquement

Les 10 155 images de test proviennent d'un petit nombre de vidéos ; les frames d'une même vidéo sont quasi identiques. Une accuracy par frame **surestime** largement la performance réelle, et c'est la façon dont on reconnaît un projet deepfake sérieux : **agréger par vidéo**.

```python
pred = pd.DataFrame({"video": test_df.video.values, "y": y_true, "p": y_prob_res})
video = pred.groupby("video").agg(y=("y", "first"), p=("p", "mean"))

print("Frame-level AUC :", roc_auc_score(y_true, y_prob_res).round(4))
print("Video-level AUC :", roc_auc_score(video.y, video.p).round(4))
print("Video-level acc :", accuracy_score(video.y, (video.p >= 0.5).astype(int)).round(4))
print(f"Vidéos de test : {len(video)}")
```

Ajouter aussi : **seuil choisi sur la validation** (Youden ou max F1) plutôt que 0,5 en dur, **intervalles de confiance bootstrap**, **test de McNemar** entre les deux modèles (à n = 10 000, 93,79 % vs 89,77 % est significatif ou non, il faut le chiffrer), et les courbes **ROC / précision-rappel** + un diagramme de fiabilité (calibration). Aujourd'hui, l'affirmation « ResNet-18 > CNN » repose sur un seul split et aucune incertitude.

---

## 2. Problèmes importants (P1)

### 2.1 Notebook — bugs et dette

1. **Boucle morte dans la cellule de comparaison** (lignes 344-347 du `.ipynb`) :

   ```python
   cnn_probs = []
   for batch_x, _ in test_loader:
       indices = len(cnn_probs)          # ne fait rien, parcourt 10 155 images
   ```

   Elle parcourt tout le test set sans effet et laisse une variable `indices` inutilisée : signe d'un refactor inachevé. À supprimer.

2. **Mémoire** : la même cellule charge les 10 155 images dans une liste puis `np.stack` (~500 Mo en uint8 + ~2 Go en float32 côté TF), avec ResNet déjà en mémoire. Passer par un `tf.data.Dataset` par lots, ou réutiliser le `DataLoader` avec un dataset « pixels bruts ».

3. **`except ImportError` seulement** : un `.keras` corrompu ou une version Keras incompatible fait planter la cellule. Utiliser `except Exception as exc: print("Comparaison ignorée :", exc)`.

4. **Code dupliqué** : la phase fine-tuning réimplémente entièrement la boucle de `fit_phase` au lieu de l'appeler (avec en plus le scheduler absent de `fit_phase`). Factoriser `fit_phase(phase, epochs, optimizer, scheduler=None)`.

5. **`drop_last=True` manquant** sur le train loader : avec `BatchNorm`, un dernier lot de taille 1 fait échouer l'entraînement. Ici 80 798 % 64 = 30, donc ça passe — mais c'est fragile.

6. **Performance** : `NUM_WORKERS = 0` et pas d'AMP. Décodage JPEG de 80 k images à chaque époque sur un seul thread = goulot majeur. `NUM_WORKERS = 4`, `persistent_workers=True`, `prefetch_factor=2`, et `torch.amp.autocast("cuda")` + `GradScaler` sur GPU.

7. **Reproductibilité incomplète** : seed global posé, mais pas `worker_init_fn`, pas `generator` pour `shuffle`, pas `torch.use_deterministic_algorithms`. Ajouter au minimum un `torch.Generator().manual_seed(SEED)` passé à `DataLoader`.

8. **Métadonnées Kaggle incohérentes** : `"accelerator": "none"` et `"isInternetEnabled": false`, alors que `ResNet18_Weights.DEFAULT` doit **télécharger** les poids ImageNet et que 3 + 16 époques sur CPU sont infaisables. Le notebook tel qu'il est configuré ne peut pas tourner. Activer GPU + Internet, ou attacher les poids en dataset.

9. **Pas de sauvegarde des probabilités** : impossible de recalculer les métriques, d'analyser les erreurs ou de changer de seuil sans réentraîner. Ajouter `predictions.csv` (chemin, vidéo, label, probabilité).

10. **Sélection du checkpoint sur `val_loss` uniquement** : avec l'overfitting visible dans l'historique archivé (train loss 0,007 vs val loss qui remonte après l'époque 13), sélectionner sur `val_auc` ou `val_f1` est plus aligné avec ce qu'on veut mesurer.

11. **Pas d'explicabilité** : aucune Grad-CAM. Pour un projet de détection de deepfake c'est presque indispensable — cela montre *où* le modèle regarde (artefacts de blending, contours du visage) et permet de détecter un modèle qui triche (fond, compression).

12. **Pas d'analyse d'erreurs** : le README promet `resnet18_erreurs.png`, le code ne le génère plus. Grille des top-k erreurs les plus confiantes = la figure la plus parlante du rapport.

13. **Pas de test de robustesse** : recompresser le test en JPEG q=70, redimensionner, flouter, et remesurer. Un détecteur de deepfake qui s'effondre après une recompression, c'est le résultat le plus intéressant à présenter (et cela justifie l'augmentation JPEG/compression à l'entraînement).

14. **Augmentation discutable pour ce domaine** : `RandomRotation(15°)` avec remplissage noir crée des coins noirs que le modèle peut apprendre comme artefact. Réduire à ±5-7°, ajouter `RandomResizedCrop(scale=(0.85, 1.0))`, du flou léger et de la compression JPEG. Le flip horizontal est acceptable ici.

15. **Fine-tuning** : le backbone est dégelé d'un coup à `lr=1e-4`. Mieux : freeze des BatchNorm (`model.eval()` sur les BN) ou LR discriminatifs (backbone 1e-5, tête 1e-3). Et `ReduceLROnPlateau` sur `val_auc`.

16. **Overfitting non traité** : le meilleur modèle de l'historique archivé a une val loss qui remonte dès l'époque 14 avec train loss 0,007. Ajouter du label smoothing, `weight_decay`, et éventuellement un early stopping sur l'AUC.

### 2.2 Application Flask

1. **Chargement silencieux** : si un `.keras`/`.pt` ne se charge pas, `print()` et l'UI affiche « — » sans explication, et `consensus` renvoie `True` (« — » == « — »). Exposer un état clair : bandeau d'erreur, `/health` renvoyant `{"resnet": true, "cnn": false, "errors": [...]}`, et `consensus: null` si un modèle manque.

2. **`torch.load` sans `weights_only=True`** (ligne 85) : risque d'exécution de pickle arbitraire. Passer `weights_only=True`.

3. **`strict=False`** (ligne 89) : un checkpoint incompatible est accepté en silence. Utiliser `strict=True`, ou comparer les clés et **refuser** de servir un modèle partiellement chargé.

4. **Import TensorFlow obligatoire en haut de fichier** : si TF n'est pas installé, l'application entière ne démarre pas, même pour servir uniquement le ResNet. Rendre l'import optionnel, comme OpenCV.

5. **Repli MPS → CPU incomplet** : dans le `except`, le modèle est déplacé sur CPU mais `PYTORCH_DEVICE` reste `mps`, donc chaque requête suivante refait l'aller-retour et lève de nouveau l'exception. Mettre à jour une variable globale une seule fois.

6. **Serveur** : `app.run(host="0.0.0.0")` par défaut, sans `PORT`/`HOST` documentés au-delà du port ; `FLASK_SECRET_KEY` avec une valeur par défaut codée en dur. Rendre `HOST` configurable et refuser de démarrer en production sans clé fournie.

7. **Erreurs HTTP** : `/api/predict` renvoie 500 avec le message d'exception (fuite d'information) pour une image invalide — ce devrait être 400. `MAX_CONTENT_LENGTH` renvoie la page 413 par défaut : ajouter un handler.

8. **Décompression bomb** : aucune limite sur `Image.MAX_IMAGE_PIXELS` ; un fichier de 32 Mo peut décoder en image gigantesque. Ajouter un garde-fou (`Image.MAX_IMAGE_PIXELS = 4000 * 4000`, `DecompressionBombError` → 400).

9. **Verrou global** : `model_inference_lock` sérialise toutes les requêtes. Acceptable en démo, mais à documenter — et si vous voulez du parallélisme, un verrou par modèle suffit (TF `predict` est thread-safe pour l'inférence, PyTorch en lecture aussi).

10. **Fichier unique de 927 lignes** dont ~580 lignes de HTML/CSS/JS dans une chaîne Python. Sortir `templates/index.html`, `static/app.css`, `static/app.js` : c'est la première chose qu'un correcteur remarque, et cela permet de tester le backend sans le gabarit.

11. **Mode démo** : une image unie grise n'a aucun sens. Choisir aléatoirement un fichier de `echantillons_20_photos/` et afficher la vérité terrain à côté de la prédiction — c'est une démo autrement plus convaincante (et cela révèle immédiatement les erreurs du modèle).

12. **Duplication de la logique** : `execute_inference` renvoie des dicts dont les clés sont consommées par le gabarit ET par l'API, mais `index()` et `demo()` dupliquent le `render_template_string`. Factoriser un `render_result(...)`.

13. **Pas de constantes partagées avec l'entraînement** : `128`, les moyennes/écarts ImageNet, l'orientation des labels sont codés en dur à deux endroits (notebook et app). Créer `model_card.json` (`img_size`, `mean`, `std`, `threshold`, `class_1_is_fake`, `architecture`, `metrics`) écrit par le notebook et lu par l'app : c'est ce qui évite définitivement le décalage train/serve.

### 2.3 Reproductibilité / dépendances

- `requirements.txt` **non épinglé** alors que le projet dépend de versions précises (les couches `v2`, `weights_only`, Keras 3). Épingler, et séparer `requirements-dev.txt`.
- `opencv-python-headless` est listé comme obligatoire alors que le code le traite comme optionnel, et le README l'annote « optional ». Choisir.
- Le README annonce Python 3.10–3.14 ; la compatibilité TensorFlow dépend de la version (le venv local est en 3.14 avec TF 2.22rc). Annoncer une version testée, pas une plage optimiste.
- Aucun `LICENSE` alors que le README a une section License.
- Aucun test, aucune CI, aucun `Dockerfile`.

---

## 3. Hygiène du dépôt (P2) — rapide mais visible

- **50 Mo de `.git`** dont `resnet18.pt` (44,8 Mo) et `cnn_simple.keras` (4,8 Mo) versionnés. Utiliser Git LFS, ou mieux : publier les poids en *release* et les télécharger par script. Au minimum, documenter les empreintes SHA-256.
- `result_etudiant2/.gradio/certificate.pem` est **versionné** : certificat/clé privée d'une session Gradio. À supprimer du dépôt et à ajouter au `.gitignore` (`**/.gradio/`, `*.pem`).
- `result_etudiant1/split_par_video.csv` (5,7 Mo, 101 032 lignes) est versionné : compression en `.csv.gz` ou chargement à la demande.
- Fichiers supprimés non commités (`deepfake-resnet18-etudiant1.ipynb`, `result_etudiant1/cnn_simple.keras`, `cnn_simple_history.csv`, `result_etudiant2/resnet18.pt`) et fichiers non suivis (capture d'écran `.png`, `deepfake_architecture.png`). L'arbre de travail est sale : commiter la suppression, `.gitignore` les captures, ou intégrer `deepfake_architecture.png` au README avec une légende.
- `result_etudiant*/__results___files/` et `.DS_Store` : résidus d'export de notebook, à exclure.
- `__pycache__/` à la racine : exclu par `.gitignore` mais présent sur le disque.
- README : corriger la liste des fichiers (voir P0-3), et déplacer `deepfake_architecture.png` dans une section « Architecture » si le fichier est conservé.

---

## 4. Structure cible recommandée

```
projet/
├── src/deepfake/
│   ├── preprocess.py      # resize + EXIF + alpha + normalisation  ← partagé notebook/app
│   ├── models.py          # build_resnet18(), build_cnn(), strip_random_layers()
│   ├── data.py            # FaceDataset, split checks, agrégation par vidéo
│   └── metrics.py         # frame-level, video-level, bootstrap, McNemar, ECE
├── app/
│   ├── __init__.py        # create_app()
│   ├── routes.py
│   ├── inference.py       # chargement + self-test de déterminisme
│   ├── templates/index.html
│   └── static/{app.css,app.js}
├── notebooks/01_resnet18.ipynb
├── tests/test_preprocess.py, test_api.py, test_artifacts.py
├── model_card.json
├── requirements.txt / requirements-dev.txt
├── Dockerfile
└── README.md
```

Le point clé : **`preprocess.py` et `models.py` sont importés par le notebook et par l'app**.
Tant que le prétraitement est dupliqué, le décalage train/serve reviendra.

---

## 5. Plan d'action ordonné

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | Réexporter le CNN sans couches aléatoires + self-test de déterminisme | 1 h | **Critique** |
| 2 | Recalculer les métriques CNN déterministes et corriger le README | 2 h | **Critique** |
| 3 | Ajouter la détection/crop de visage (ou verrouiller l'entrée « crop 128×128 ») | 3 h | **Critique** |
| 4 | Métriques vidéo + seuil sur val + ROC/PR + McNemar/bootstrap | 3 h | **Critique** |
| 5 | Supprimer la boucle morte, factoriser `fit_phase`, `drop_last`, workers + AMP | 2 h | Élevé |
| 6 | Sauver `predictions.csv`, `model_card.json`, grille d'erreurs, Grad-CAM | 3 h | Élevé |
| 7 | Test de robustesse JPEG / resize | 2 h | Élevé |
| 8 | Découper `app.py` en package, imports optionnels, codes HTTP, `/health` | 4 h | Moyen |
| 9 | 4-5 tests pytest + CI (ruff + pytest) | 2 h | Moyen |
| 10 | Nettoyage git (LFS, `.gradio`, captures, `.DS_Store`, fichiers supprimés) | 1 h | Moyen |

Les points 1 à 4 changent la valeur du projet ; les suivants le rendent présentable et maintenable.

---

## 6. Note pour la soutenance

Trois phrases à préparer, parce qu'elles seront demandées :

1. **« Pourquoi 93,79 % et pas mieux ? »** → parce que c'est du niveau *frame* ; donner le chiffre *vidéo*, plus bas et plus honnête.
2. **« Comment savez-vous que le modèle regarde le visage ? »** → Grad-CAM (à ajouter), sinon la réponse est « on ne sait pas ».
3. **« Que se passe-t-il avec une photo quelconque ? »** → actuellement, une prédiction non fiable ; après le point 3, un refus explicite ou un crop correct.
