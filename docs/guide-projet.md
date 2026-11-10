# Guide du projet — comprendre, terminer, soutenir

Ce document répond à trois questions : **qu'est-ce qui est fait**, **comment ça marche**, et
**quelles étapes il te reste** pour terminer le projet et pouvoir l'expliquer sans notes.

---

## 0. Où en est le projet

| Élément | État |
|---|---|
| Code de l'application | ✅ refactoré, testé (45 tests) |
| Prétraitement + détection de visage | ✅ `deepfake/preprocess.py`, `deepfake/face.py` |
| CNN déterministe (bug corrigé) | ✅ `deepfake/models.py` + test de non-régression |
| Notebook d'entraînement | ✅ réécrit, exécuté de bout en bout sur un mini-dataset |
| Métriques reproductibles (frame + vidéo) | ✅ `scripts/evaluate_on_split.py` |
| **Entraînement sur Celeb-DF v2 complet** | ❌ à relancer (dataset non fourni) |
| **Chiffres définitifs pour le rapport** | ❌ dépendent de l'entraînement |
| Workflow CI | ⚠️ écrit, pas encore poussé (token sans scope `workflow`) |

Les seuls chiffres mesurés aujourd'hui sont ceux des 20 échantillons : **ResNet 17/20,
CNN 15/20, consensus 18/20**. Ce n'est pas un résultat scientifique, c'est un contrôle de bon
fonctionnement — assume-le comme tel.

---

## 1. Le problème en dix lignes

Un **deepfake** est un visage synthétique produit par un réseau (ici par *FaceSwap* : le visage
d'une personne est remplacé par celui d'une autre). Le projet ne « détecte » pas une manipulation
en comprenant l'image : il apprend une **classification binaire** — un crop de visage est-il
réel (`0`) ou manipulé (`1`) ?

Celeb-DF v2 fournit des milliers de vidéos réelles et synthétiques. On en extrait des visages
128 × 128. Pourquoi 128 ? Parce que les artefacts de *blending* (le raccord entre le visage
collé et le contour) restent visibles à cette résolution, tout en divisant par 4 le coût de calcul
par rapport à 256. Deux modèles sont comparés : un CNN entraîné de zéro (baseline honnête) et un
ResNet-18 pré-entraîné ImageNet (transfer learning). Le ResNet gagne — c'est le résultat attendu,
puisque les filtres ImageNet encodent déjà des textures et des contours utiles.

---

## 2. Le flux complet

```
Celeb-DF v2 (vidéos)
   │  extraction des visages (externe au dépôt)
   ▼
visages 128×128 ──► split_par_video.csv (rel_path, label, video, split)
   │                   └─ split PAR VIDÉO : aucune vidéo dans deux splits
   ▼
notebook ResNet-18
   ├─ phase 1 : backbone gelé, on entraîne la tête (3 époques)
   ├─ phase 2 : tout dégelé, LR différenciés (16 époques max)
   ├─ checkpoint = meilleure AUC de validation
   ├─ seuil = Youden sur validation (jamais sur le test)
   └─ évaluation test : frame + vidéo, IC95, McNemar, calibration, Grad-CAM
   ▼
artefacts (resnet18.pt, history, figures, predictions.csv, model_card.json)
   ▼
application Flask
   ├─ YuNet : détecte le visage, recadre avec marge
   ├─ ResNet-18 (PyTorch) + CNN Simple (Keras, déterministe)
   └─ verdict + confiance + consensus, ou refus explicite si pas de visage
```

### Fichier → rôle → ce que tu dois savoir expliquer

| Fichier | Rôle | À savoir dire |
|---|---|---|
| `deepfake/preprocess.py` | EXIF, alpha sur fond blanc, resize bilinéaire 128×128, normalisation ImageNet | « le même prétraitement à l'entraînement et à l'inférence, sinon les scores sont faux » |
| `deepfake/face.py` | YuNet détecte le visage, crop carré avec 25 % de marge | « le modèle a appris sur des crops ; sans recadrage il ne voit jamais le visage » |
| `deepfake/models.py` | chargement strict, suppression des couches aléatoires, test de déterminisme | « le .keras livré augmente les images à l'inférence : la même image changeait de verdict » |
| `deepfake/metrics.py` | AUC, seuils, agrégation vidéo, bootstrap, McNemar, ECE | « une métrique par image surestime, car les frames d'une vidéo sont corrélées » |
| `model_card.json` | contrat entraînement ↔ déploiement | « la taille, la normalisation, le seuil et le sens des labels sont écrits une fois » |
| `app.py` | routes, pipeline, codes HTTP | « l'app refuse une image sans visage plutôt que d'inventer un verdict » |
| `deepfake-resnet18-etudiant2.ipynb` | entraînement + évaluation | les 2 phases, le choix du checkpoint, le seuil sur validation |
| `scripts/evaluate_on_split.py` | réévaluation reproductible | « seuil sur validation, métriques frame et vidéo, IC95, McNemar » |
| `scripts/export_inference_model.py` | exporte un CNN propre | « vérifie que deux appels identiques donnent la même sortie » |
| `tests/` | 45 tests | « le bug du CNN a un test de non-régression » |

---

## 3. Le notebook, cellule par cellule

| # | Cellule | Ce qu'elle fait | Concept |
|---|---|---|---|
| 1 | Config | chemins (env ou Kaggle), seed, device, hyperparamètres | reproductibilité |
| 2 | Vérifications | colonnes, labels, splits, **fuite par vidéo**, fichiers présents | data leakage |
| 3 | Données | transforms, `FaceDataset`, 3 loaders | augmentation, `drop_last` |
| 4 | Modèle | ResNet-18 ImageNet, backbone gelé, nouvelle tête | transfer learning |
| 5 | Boucle | `run_epoch`, `fit`, checkpoint sur AUC val, early stopping | AUC, patience |
| 6 | Entraînement | phase tête (LR 1e-3) puis fine-tuning (1e-5 / 1e-4) | LR différenciés, AdamW |
| 7 | Prédiction | probabilités sur validation et test | sigmoid |
| 8 | Seuil | Youden sur la validation, puis appliqué au test | seuil de décision |
| 9 | Métriques | frame + vidéo, IC95 bootstrap, rapport détaillé | agrégation vidéo |
| 10 | Figures | courbes, ROC/PR, matrice de confusion, calibration (ECE) | calibration |
| 11 | CNN | chargement, retrait des couches aléatoires, test de déterminisme, McNemar | comparaison loyale |
| 12 | Erreurs | `predictions.csv` + 12 erreurs les plus confiantes | analyse d'erreurs |
| 13 | Grad-CAM | carte d'activation sur `layer4` | explicabilité |
| 14 | Robustesse | recompression JPEG q=90/70/50 | généralisation |
| 15 | Sauvegarde | artefacts + `model_card.json` | traçabilité |

---

## 4. Les notions à savoir expliquer (fiche de révision)

**Transfer learning** — réutiliser un réseau entraîné sur ImageNet (1,2 M images) et remplacer sa
dernière couche par une tête binaire. *Pourquoi ici* : avec ~80 k images, apprendre des filtres
depuis zéro est lent et moins bon. *Piège* : ce n'est pas magique — le modèle reste dépendant du
domaine (visages, compression).

**Backbone gelé / fine-tuning** — phase 1 : on gele les poids ImageNet et on n'entraîne que la
tête (rapide, évite de détruire les filtres). Phase 2 : on dégèle tout avec un LR 10× plus petit
sur le backbone. *Question type* : « pourquoi deux phases ? » → pour ne pas casser les
représentations pré-entraînées par un gradient trop fort au début.

**BCEWithLogitsLoss** — perte d'entropie binaire calculée sur le logit (avant sigmoid). *Pourquoi*
: numériquement stable (pas de `log(0)`). Le modèle sort un logit ; on applique `sigmoid` seulement
pour l'affichage.

**Sigmoid et seuil** — sigmoid transforme le logit en probabilité. Le verdict dépend d'un **seuil**.
0,5 est arbitraire : il suppose que les deux erreurs coûtent pareil. Ici le seuil est choisi sur la
validation (indice de Youden = sensibilité + spécificité − 1).

**AUC** — probabilité qu'une image FAKE tirée au hasard ait un score plus élevé qu'une image REAL
tirée au hasard. Indépendante du seuil : c'est la mesure de la **qualité du classement**. Pour
l'usage réel, il faut aussi le seuil et les métriques associées.

**Précision / rappel / F1** — précision FAKE = parmi les images déclarées FAKE, combien le sont
vraiment (faux positifs = accuser à tort). Rappel FAKE = parmi les vraies falsifications, combien
sont détectées (faux négatifs = laisser passer). F1 = moyenne harmonique. *Le choix dépend de
l'enjeu* : modération → privilégier le rappel ; expertise judiciaire → privilégier la précision.

**Data leakage** — information du test qui fuit dans l'entraînement. Deux formes ici : (1) même
vidéo dans deux splits → contrôlé par code ; (2) même **personne** dans deux splits → non contrôlé,
à citer comme limite. C'est la première question d'un jury sur un projet deepfake.

**Frame vs vidéo** — 10 155 images de test proviennent de quelques dizaines de vidéos ; les frames
d'une vidéo sont presque identiques. Compter par image revient à compter 300 fois la même
personne : l'accuracy monte artificiellement. Le niveau vidéo (moyenne des probabilités par vidéo)
est le chiffre à présenter.

**Calibration / ECE** — un modèle peut être juste mais mal calibré : dire « 90 % de confiance »
alors qu'il n'a raison que 70 % du temps. L'ECE mesure l'écart moyen entre confiance annoncée et
fréquence observée. *Conséquence* : afficher « confiance » dans une interface est trompeur si
l'ECE est élevé.

**Bootstrap / McNemar** — le bootstrap rééchantillonne les images pour donner un intervalle de
confiance sur l'AUC. McNemar teste si deux modèles diffèrent significativement sur les **mêmes**
images (paires discordantes). *Question type* : « ton ResNet est-il vraiment meilleur ? » → avec
un IC95 et un test, tu peux répondre au lieu de comparer deux nombres.

**Grad-CAM** — carte de chaleur des zones qui poussent la décision vers FAKE, obtenue en pondérant
les activations de la dernière couche convolutive par le gradient. *Pourquoi* : vérifier que le
modèle regarde le visage et pas le fond ou la compression.

**Augmentation** — transformations aléatoires à l'entraînement uniquement (flip, rotation ±7°,
variations de couleur). *Piège* : sur des crops de visage, une rotation forte crée des coins noirs
que le réseau peut apprendre comme artefact.

---

## 5. Les étapes concrètes pour terminer

### Étape 1 — Vérifier l'environnement (10 min)

```bash
cd ~/Desktop/DeepLearningProjetCode
source venv/bin/activate
python -c "import torch, tensorflow, flask, cv2; print('ok')"
pytest -q                      # 45 tests doivent passer
python app.py                  # http://localhost:5001
```

Attendu : page d'accueil, « 2 / 2 MODELS LOADED ». Clique sur **Run demo sample** : l'échantillon
et sa vérité terrain s'affichent — compare-les, c'est ta première preuve visuelle.

### Étape 2 — Récupérer le dataset et vérifier le split (30 min)

Celeb-DF v2 se demande sur le dépôt officiel (conditions d'utilisation). L'arborescence attendue
par le CSV est :

```
Celeb_V2/
├── Train/real/*.jpg     Train/fake/*.jpg
├── Val/real/*.jpg       Val/fake/*.jpg
└── Test/real/*.jpg      Test/fake/*.jpg
```

Le CSV `result_etudiant1/split_par_video.csv` contient des chemins relatifs de ce type
(`Train/real/00000_face_0.jpg`). Vérifie que les fichiers existent :

```bash
python - <<'EOF'
import pandas as pd
from pathlib import Path
df = pd.read_csv('result_etudiant1/split_par_video.csv')
root = Path('Celeb_V2')
missing = [r for r in df.rel_path if not (root / r).is_file()]
print(f"{len(df):,} lignes | manquantes : {len(missing):,}")
print(missing[:3])
EOF
```

Si le nombre de manquantes est élevé, ton extraction a une autre arborescence : dis-le-moi, j'écris
un script qui reconstruit le CSV depuis ton dossier (labels et identifiants vidéo inclus).

### Étape 3 — Entraîner (Kaggle, GPU, 1 à 3 h)

1. Sur Kaggle : *New Notebook* → upload `deepfake-resnet18-etudiant2.ipynb`.
2. *Add Data* → ton dataset Celeb-DF v2 + un dataset contenant `split_par_video.csv`
   (et `cnn_simple.keras` si tu veux la comparaison).
3. *Settings* → Accelerator **GPU T4**, Internet **On** (les poids ImageNet sont téléchargés).
4. *Run all*.

Le notebook détecte tout seul `/kaggle/input`. Ce qu'il faut regarder pendant l'exécution :
- la phase *head* doit déjà atteindre ~90 % d'exactitude en 3 époques ;
- l'AUC de validation doit monter puis plafonner ; l'early stopping s'arrête après 4 époques sans
  progrès ;
- à la fin : le seuil retenu (souvent entre 0,1 et 0,6), l'AUC frame vs vidéo, l'ECE.

### Étape 4 — Rapatrier et déployer le résultat (20 min)

```bash
# depuis /kaggle/working/result_etudiant2/reproduction/
cp model_card.json            ~/Desktop/DeepLearningProjetCode/model_card.json
cp resnet18_best.pt           ~/Desktop/DeepLearningProjetCode/resnet18.pt
cp cnn_simple_inference.keras ~/Desktop/DeepLearningProjetCode/cnn_simple.keras
cd ~/Desktop/DeepLearningProjetCode && python app.py
```

L'application lit le seuil dans `model_card.json` : elle utilise donc le seuil validé, pas 0,5.
Vérifie avec `/health` que `threshold` correspond.

### Étape 5 — Réévaluer proprement (optionnel mais convaincant, 30 min)

```bash
python scripts/evaluate_on_split.py \
  --dataset Celeb_V2 \
  --splits result_etudiant1/split_par_video.csv
```

Produit `result_etudiant2/reevaluation/metrics.json` : seuil, métriques frame et vidéo, IC95,
McNemar. C'est le chiffre à mettre dans le rapport.

### Étape 6 — Mettre à jour le README avec tes chiffres (15 min)

Remplace le tableau « Statut des métriques » par ton tableau de résultats et enlève l'avertissement.
Tant que ce n'est pas fait, le README dit honnêtement que les chiffres sont historiques.

### Étape 7 — Pousser le workflow CI (5 min)

```bash
git add .github/workflows/ci.yml
git commit -m "ci: workflow GitHub Actions"
git push
```

Si GitHub refuse (`workflow` scope) : crée un token avec la case `workflow`, ou dépose le fichier
via l'interface web (*Add file → Upload files*).

### Étape 8 — Préparer la démo et la soutenance (2 h)

Scénario de démonstration en 3 minutes :

1. `python app.py`, montrer `/health` (modèles, device, seuil).
2. **Run demo sample** : verdict affiché à côté de la vérité terrain.
3. Upload d'une photo personnelle : montrer le recadrage du visage et le verdict.
4. Upload d'une image **sans visage** : montrer le refus (`422`) — c'est la preuve que le décalage
   train/production a été compris.
5. Ouvrir `resnet18_gradcam.png` et `resnet18_roc_pr.png` : où le modèle regarde, et pourquoi l'AUC
   ne suffit pas.
6. Terminer sur `metrics.json` : frame vs vidéo, IC95, McNemar.

---

## 6. Comment lire tes résultats (et quoi en dire)

| Situation | Interprétation | Ce que tu dis |
|---|---|---|
| AUC frame 0,98 et vidéo 0,93 | écart normal (frames corrélées) | « le niveau vidéo est le chiffre honnête, l'écart mesure la redondance des frames » |
| AUC frame 0,99 et vidéo 0,70 | le modèle exploite des indices de **vidéo**, pas de visage | suspect : vérifier Grad-CAM et le split par identité |
| ECE > 0,1 | probabilités mal calibrées | « la confiance affichée est indicative, pas une probabilité calibrée » |
| Chute forte en JPEG q=50 | modèle fragile à la compression | argument pour de l'augmentation JPEG à l'entraînement |
| McNemar p > 0,05 | écart non significatif | « on ne peut pas conclure que le ResNet est meilleur » — c'est une réponse mature |

---

## 7. Plan de révision sur 5 jours (2 h/jour)

| Jour | Objectif | Tâches |
|---|---|---|
| 1 | Vue d'ensemble | lire `README.md`, section 1-3 de ce guide, lancer l'app et la démo |
| 2 | Données et protocole | cellule 2 du notebook, comprendre le split par vidéo, refaire le contrôle de fuite à la main sur 5 vidéos |
| 3 | Modèles | cellule 4-6, relire `deepfake/models.py`, expliquer à voix haute les 2 phases et le choix du checkpoint |
| 4 | Métriques | cellules 8-10, savoir refaire à la main : matrice de confusion → précision/rappel/F1 ; expliquer AUC vs accuracy |
| 5 | Explicabilité et oral | Grad-CAM + robustesse + les 15 questions du §8, démo chronométrée |

---

## 8. Questions probables du jury

1. **Pourquoi 128 × 128 ?** Compromis coût/artefacts ; à 64 les traces de blending disparaissent, à 256 le coût quadruple pour un gain faible.
2. **Pourquoi un split par vidéo ?** Sinon la même scène apparaît en train et en test : le score est gonflé (leakage).
3. **Votre modèle a-t-il vu le test ?** Non : test évalué une seule fois, seuil et checkpoint choisis sur la validation. C'est vérifiable dans le code.
4. **Pourquoi l'AUC et pas l'accuracy ?** L'AUC est indépendante du seuil et robuste au déséquilibre ; l'accuracy dépend d'un seuil arbitraire.
5. **Pourquoi le niveau vidéo ?** Les frames sont corrélées ; une décision par vidéo est l'unité réelle d'usage.
6. **Pourquoi deux phases d'entraînement ?** Éviter de détruire les poids ImageNet avec un gradient trop fort.
7. **Pourquoi le CNN est-il moins bon ?** Il part de zéro avec 80 k images, là où le ResNet part de 1,2 M images : c'est la démonstration de l'intérêt du transfer learning.
8. **Votre CNN est-il fiable ?** Il **était** non déterministe (couches d'augmentation dans le graphe). Corrigé : couches retirées + test automatique. Avant : 6 images sur 20 changeaient de verdict.
9. **Et si on vous donne une photo sans visage ?** Refus explicite (422). Avant, l'app produisait un verdict sur une image compressée : c'était le principal défaut.
10. **Comment savez-vous que le modèle regarde le visage ?** Grad-CAM : la carte se concentre sur les yeux/la bouche et le contour du visage.
11. **Que se passe-t-il sur un autre générateur (diffusion) ?** Non garanti : Celeb-DF ne contient que FaceSwap. Le test JPEG donne une idée de la robustesse, pas une preuve.
12. **Le modèle est-il biaisé ?** Celeb-DF est majoritairement composé de célébrités ; peu de diversité d'âge/peau. À citer comme limite.
13. **Différence précision/rappel, et lequel privilégier ?** Voir §4 ; le choix dépend du coût des faux positifs vs faux négatifs.
14. **Vos probabilités sont-elles fiables ?** Mesurées par l'ECE ; si > 0,1 elles sont indicatives.
15. **Que feriez-vous avec plus de temps ?** Volet audio (le nom du dépôt parle d'audio-visuel), entraînement multi-générateurs, calibration (temperature scaling), déploiement ONNX.

---

## 9. Ce que je peux coder ensuite

Dis-moi simplement le numéro :

1. **Volet audio** — le nom du dépôt (`...audio-visuels...`) suggère une partie audio absente du code. Si le sujet l'impose, c'est le manque le plus visible : extraction MFCC/mel-spectrogramme + petit CNN + fusion des deux scores.
2. **`scripts/build_split.py`** — reconstruire `split_par_video.csv` depuis ton arborescence réelle, avec contrôle de fuite par vidéo.
3. **Script de démonstration** — analyse des 20 échantillons, tableau + figure, pour la soutenance.
4. **Export ONNX + Dockerfile** — déploiement reproductible, utile si on te demande « et en production ? ».
5. **Rapport PDF / DOCX** — mise en forme du projet (je peux générer un document propre avec les figures).
6. **Extension du notebook** — temperature scaling (calibration), TTA, ou split par identité.

---

## 10. Checklist finale

- [ ] `pytest -q` → 45 tests passent
- [ ] `python app.py` → « 2 / 2 MODELS LOADED »
- [ ] démo : vérité terrain vs verdict cohérents sur plusieurs tirages
- [ ] image sans visage → refus explicite
- [ ] notebook relancé sur le dataset complet, artefacts récupérés
- [ ] `model_card.json` copié à la racine, seuil vérifié via `/health`
- [ ] `scripts/evaluate_on_split.py` exécuté, `metrics.json` obtenu
- [ ] README mis à jour avec **tes** chiffres (frame + vidéo + IC95)
- [ ] workflow CI poussé
- [ ] 15 questions du §8 répétées à voix haute
- [ ] démo chronométrée en 3 minutes
