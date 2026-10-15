# Decision

Un oracle qui tranche entre les cas que tu proposes, calculé entièrement dans ton navigateur.

**→ https://l0d0v1c.github.io/Decision/**

Tu décris le **contexte**, tu poses une **question**, tu listes les **cas** possibles et, pour
chacun, quand il faut le choisir (« Choisir si… »). L'oracle pèse ces critères et renvoie le cas
retenu avec un pourcentage pour chacun. Rien de ce que tu écris ne quitte ta machine : le modèle
est téléchargé une fois (335 Mo), gardé en cache, puis exécuté localement en WebAssembly sur le CPU.
La page ne dépend d'aucun site tiers : bibliothèques, moteur WebAssembly, polices et modèle sont
tous servis depuis ce dépôt.

## Le modèle

- [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual),
  révision `1720e3e`, licence Apache 2.0, par Convai Innovations. C'est un modèle de décision, pas
  un générateur de texte : il choisit parmi les cas fournis, en une seule passe.
- Converti en ONNX, poids quantifiés en int8 (une échelle par canal de sortie, activations en
  fp32), découpé en 4 fichiers de moins de 100 Mo dans `model/` (sha256 de chaque morceau dans
  `model/MANIFEST.json`, vérifiés par la page au chargement).
- Fidélité au modèle d'origine (PyTorch fp16) sur 23 cas de test : même réponse sur 22, écart de
  probabilité ≤ 0,042. Le 23ᵉ est une quasi-égalité dans l'original (0,38 contre 0,35) qui bascule.

## Limites

- Le modèle n'est pas calibré : ses pourcentages sont souvent trop sûrs d'eux.
- Il se trompe sur des évidences. Sur 7 dilemmes de risque testés, il en rate 3, dont une alarme
  de monoxyde de carbone qu'il conseille de couper (0,82). Les 4 exemples de la page sont ceux
  qu'il réussit. **Ne t'en sers pas pour décider de ta sécurité.**
- Il faut environ 1 Go de mémoire dans l'onglet ; c'est confortable sur ordinateur, incertain sur
  mobile. Une décision prend environ 1 s (un seul thread : GitHub Pages ne permet pas les en-têtes
  qu'exige le WebAssembly multithread).

## Tester

Le harnais vérifie que la tokenisation JS est identique à celle de laya en Python, puis que le
modèle ONNX sur onnxruntime-web donne les réponses de référence PyTorch :

```bash
npm install
npm test
```

## Contenu

| chemin | rôle |
|---|---|
| `index.html` | la page |
| `js/oracle.js` | validation, construction de l'entrée du modèle (identique à laya), inférence |
| `model/` | modèle ONNX int8 en morceaux, tokenizer, configuration, manifeste |
| `vendor/onnxruntime-web/` | onnxruntime-web 1.30.0 (module WASM et moteur), licence MIT |
| `vendor/tokenizers/` | @huggingface/tokenizers 0.2.0, licence Apache 2.0 |
| `vendor/fonts/` | Public Sans et Spectral (sous-ensembles latins), licence SIL OFL 1.1 |
| `tests/` | harnais golden et références PyTorch |

Les fichiers de `vendor/` sont identiques à ceux des paquets npm de `package.json`, que le
harnais de test exécute.
