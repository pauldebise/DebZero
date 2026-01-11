Le projet à été initialement développé en python 3.11.

Le projet contient un exécutable déjà compilé du moteur, pour le recompiler:
-Créer un virtual environnement (.venv) et y installer les libs nécessaires (requirements.txt)
-Dans le dossier du projet et avec le venv activé, exécuter la commande suivante :

python -m nuitka --onefile --lto=yes --follow-imports --include-data-file=model2/model2_onnx_fp32.onnx=model2/model2_onnx_fp32.onnx uci.py

