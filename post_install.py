import subprocess

# Función para descargar el modelo de Spacy
def download_spacy_model(model_name):
    subprocess.run(["python", "-m", "spacy", "download", model_name])

if __name__ == "__main__":
    download_spacy_model("es_core_news_md")
