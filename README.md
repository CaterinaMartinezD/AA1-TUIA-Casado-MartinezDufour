# Aprendizaje Automático 1
## TRABAJO PRÁCTICO N° 2: Modelo predictivo de lluvia en Australia.

**Integrantes:**
- Casado, Gustavo.
- Martinez Dufour, Caterina.

## Instrucciones de instalación
### Clonar el repositorio
Para clonar el repositorio se debe ejecuta el siguiente comando en la terminal o consola:

```bash
git clone --depth 1 https://github.com/CaterinaMartinezD/AA1-TUIA-Casado-MartinezDufour.git
```
### Crear un Entorno Virtual (venv)
Crea un entorno virtual llamado "venv" y lo activa:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux
conda activate venv # Si tenes Miniconda
```

### Librerías
Requiere python 3.11 por el uso de PyCaret.
Una vez activado el entorno virtual, es necesario instalar los siguientes módulos:

```bash
pip install -r requirements.txt
```

### Instrucciones para construir la imagen de docker y ejecutar el container

Dentro de la carpeta del proyecto ejecutar:
```bash
docker build -t inference-python-test ./docker
docker run -it --rm --name inference-python-test -v ./files:/files  inference-python-test
```