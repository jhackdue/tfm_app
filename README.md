# TFM APP

En este repositorio encontraremos una aplicación web en la que tenemos que pasar dos textos. Uno de ellos será una
pregunta y el otro, el contexto para esa pregunta. Como resultado, se nos devolverá un fragmento del contexto que
se interpreta como respuesta a la pregunta.

Como Backend de la aplicación estará un modelo de `Question Answering` entrenado localmente. El cual se ha apificado
usando la librería `FastApi` y también se ha creado un `FrontEnd` usando la librería `Streamlit`.

## Configuración
El modelo usado se entrenó con un modelo de Language Modelling llamado Beto
* https://github.com/dccuchile/beto

Para el `transfer learning` se usó el dataset `SQuAD v2.0 Español`.
* https://rajpurkar.github.io/SQuAD-explorer/

Se puede obtener más detalle sobre cómo es el proceso de entrenamiento del modelo leyendo la memoria del 
`Trabajo de final de Máster`.

## Requisitos

Podremos ejecutar este proyecto de dos formas. Desde un notebook donde se detalla los pasos a seguir para o entrenar 
un modelo desde cero o si se quiere, utilizando ya un modelo por defecto, hacer una inferencia.

Para ello necesitaremos realizar lo siguiente:

### Git

Realizar primero una clonación del proyecto a nivel local. Debemos instalar git y luego se ejecuta:
```
git clone githttps://github.com/jhackdue/tfm_app.git tfm_app
``` 

### Python

Para poder ejecutar el notebook, es necesario tener instalado todos los paquetes. 

Para este proyecto se ha utilizado `Python 3.8` y se ha creado un entorno local utilizando el comando:
````
python -m venv tf_gpu
````

Una vez creado, activaremos el entorno
* Windows
````
tf_gpu\Scripts\activate
````
* Linux
````
source tf_gpu\bin\activate
````

Una vez activado, nos posicionamos dentro de la carpeta generada por git tfm_app y ejecutaremos:
````
pip install -r requirements.txt
````
### Docker

Si no queremos hacer uso del notebook y no queremos instalar paquetes de `python`, podemos simplemente instalarnos 
`Docker`y `Docker compose`. Para ello seguiremos la siguiente url: https://docs.docker.com/desktop/windows/install/

Tanto el BackEnd como el FrontEnd están subidos en el `DockerHub` como imágenes públicas y, por tanto, podemos hacer uso
de ellas con el docker compose.

Hay que tener en cuenta las especificaciones de la máquina cuando instalemos Docker.

## Despliegue

Para desplegar el docker compose:
1. Abriremos una terminal
2. Nos posicionaremos en el directorio `tfm_app` o donde esté el fichero `docker-compose.yml`
3. Ejecutaremos ``docker compose up -d --remove-orphans``
4. Abriremos nuestro navegador e iremos a la siguiente dirección: http://localhost:8501
5. Cuando queramos cerrar nuestra aplicación, ejecutaremos: ``docker compose down``