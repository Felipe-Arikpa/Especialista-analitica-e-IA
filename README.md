## Proyecto de Clasificación de Texto e Imágenes

Este paquete realiza predicciones de tratamientos médicos a partir de notas clínicas y clasificaciones de imágenes MRI.

### Requisitos Previos

* Python 3.8+
* pip

### Instalación

1. **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/tu-repositorio.git
    cd ESPECIALISTA-ANALITICA-E-IA
    ```

2. **Crea y activa un ambiente virtual:**
    ```bash
    python -m venv .venv
    # En Windows
    .venv\Scripts\activate
    # En macOS/Linux
    source .venv/bin/activate
    ```

3. **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### entrenar el modelo
Es necesario entrenar los modelos, ya que por su peso no hace parte del repositorio.
una vez activo el ambiente, usa los siguientes comandos desde la raíz del proyecto

```bash
python -m Homework.src._internals.txt_model
python -m Homework.src._internals.img_model
```

### Uso

Una vez activado el ambiente virtual, con las dependencias instaladas, y los modelos entrenados, puedes ejecutar los predictores desde la raíz del proyecto (`ESPECIALISTA-ANALITICA-E-IA`), reemplazando las rutas de ejemplo con tus propias rutas. Antes de probarlo, ten en cuenta qué:

`input_dir` es una carpeta que debe contener un único archivo `.csv` separado por punto y coma y/o una o varias imágenes en los formatos `.jpg`, `.jpeg` o `.png`.

El archivo `.csv` debe contener las siguientes columnas:

- **Case ID**: ID que identifica el caso.
- **Condition**: La clase de tumor identificado para el paciente.
- **Age**: Edad del paciente en años (número).
- **Sex**: Sexo del paciente representado con la letra `M` para masculino o `F` para femenino.
- **Clinical Note**: Texto narrativo tipo historia clínica, donde se describen los síntomas del paciente, así como su duración y severidad. Este texto debe estar en inglés.

---

### Ejecutar predicción

```bash
python -m Homework --input_dir "/ruta/a/tu/directorio_de_entrada" --output_dir "/ruta/donde/guardar/predicciones"
```
