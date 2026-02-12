from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import os
import datetime


from data_utils import get_data
from model_manager import manager
import predictor

app = Flask(__name__)


DOWNLOAD_FOLDER = os.path.join("static", "downloads")
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# obtiene datasets
def _get_context(prediction=None, error=None):
    st = manager.get_status()
    return {
        "model_ready": st["model_ready"],
        "training_in_progress": st["training_in_progress"],
        "training_message": st["training_message"],
        "trained_mode": st["trained_mode"],
        "sectors": predictor.SECTOR_LIST,
        "prediction": prediction,
        "error_message": error,
    }


# pagina principal
@app.route("/")
def index():
    error = request.args.get("error")
    return render_template("index.html", **_get_context(error=error))


# carga de modelo
@app.route("/upload_model", methods=["POST"])
def load_model():
    if "model_file" not in request.files:
        return render_template(
            "index.html", **_get_context(error="No se envió ningún archivo.")
        )

    file = request.files["model_file"]

    if file.filename == "":
        return render_template(
            "index.html", **_get_context(error="Nombre de archivo vacío.")
        )
    if not file.filename.endswith(".npz"):  # type: ignore
        return render_template(
            "index.html", **_get_context(error="Formato inválido. Debe ser .npz")
        )

    try:

        save_path = os.path.join(DOWNLOAD_FOLDER, "model.npz")
        file.save(save_path)

        success = manager.load_model(save_path)

        if not success:
            return render_template(
                "index.html",
                **_get_context(
                    error="Error interno al cargar la estructura del modelo."
                ),
            )

        return redirect(url_for("index"))

    except Exception as e:
        return render_template(
            "index.html", **_get_context(error=f"Error procesando archivo: {str(e)}")
        )


# entrenamiento
@app.route("/train", methods=["POST"])
def train():
    try:
        lr = float(request.form.get("learningRate", 0.001))
        steps = int(request.form.get("steps", 50000))
        mode = request.form.get("mode", "simple")

        data = get_data(mode)

        if not data or data[0] is None:
            return render_template(
                "index.html",
                **_get_context(
                    error="No se pudieron cargar los datos de entrenamiento (CSV faltante o corrupto)."
                ),
            )

        success = manager.start_training(data[0], data[1], lr, steps, mode)

        if not success:
            return render_template(
                "index.html", **_get_context(error="Ya hay un entrenamiento en curso.")
            )

        return redirect(url_for("index"))

    except Exception as e:
        return render_template(
            "index.html", **_get_context(error=f"Error fatal al iniciar: {str(e)}")
        )


# predicciones
@app.route("/predict", methods=["POST"])
def predict():
    try:
        st = manager.get_status()

        if not st["model_ready"] or manager.model is None:
            return render_template(
                "index.html",
                **_get_context(error="El modelo no está listo. Entrena o carga uno."),
            )

        req_mode = request.form.get("mode")
        current_mode = st["trained_mode"]

        if req_mode and req_mode != current_mode:

            req_mode = current_mode

        years = int(request.form.get("years", 100))
        smoothing = float(request.form.get("smoothing", 0.2))

        predictions = []
        prediction_years = []

        if current_mode == "simple":
            preset = request.form.get("preset", "moderate")

            _, _, (x_min, x_max), (y_min, y_max), meta = get_data("simple")

            last_val = float(meta.get("last_annual", 0.0) or 0.0)
            last_added = float(meta.get("last_added_emissions", 0.0) or 0.0)

            if preset == "pessimistic":

                ems = last_val * (1.008 ** np.arange(1, years + 1))
            elif preset == "aggressive":

                ems = np.linspace(last_val, 0.0, years)
            elif preset == "conservative":

                ems = np.linspace(last_val, last_val * 0.9, years)
            elif preset == "optimistic":

                ems = np.linspace(last_val, last_val * 0.7, years)
            else:
                ems = np.linspace(last_val, last_val * 0.5, years)

            load = last_added + np.cumsum(ems)  # type: ignore

            predictions = predictor.predict_simple(
                manager.model,
                load,
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
                inertia=smoothing,
            )

            current_year = datetime.datetime.now().year
            prediction_years = list(
                range(current_year, current_year + len(predictions))
            )

        else:
            weights = {}
            for sector in predictor.SECTOR_LIST:
                val = request.form.get(f"weight_{sector}", "0")
                weights[sector] = float(val)

            prediction_years, predictions = predictor.simulate_multi_sector(
                manager.model,
                years,
                weights,
                inertia=smoothing,
                start_delay=5,
                transition_speed=20,
            )

        df = pd.DataFrame({"years": prediction_years, "predictions": predictions})
        csv_path = os.path.join(DOWNLOAD_FOLDER, "graph.csv")
        df.to_csv(csv_path, index=False)

        return render_template(
            "index.html",
            **_get_context(
                prediction={"years": prediction_years, "predictions": predictions}
            ),
        )

    except Exception as e:
        print(f"Error prediction: {e}")
        return render_template(
            "index.html", **_get_context(error=f"Error en predicción: {str(e)}")
        )


# expone el estado del modelo
@app.route("/status")
def status_api():
    return jsonify(manager.get_status())


# main func
if __name__ == "__main__":
    model_path = os.path.join(DOWNLOAD_FOLDER, "model.npz")
    graph_path = os.path.join(DOWNLOAD_FOLDER, "graph.csv")

    if os.path.exists(graph_path):
        os.remove(graph_path)

    app.run(host="0.0.0.0", port=3434, debug=True, use_reloader=False)
