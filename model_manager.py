import threading
import time
import numpy as np
from model import Network
from typing import Optional, List, Dict


# encargado del training
class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model: Optional[Network] = None
        self.training_in_progress: bool = False
        self.model_ready: bool = False
        self.message: str = ""
        self.trained_mode: Optional[str] = None

        self._training_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_training(self, x, y, learning_rate: float, steps: int, mode: str):
        if self.training_in_progress:
            return False

        self.training_in_progress = True
        self.model_ready = False
        self.message = "Inicializando redes neuronales..."
        self.trained_mode = mode

        input_dim = (
            1 if mode == "simple" else (x.shape[1] if hasattr(x, "shape") else 11)
        )

        self._training_thread = threading.Thread(
            target=self._background_train,
            args=(x, y, learning_rate, steps, [32, 32], input_dim),
            daemon=True,
        )
        self._training_thread.start()
        return True

    def _background_train(self, x, y, lr, steps, shape, input_dim):
        try:
            full_shape = [input_dim] + shape + [1]
            print(f"--- Iniciando entrenamiento: {full_shape} ---")

            self.model = Network(full_shape)
            self.message = "Entrenando modelo (optimizando pesos)..."

            time.sleep(0.1)

            self.model.train(x, y, lr, steps, None, False, 1000)

            self.model_ready = True
            self.download_model("./static/downloads/model.npz")
            self.message = f"Modelo {self.trained_mode.capitalize() if self.trained_mode else 'desconocido'} entrenado exitosamente."

        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            self.message = f"Error crítico: {str(e)}"
            self.model_ready = False
            self.model = None
        finally:
            self.training_in_progress = False

    def get_status(self) -> Dict:
        return {
            "model_ready": self.model_ready,
            "training_in_progress": self.training_in_progress,
            "training_message": self.message,
            "trained_mode": self.trained_mode,
        }

    def download_model(self, path):
        if self.model:
            self.model.save_params(path)

    def load_model(self, file_path):
        try:
            with np.load(file_path) as data:
                if "W1" not in data:
                    raise ValueError("El archivo no contiene pesos válidos (W1).")
                w1 = data["W1"]
                input_dim = w1.shape[0]

            mode = "simple" if input_dim == 1 else "advanced"
            self.trained_mode = mode

            full_shape = [input_dim, 32, 32, 1]

            print(f"Cargando modelo externo. Modo: {mode}, Shape: {full_shape}")
            self.model = Network(full_shape)

            self.model.load_params(file_path)

            self.model_ready = True
            self.training_in_progress = False
            self.message = f"Modelo cargado desde archivo ({mode})."

            self.download_model("./static/downloads/model")

            return True

        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.message = f"Error al cargar archivo: {str(e)}"
            self.model_ready = False
            return False


manager = ModelManager.get_instance()
