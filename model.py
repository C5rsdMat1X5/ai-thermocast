class Network:
    # inicializar pesos
    def __init__(self, shape):
        import numpy

        self.np = numpy
        self.shape = shape
        self.params = {}

        for l in range(1, len(shape)):
            std = self.np.sqrt(2 / shape[l - 1])
            self.params[f"W{l}"] = self.np.random.normal(
                0.0, std, (shape[l], shape[l - 1])
            )
            self.params[f"b{l}"] = self.np.zeros((1, shape[l]))
        print(
            "Network initialized. Layer shapes:",
            [
                (self.params[f"W{l}"].shape, self.params[f"b{l}"].shape)
                for l in range(1, len(shape))
            ],
        )

    # forward pass
    def predict(self, inputs):
        activ = self.np.array(inputs).reshape(inputs.shape[0], -1)
        l_zs = []
        A_values = [activ]

        for l in range(1, len(self.shape)):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            z = self.np.dot(activ, W.T) + b
            l_zs.append(z)

            if l != len(self.shape) - 1:
                activ = self.np.maximum(0, z)
            else:
                activ = z
            A_values.append(activ)

        return A_values[-1], l_zs, A_values

    # backpropagation
    def compute_gradients(self, A_values, l_zs, Y):
        gradients = {}
        m = A_values[0].shape[0]
        L = len(l_zs)

        A_last = A_values[-1]
        Y = self.np.asarray(Y, dtype=float).reshape(A_last.shape)

        dA = A_last - Y

        for l in range(L, 0, -1):
            A = A_values[l]
            A_prev = A_values[l - 1]
            W = self.params[f"W{l}"]
            if l != L:
                Z = l_zs[l - 1]
                dZ = dA * (Z > 0).astype(float)
            else:
                dZ = dA

            dW = self.np.dot(dZ.T, A_prev) / m
            db = self.np.sum(dZ, axis=0, keepdims=True) / m

            gradients[f"dW{l}"] = dW
            gradients[f"db{l}"] = db

            dA = self.np.dot(dZ, W)

        return gradients

    # aplicar cambios del backprop
    def apply_gradients(self, gradients, lr=0.001, clip_grad=None):
        L = len(self.shape) - 1
        for l in range(1, L + 1):
            dW = gradients[f"dW{l}"]
            db = gradients[f"db{l}"]

            if clip_grad is not None:
                dW = self.np.clip(dW, -clip_grad, clip_grad)
                db = self.np.clip(db, -clip_grad, clip_grad)

            self.params[f"W{l}"] -= lr * dW
            self.params[f"b{l}"] -= lr * db

    # entrenar un paso
    def train_step(self, X, Y, lr=0.001, clip_grad=None, debug=False):
        A_last, l_zs, A_values = self.predict(X)
        loss = 0.5 * self.np.mean(self.np.power(A_last - Y, 2))

        grads = self.compute_gradients(A_values, l_zs, Y)
        self.apply_gradients(grads, lr=lr, clip_grad=clip_grad)

        if debug:
            print("Loss:", loss)
            for l in range(1, len(self.shape)):
                print(
                    f"Layer {l}: W shape {self.params[f'W{l}'].shape}, b shape {self.params[f'b{l}'].shape}"
                )
        return loss

    # realizar entrenamiento completo
    def train(
        self, X, Y, lr=0.001, steps=1000, clip_grad=None, debug=False, up_rate=100
    ):
        loss = []
        try:
            for step in range(1, steps + 1):
                t_loss = self.train_step(
                    X, Y, lr=lr, clip_grad=clip_grad, debug=(debug)
                )
                if up_rate > 0 and step % up_rate == 0:
                    loss.append(t_loss)

                    print(f"Step: {step}, Loss: {t_loss}")
        except KeyboardInterrupt:
            print("Early stop")
        return loss

    # guardar modelo
    def save_params(self, name):
        self.np.savez(f"{name}", shape=self.shape, **self.params)

    # cargar modelo
    def load_params(self, file):
        loaded = self.np.load(file)
        self.shape = (
            loaded["shape"].tolist()
            if isinstance(loaded["shape"], self.np.ndarray)
            else loaded["shape"]
        )
        self.params = {key: loaded[key] for key in loaded.keys() if key != "shape"}
