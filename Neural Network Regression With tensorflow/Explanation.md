# Everything You Need To Know About Neural Network Regression With Tf2_Keras

## 1. General Neural Network Concepts

* **Neuron (or unit)** → A small mathematical function that takes numbers, does a calculation, and passes it forward. Like a mini calculator.
* **Layer** → A group of neurons working together.
* **Activation function** → A rule that decides if a neuron "fires" or not (adds non-linearity). Common ones:

  * `relu` → Keeps positive values, turns negatives into 0.
  * `sigmoid` → Squashes values into 0–1 range.
  * `tanh` → Squashes values into -1–1 range.
* **Input layer** → First layer; takes in your features (like house size, location, etc.).
* **Hidden layer** → Middle layers that do the “thinking” (feature extraction).
* **Output layer** → Produces the prediction (e.g., one number for regression).

## 2. TensorFlow / Keras Model Building Keywords

* **`Sequential`** → The simplest model type in Keras. Layers are stacked one after the other (like a sandwich).

  ```python
  model = keras.Sequential([...])
  ```
* **`Dense`** → A fully connected layer (every neuron connects to every neuron in the next layer).

  ```python
  layers.Dense(64, activation="relu")
  ```
* **`input_shape`** → Tells the model what shape your data has (number of features). Example: `(10,)` if you have 10 input features.
* **`activation`** → Function applied to neuron outputs (`relu`, `sigmoid`, etc.).
* **`Dropout`** → A regularization method: randomly turns off some neurons during training so the network doesn’t “memorize” data.
* **`Normalization`** → A layer that scales inputs so all features have similar ranges (important for stability).

## 3. Training Configuration Keywords

* **`compile()`** → Sets up the model for training: choose optimizer, loss, metrics.

  ```python
  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  ```
* **`optimizer`** → Algorithm that adjusts weights to reduce error.

  * `adam` → A popular optimizer that adapts learning rates automatically.
  * `sgd` → Stochastic Gradient Descent (simpler, but slower).
* **`loss`** → What the model tries to minimize (the “error” measure).

  * `mse` (Mean Squared Error) → Squares differences, punishes large errors.
  * `mae` (Mean Absolute Error) → Average of absolute differences, more robust to outliers.
  * `huber` → Mix between MSE and MAE.
* **`metrics`** → Extra measurements you want to track (not used for training, just reporting). Example: `"mae"`.

## 4. Training Process Keywords

* **`fit()`** → Starts the training process. You give it data, and it updates weights.

  ```python
  model.fit(X_train, y_train, epochs=100, batch_size=32)
  ```
* **`epochs`** → How many times the model goes through the entire training dataset.
* **`batch_size`** → How many samples are processed before the model updates its weights.
* **`validation_split`** → Fraction of training data kept aside for checking progress (not used for training).
* **`callbacks`** → Extra tools during training (like saving models, stopping early, adjusting learning rate).

## 5. Evaluation & Prediction Keywords

* **`evaluate()`** → Tests the model on unseen data (gives loss and metrics).
* **`predict()`** → Generates outputs for new input data. Example: predict house price for a new house.
* **`history`** → Object returned by `fit()`, stores training/validation loss & metrics per epoch (useful for plotting learning curves).


## 6. Regularization & Optimization Keywords

* **Overfitting** → When a model memorizes training data but fails on new data.
* **Regularization** → Techniques to prevent overfitting:

  * **L1/L2 penalties** → Add extra cost for big weights.

    ```python
    keras.regularizers.l2(0.01)
    ```
  * **Dropout** → Randomly disable neurons.
* **Learning rate** → Controls how big the steps are when updating weights. Too high = unstable; too low = slow.
* **Learning rate schedule** → Adjust learning rate during training.

  * Example: `ReduceLROnPlateau` lowers learning rate when validation error stops improving.

## 7. Callbacks Keywords

* **`EarlyStopping`** → Stops training if validation doesn’t improve (prevents overfitting & saves time).
* **`ModelCheckpoint`** → Saves the best model during training.
* **`TensorBoard`** → Visualization tool to track training progress.

## 8. Advanced Keywords

* **Custom loss function** → You can write your own error function.
* **Multi-output regression** → A model that predicts multiple numbers at once.
* **Ensemble** → Use several models together to improve predictions.
* **Uncertainty estimation** → Measuring how confident the model is (important in regression tasks like forecasting).


**In short:**

* **Model building keywords**: `Sequential`, `Dense`, `activation`, `input_shape`
* **Training keywords**: `compile`, `fit`, `epochs`, `batch_size`, `loss`, `optimizer`, `metrics`
* **Evaluation keywords**: `evaluate`, `predict`, `history`
* **Regularization keywords**: `Dropout`, `L1/L2`, `early_stopping`
* **Optimization keywords**: `learning_rate`, `ReduceLROnPlateau`, `adam`