# XGBoost Regression with TensorFlow Pooling and Loss

Consider XGBoost features are available on Individual level, predictions are required also on the Individual level but target is available for Groups of Individuals only. In this case, some sort of pooling known from neural networks, would be suitable. This library provides a way, how to use pooling from TensorFlow in XGBoost custom objective function.

![Architecture](/img/arch.png)

Predictions of XGBoost on the Individual level will be pooled to the Group level using a custom TensorFlow function. The same function uses one of TensorFlow losses to calculate the final scalar loss by comparing the target on Group level with the pooled predictions to the Group level.

This library provides a decorator, which turns the mentioned TensorFlow pooling and loss function to the XGBoost custom objective function, such that the whole aggregation and calculation of 1st and 2nd order derivatives is done seamlessly during XGBoost training.

The ![Example notebook](examples/example.ipynb) shows how to use this library on artificial data and compares performance of predictions on Group- vs Individual-level targets.


