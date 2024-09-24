import tensorflow as tf

def modelGradients(dlnet, dlX, Y):
    """
    Compute gradients and loss for the neural network using TensorFlow.

    Parameters:
    - dlnet: The neural network model (similar to `dlnet` in MATLAB).
    - dlX: The input data (equivalent to `dlX` in MATLAB).
    - Y: The target output data (equivalent to `Y` in MATLAB).

    Returns:
    - gradients: The computed gradients of the loss with respect to the model's trainable variables.
    - loss: The computed loss value (Mean Squared Error).
    """
    # Use GradientTape to track operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass: predict the output based on input dlX
        dlYPred = dlnet(dlX, training=True)  # Ensure training=True for correct layer behavior

        # Compute the Mean Squared Error (MSE) loss
        loss = tf.keras.losses.mean_squared_error(Y, dlYPred)
        loss = tf.reduce_mean(loss)  # Reduce mean to get scalar loss
    
    # Compute gradients of the loss with respect to the model's trainable variables
    gradients = tape.gradient(loss, dlnet.trainable_variables)
    
    return gradients, loss
