
import tensorflow as tf
import numpy as np

def DRL_Agent2(s, eps, policy_net, actions, soft):
    """
    DRL_Agent2 function in Python.
    
    Parameters:
    s : current state 
    eps : from strategy
    policy_net : policy network
    actions : either number of actions (soft) or action index/indices (hard)
    soft : soft=1, hard=0
    
    Returns:
    a : selected action
    """

    rate = np.random.rand()
    s_dlarray = tf.convert_to_tensor(s, dtype=tf.float32)  # Convert state to TensorFlow tensor

    # Reshape s_dlarray to ensure it has a batch dimension (1, input_dim)
    s_dlarray = tf.expand_dims(s_dlarray, axis=0)

    b = policy_net(s_dlarray)  # Forward pass through the policy network
    b = tf.squeeze(b)  # Remove batch dimension if necessary (shape: [n_actions])

    n_actions = b.shape[0]  # Get the number of actions from the policy network output

    if soft == 1:
        if eps >= rate:
            a = np.random.choice(actions)  # Select random action
        else:
            m = tf.reduce_max(b)  # Find the maximum value
            c = tf.argmax(b)  # Find the index of the max value
            a = c.numpy()  # Convert TensorFlow tensor to a NumPy array
    else:
        if eps >= rate:
            if len(actions) > 1:
                a = np.random.choice(actions)  # Select random action from actions
            else:
                a = actions[0]  # Use the single available action
        else:
            # Ensure that `actions` contains valid indices within the range of policy network output
            valid_actions = [action for action in actions if 0 <= action < n_actions]
            print("here in drl agent 2 line 91")
            if len(valid_actions) > 0:
                b_actions = tf.gather(b, valid_actions)  # Select actions from b based on valid indices
                c = tf.argmax(b_actions)  # Find the index of the max value in the selected actions
                a = valid_actions[c.numpy()]  # Map back to original action index
            else:
                raise ValueError("No valid actions available based on the policy network output and given actions.")
    
    return int(a)  # Ensure 'a' is an integer
