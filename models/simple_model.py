"""using an lstm module to preprocess a variable sequence of agents."""
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf


class SimpletModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(SimpletModel, self).__init__(obs_space, action_space,
                                             num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="input_observations")
        # process with a fully
        layer_1 = tf.keras.layers.Dense(
            64,
            name="FC1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        # process with a fully
        layer_2 = tf.keras.layers.Dense(
            64,
            name="FC2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        # output actions and values
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="ACTIONS",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)
        value_out = tf.keras.layers.Dense(
            1,
            name="VALUE",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)

        self.base_model = tf.keras.Model(inputs=self.inputs, outputs=[layer_out, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
