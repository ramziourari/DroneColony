"""using an lstm module to preprocess a variable sequence of agents."""
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf

agent_own_obs = 6
observation_per_agent = 4
max_num_agents = 2


class RecurrentModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(RecurrentModel, self).__init__(obs_space, action_space,
                                             num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="input_observations")
        own_observations = tf.slice(self.inputs, [0, 0], [-1, agent_own_obs], name="OWN_OBSERVATIONS")

        # split observations in own and others
        others_observations = tf.slice(self.inputs, [0, agent_own_obs], [-1, (max_num_agents - 1) * observation_per_agent], name="OTHERS_OBSERVATIONS")
        others_observations = tf.reshape(others_observations, [-1, max_num_agents - 1, observation_per_agent])

        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking for more infos
        tf.keras.layers.Masking(mask_value=0.0, input_shape=((max_num_agents - 1), observation_per_agent))

        # we process others and propagate the hidden state forward ignoring rows with 0's
        _, state_h, _ = tf.keras.layers.LSTM(
            32,
            return_sequences=True,
            return_state=True,
            name="LSTM_LAYER")(inputs=others_observations)

        # we concatenate observations (own and last hidden-state)
        concat_layer = tf.concat([own_observations, state_h], axis=1, name="CONCATENATE_OBSERVATIONS")

        # process with a fully
        layer_1 = tf.keras.layers.Dense(
            64,
            name="FC1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(concat_layer)
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
