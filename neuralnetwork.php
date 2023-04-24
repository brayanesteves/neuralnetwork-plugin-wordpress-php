<?php
/*
Plugin Name: Neural Network
Version: 0.0.1
Description: A neural network plugin without a library.
Author: Brayan Esteves
*/

class NeuralNetwork {

    private $input_neurons;
    private $hidden_neurons;
    private $output_neurons;
    private $weights_input_hidden;
    private $weights_hidden_output;
    private $bias_hidden;
    private $bias_output;

    function __construct($input_neurons, $hidden_neurons, $output_neurons) {
        $this->input_neurons = $input_neurons;
        $this->hidden_neurons = $hidden_neurons;
        $this->output_neurons = $output_neurons;
        $this->initialize_weights();
    }

    private function initialize_weights() {
        //initialize random weights for input->hidden and hidden->output layers
        for ($i = 0; $i < $this->input_neurons; $i++) {
            for ($j = 0; $j < $this->hidden_neurons; $j++) {
                $this->weights_input_hidden[$i][$j] = rand() / getrandmax();
            }
        }
        for ($i = 0; $i < $this->hidden_neurons; $i++) {
            for ($j = 0; $j < $this->output_neurons; $j++) {
                $this->weights_hidden_output[$i][$j] = rand() / getrandmax();
            }
        }

        //initialize bias values for hidden and output layers
        $this->bias_hidden = array();
        for ($i = 0; $i < $this->hidden_neurons; $i++) {
            $this->bias_hidden[$i] = rand() / getrandmax();
        }
        $this->bias_output = array();
        for ($i = 0; $i < $this->output_neurons; $i++) {
            $this->bias_output[$i] = rand() / getrandmax();
        }
    }

    private function sigmoid($x) {
        //apply sigmoid activation function
        return 1 / (1 + exp(-$x));
    }

    private function forward_propagation($input) {
        //calculate activation of hidden neurons
        $hidden = array();
        for ($j = 0; $j < $this->hidden_neurons; $j++) {
            $sum = 0;
            for ($i = 0; $i < $this->input_neurons; $i++) {
                $sum += $input[$i] * $this->weights_input_hidden[$i][$j];
            }
            $sum += $this->bias_hidden[$j];
            $hidden[$j] = $this->sigmoid($sum);
        }

        //calculate activation of output neurons
        $output = array();
        for ($j = 0; $j < $this->output_neurons; $j++) {
            $sum = 0;
            for ($i = 0; $i < $this->hidden_neurons; $i++) {
                $sum += $hidden[$i] * $this->weights_hidden_output[$i][$j];
            }
            $sum += $this->bias_output[$j];
            $output[$j] = $this->sigmoid($sum);
        }

        //return activations of both layers
        return array($hidden, $output);
    }

    private function backward_propagation($input, $target_output, $output) {
        //calculate output layer's error
        $output_error = array();
        for ($j = 0; $j < $this->output_neurons; $j++) {
            $output_error[$j] = ($target_output[$j] - $output[$j]) * $output[$j] * (1 - $output[$j]);
        }

        //calculate hidden layer's error
        $hidden_error = array();
        for ($j = 0; $j < $this->hidden_neurons; $j++) {
            $sum = 0;
            for ($k = 0; $k < $this->output_neurons; $k++) {
                $sum += $output_error[$k] * $this->weights_hidden_output[$j][$k];
            }
            $hidden_error[$j] = $sum * $output[$j] * (1 - $output[$j]);
        }

        //update weights and biases using errors and gradients
        for ($j = 0; $j < $this->hidden_neurons; $j++) {
            for ($k = 0; $k < $this->output_neurons; $k++) {
                $gradient = $output_error[$k] * $output[$k] * (1 - $output[$k]);
                $this->weights_hidden_output[$j][$k] += $gradient * $this->learning_rate * $this->hidden[$j];
                $this->bias_output[$k] += $gradient * $this->learning_rate;
            }
        }

        for ($i = 0; $i < $this->input_neurons; $i++) {
            for ($j = 0; $j < $this->hidden_neurons; $j++) {
                $gradient = $hidden_error[$j] * $this->hidden[$j] * (1 - $this->hidden[$j]);
                $this->weights_input_hidden[$i][$j] += $gradient * $this->learning_rate * $input[$i];
                $this->bias_hidden[$j] += $gradient * $this->learning_rate;
            }
        }
    }

    public function train($input_data, $output_data, $learning_rate = 0.1, $epochs = 1000) {
        $this->learning_rate = $learning_rate;
        //run training loop for specified epochs
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            for ($i = 0; $i < count($input_data); $i++) {
                list($hidden, $output) = $this->forward_propagation($input_data[$i]);
                $this->backward_propagation($input_data[$i], $output_data[$i], $output);
            }
        }
    }

    public function predict($input) {
        //return output of neural network given input
        list($hidden, $output) = $this->forward_propagation($input);
        return $output;
    }

}


function neural_network_shortcode($atts) {
    //extract input values from shortcode attributes
    extract(shortcode_atts(array(
        'input' => '',
        'weights_input_hidden' => '',
        'weights_hidden_output' => '',
        'bias_hidden' => '',
        'bias_output' => ''
    ), $atts));

    //parse input values
    $input = array_map('floatval', explode(',', $input));
    $weights_input_hidden = json_decode($weights_input_hidden);
    $weights_hidden_output = json_decode($weights_hidden_output);
    $bias_hidden = json_decode($bias_hidden);
    $bias_output = json_decode($bias_output);

    //initialize neural network using provided parameters
    $nn = new NeuralNetwork(count($input), count($weights_input_hidden[0]), count($bias_output));
    $nn->weights_input_hidden = $weights_input_hidden;
    $nn->weights_hidden_output = $weights_hidden_output;
    $nn->bias_hidden = $bias_hidden;
    $nn->bias_output = $bias_output;

    //generate prediction using neural network and input
    $prediction = $nn->predict($input);
    return implode(',', $prediction);
}

add_shortcode('neuralnetwork', 'neural_network_shortcode');