//
// Created by Rexol on 25.10.2021.
//
#include "myNeuro.h"
#include <bits/stdc++.h>

using namespace std;


// Functions that will be used in network.
vector<double> ReLU(vector<double> &x) {
    vector<double> arr(x.size());
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = x[i] > 0.0? x[i]: 0.0;
    }
    return arr;
}
vector<double> ReLuDerivative(vector<double> &x) {
    vector<double> arr(x.size());
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = x[i] > 0.0? 1.0: 0.0;
    }
    return arr;
}

vector<double> sigmoid(vector<double> &x) {
    vector<double> arr(x.size());
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = (1.0 / (1.0 + exp(-x[i])));
    }
    return arr;
}
vector<double> sigmoidDerivative(vector<double> &x) {
    vector<double> arr(x.size());
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = (1.0 / (1.0 + exp(-x[i])))*(1-(1.0 / (1.0 + exp(-x[i]))));
    }
    return arr;
}

vector<double> softMax(vector<double> &x) {
    vector<double> arr(x.size());
    double sumOfExp = 0.0;
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = exp(x[i]);
        sumOfExp += arr[i];
    }
    for(int i = 0; i < x.size(); ++i) {
        arr[i] /= sumOfExp;
    }
    return arr;
}
vector<double> softMaxDerivative(vector<double> &x){
    vector<double> arr(x.size());
    double sumOfExp = 0.0;
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = exp(x[i]);
        sumOfExp += arr[i];
    }
    for(int i = 0; i < x.size(); ++i) {
        arr[i] = arr[i]*(sumOfExp-arr[i]) / pow(sumOfExp,2);
    }
    return arr;
}

double NLLLoss (vector<vector<double>> &predicted, vector<char> &target) { // Negative log likelihood.
    double res = 0.0;
    for (int i = 0; i < target.size(); ++i) { // Since we have multi-class classification we can just get probability
        res -= 0.0 + log(predicted[i][(int)target[i]]);  // of target class.
    }
    return res;
}
vector<double> NLLLossDerivative (vector<double> &predicted, char target) { // Counts loss for one prediction.
    vector<double> arr(predicted.size(), 0);
    arr[target] = -1.0*(double)target / predicted[(int)target]; // Can divide, since softmax is positive.

    return arr;
}

double MSELoss (vector<vector<double>> &pred, vector<char> & target) {
    double res = 0.0;
    for(int i = 0; i < target.size(); ++i) {
        for(int j = 0; j < pred[i].size(); ++j) {
            if (j == (int)target[i]) {
                res += pow(1-pred[i][j],2);
            } else {
                res += pow(pred[i][j], 2);
            }
        }
    }
    return res / (double)target.size();
}

vector<double> MSELossDerivative(vector<double> &pred, char target) {
    vector<double> res(pred.size());
    for(int i = 0; i < pred.size(); ++i) {
        if (i == (int)target){
            res[i] = 2*(1 - pred[i]);
        } else {
            res[i] = 2*(0 - pred[i]);
        }
    }
    return res;
}

//-CLASS-LAYER----------

/*
 * Constructor:
 *  input params:
 *      (int) inputs -- number of inputs to layer
 *      (int) outputs -- number of outputs of layer
 *      (string) actFunc -- name of activation function:
 *          possible values: "sigmoid", "relu", "softmax"
 *          "relu" by default.
 *      (double) lr -- learning rate
 *      (double) dropout -- dropout probability (NOT RELEASED YET)
 */
Layer::Layer(int inputs, int outputs, const std::string& actFunc, double lr, double dropout) {
    _in = inputs;
    _out = outputs;
    _learning_rate = lr;
    _dropoutP = dropout;

    _hidden.resize(_out);
    _errors.resize(_out);

    if (actFunc == "sigmoid") {
        _afname = "sigmoid";
        _activationFunction = sigmoid;
        _derivateAF = sigmoidDerivative;
    } else if (actFunc == "softmax") {
        _afname = "softmax";
        _activationFunction = softMax;
        _derivateAF = softMaxDerivative;
    } else {
        _afname = "relu";
        _activationFunction = ReLU;
        _derivateAF = ReLuDerivative;
    }

    _matrix.resize(_in+1); // We'll be storing bias vector in _matrix[_in].
    for(int ins = 0; ins < _in+1; ++ins) {
        _matrix[ins].resize(_out);
    }
    _gradient.resize(_in); // Don't have gradient for biases since we can calculate it from _errors vector.
    for(int ins = 0; ins < _in; ++ins) {
        _gradient[ins].resize(_out);
    }
}

/*
 * This method sets number of inputs and outputs of the layer.
 *  input params:
 *      (int) inputs -- number of inputs to layer
*       (int) outputs -- number of outputs of layer
 */
void Layer::setIO(int inputs, int outputs) {
    _in = inputs;
    _out = outputs;

    _hidden.resize(_out);
    _errors.resize(_out);
    _gradient.resize(_out);

    _matrix.resize(_in+1); // We'll be storing bias vector in _matrix[_in].
    for(int i = 0; i < _in + 1; ++i) {
        _matrix[i].resize(_out);
    }
    _gradient.resize(_in); // Don't have gradient for biases since we can calculate it from _errors vector.
    for(int i = 0; i < _in; ++i) {
        _gradient[i].resize(_out);
    }
}

/*
 * This method sets activation function for the layer.
 *  input params:
 *      (string) actFunc -- name of activation function
 *          possible values: "sigmoid", "relu", "softmax";
 *          "relu" by default.
 */
void Layer::setFunct(const std::string& actFunc) {
    if (actFunc == "sigmoid") {
        _afname = "sigmoid";
        _activationFunction = sigmoid;
        _derivateAF = sigmoidDerivative;
    } else if (actFunc == "softmax") {
        _afname = "softmax";
        _activationFunction = softMax;
        _derivateAF = softMaxDerivative;
    } else {
        _afname = "relu";
        _activationFunction = ReLU;
        _derivateAF = ReLuDerivative;
    }
}

/*
 * This method fills _matrix with random values depending on activation function.
 */
void Layer::fillRandom() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    if(_afname == "relu") {
        std::normal_distribution<> dis{0.0, sqrt(2.0/(double)_in)}; // He initialization.
        for(int ins = 0; ins < _in+1; ++ins) {
            for (int outs = 0; outs < _out; ++outs) {
                _matrix[ins][outs] = dis(gen);
            }
        }
    } else if(_afname == "sigmoid" || _afname == "softmax") { // Xavier uniform initialization.
        std::uniform_real_distribution<> dis(-sqrt(6.0) / sqrt(_in + _out), sqrt(6.0) / sqrt(_in + _out));
        for (int ins = 0; ins < _in + 1; ++ins) {
            for (int outs = 0; outs < _out; ++outs) {
                _matrix[ins][outs] = dis(gen);
            }
        }
    } else {
        for (int ins = 0; ins < _in + 1; ++ins) { // Not preferable.
            for (int outs = 0; outs < _out; ++outs) {
                _matrix[ins][outs] = ((((double) rand() / (double) RAND_MAX) - 0.5) * pow(_out, -0.5));
            }
        }
    }
}

/*
 * This function updates _matrix according to gradients.
 */
void Layer::updateWeights() { // bias gradient from _errors, not from _grad.
    for(int outs = 0; outs < _out; outs++) {
        //_errors[outs] = _errors[outs] / _gradNum; // Change somehow large powers of errors.
        for(int ins = 0; ins < _in; ins++) {
            //_gradient[ins][outs] = _gradient[ins][outs] / _gradNum; // Change somehow large powers of gradient. TODO: try to leave just sign.
            _matrix[ins][outs] -= (_learning_rate * _gradient[ins][outs]);
        }
        _matrix[_in][outs] -= (_learning_rate * _errors[outs]);
    }
}

/*
 * This function calculates output of the layer.
 *  input params:
 *      (vector<double>) inputs -- input values
 *      (bool) mode -- enables dropout if true (training mode)
 */
void Layer::calculate(vector<double> &inputs, bool mode) {
    for(int hid = 0; hid < _out; ++hid) {
        _hidden[hid] = 0.0;
        for(int ins = 0; ins < _in; ++ins) {
            _hidden[hid] += inputs[ins] * _matrix[ins][hid];
        }
        _hidden[hid] += _matrix[_in][hid];
    }
    _hidden = _activationFunction(_hidden);
}

/*
 * This function calculates errors and gradient for output layer
 *  input params:
 *  (vector<double>) lossGrad -- partial derivatives of loss function
 *  (vector<double>) input -- input that was given to the layer on calculation stage
 */
void Layer::calculateOutError(vector<double> &lossGrad, vector<double> &input) {
    vector<double> layerErrors = _derivateAF(_hidden);
    for(int outs = 0; outs < _out; ++outs) {
        _errors[outs] += layerErrors[outs] * lossGrad[outs];
        for(int ins = 0; ins < _in; ins++) {
            _gradient[ins][outs] += layerErrors[outs] * lossGrad[outs] * input[ins];
        }
    }
    _gradNum += 1;
}

/*
 * This function calculates errors and gradient for first and hidden layers
 *  input params:
 *      (vector<double>) prevErrors -- errors from subsequent layer
 *      (vector<vector<double>>) outWeights -- _matrix from subsequent layer
 *      (vector<double>) input -- input that was given to layer on calculation stage
 */
void Layer::calcHiddenError(vector<double> &prevErrors, vector<vector<double>> &outWeights, vector<double> &input) {
    vector<double> newErrors(_out, 0.0);
    vector<double> derivatives = _derivateAF(_hidden);
    for(int hid = 0; hid < outWeights.size()-1; ++hid) { // Since we have 1 more row for bias in _matrix.
        for(int outs = 0; outs < outWeights[hid].size(); ++outs) {
            newErrors[hid] += prevErrors[outs] * outWeights[hid][outs];
        }
        _errors[hid] += newErrors[hid] * derivatives[hid];

        for(int ins = 0; ins < _in; ++ins) {
            _gradient[ins][hid] += newErrors[hid] * derivatives[hid] * input[ins];
        }
    }
    _gradNum += 1;
}

/*
 * This method resets _errors and _gradient vectors of the layer.
 */
void Layer::resetErrors() {
    for(auto & i : _gradient) {
        std::fill(i.begin(), i.end(), 0.0);
    }
    std::fill(_errors.begin(), _errors.end(), 0.0);
    _gradNum = 0;
}


//----------NETWORK----------

/*
 * Constructor:
 *  input params:
 *      (int) ins -- number of inputs of the network
 *      (int) outs -- number of outputs of the network
 *      (vector<Layer>) layerList -- vector of layers of network
 *      (int) batchSize -- size of minibatch
 */
Network::Network(int ins, int outs, vector<Layer> &layersList, int batchSize) {
    _in = ins;
    _out = outs;
    _batchSize = batchSize;
    _layers = layersList;
    _numLayers = layersList.size();

    for(auto & layer : _layers) {
        layer.fillRandom();
    }

    _inputBatch.resize(_batchSize);
    for(int i = 0; i < _batchSize; ++i) {
        _inputBatch[i].resize(_in);
    }
    _targets.resize(_out);

    _lossFunction = NLLLoss;
    _lossFunctionDerivative = NLLLossDerivative;
}

/*
 * This function calculates the output of the network
 *  input params:
 *      (vector<double>) input -- input vector
 *      (bool) train -- train if true
 *      (char) target -- expected class for input (requires if train == true)
 *          need for errors calculation
 */
void Network::feedForward(vector<double> &input, bool train, char target) {
    _layers[0].calculate(input, train);
    for(int i = 1; i < _numLayers; ++i) {
        _layers[i].calculate(_layers[i-1].hidden(), train);
    }

    if (train) {
        calculateErrors(input, target);
    }
}

/*
 * This method reset gradient and errors of all layers of network.
 */
void Network::zeroGrad() {
    for(auto & i : _layers) {
        i.resetErrors();
    }
}

/*
 * This function calculates errors of the network
 *  input params:
 *      (vector<double>) input -- input vector
 *      (char) target -- expected class
 */
void Network::calculateErrors(vector<double> &input, char target) {
    vector<double> lossGrad = _lossFunctionDerivative(_layers[_numLayers -1].hidden(), target);
    _layers[_numLayers-1].calculateOutError(lossGrad, _layers[_numLayers -2].hidden());

    for(int i = _numLayers - 2; i > 0; --i) {
        _layers[i].calcHiddenError(_layers[i + 1].errors(), _layers[i + 1].matrix(), _layers[i-1].hidden());
    }
    _layers[0].calcHiddenError(_layers[1].errors(), _layers[1].matrix(), input);
}

/*
 * This method updates parameters of the model.
 */
void Network::step() {
    for(auto & layer:_layers) {
        layer.updateWeights();
    }
}

/*
 * This function makes one train iteration
 *  input params:
 *      (vector<vector<double>>) input -- input training minibatch
 *      (vector<char>) target -- expected classes for every vector in input
 */
void Network::train(vector<vector<double>> &input, vector<char> &target) {
    setInputs(input);
    setTargets(target);

    for(int i = 0; i < _batchSize; ++i) {
        feedForward(_inputBatch[i], true, _targets[i]);
    }

    step();
    zeroGrad();
}

/*
 * This function calculates output of the model
 *  input params:
 *      (vector<vector<double>>) input -- input batch.
 */
vector<char> Network::predict(vector<vector<double>> &input) {
    setInputs(input);
    vector<char> output(_batchSize);

    for(int i = 0; i < _batchSize; ++i) {
        feedForward(_inputBatch[i], false);

        vector<pair<double, char>> res(10);
        for(int j = 0; j < 10; ++j) {
            res[j].first = _layers[_numLayers-1].hidden()[j];
            res[j].second = (char)j;
        }
        output[i] = std::max_element(res.begin(), res.end())->second;
    }
    return output;
}

/*
 * This function sets loss function to the model
 *  input params:
 *      (string) function -- name of loss function
 *          possible values: "nllloss", "mse";
 *          "nllloss" by default.
 */
void Network::setLossFunction(const std::string& function) {
    if (function == "mse") {
        _lossFunction = MSELoss;
        _lossFunctionDerivative = MSELossDerivative;
    } else {
        _lossFunction = NLLLoss;
        _lossFunctionDerivative = NLLLossDerivative;
    }
}



