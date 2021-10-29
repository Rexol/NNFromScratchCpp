//
// Created by Rexol on 25.10.2021.
//

// TODO: add Dropout().
// TODO: may be add trigger neuron for each layer.

#include <bits/stdc++.h>
using namespace std;

#ifndef C__MNIST_MYNEURO_H
#define C_MNIST_MYNEURO_H

class Layer {
private:
    int _in; // Number of inputs.
    int _out; // Number of outputs.
    int _gradNum = 0; // Number of training calculation passed before parameters update.
    double _learning_rate;
    std::string _afname; // Name of activation function.
    double _dropoutP; // Not released yet.

    vector<vector<double>> _matrix; // Store weights and biases.
    vector<vector<double>> _gradient; // Stores layer gradient.
    vector<double> _hidden; // Vector, calculated by layer.
    vector<double> _errors; // Stores errors for previous layers.

    vector<double> (*_activationFunction)(vector<double>&);
    vector<double> (*_derivateAF)(vector<double>&);

public:
    explicit Layer(int inputs=0, int outputs=0, const std::string& actFunc="relu",
          double lr=0.001, double dropout=0);

    void setIO(int inputs, int outputs);
    void setFunct(const std::string& actFunc);
    void setDropout(double p) {_dropoutP = p;}

    void setLearningRate(double lr) {_learning_rate = lr;}
    double learning_rate() const {return _learning_rate;}

    int in() const {return _in;}
    int out() const {return _out;}

    vector<vector<double>>& matrix() {return _matrix;}
    void fillRandom();
    void setMatrix(vector<vector<double>> &newMatrix) {_matrix = newMatrix;}

    vector<double>& errors() {return _errors;}
    void setErrors(vector<double> & newErrors) {_errors = newErrors;}

    vector<double>& hidden() {return _hidden;}
    void setHidden(vector<double> & newHidden) {_hidden = newHidden;}

    void updateWeights();

    vector<double> (*activationFunction())(vector<double>&)  {return _activationFunction;}
    vector<double> (*derivateAF())(vector<double>&) {return _derivateAF;}

    void calculate(vector<double> &inputs, bool mode); // Mode for Dropout;
    void calculateOutError(vector<double> &targets, vector<double> &input);
    void calcHiddenError(vector<double> &prevErrors, vector<vector<double>> &outWeights, vector<double> &input);

    void resetErrors();

    void Dropout(); // Not released.;
};

class Network {
private:
    int _in; // Size of input vector.
    int _out; // Size of output vector.
    int _batchSize;
    int _numLayers;

    vector<Layer> _layers; // Store layers.
    vector<vector<double>> _inputBatch; // Stores input vectors.
    vector<char> _targets; // Only labels.

    double (*_lossFunction)(vector<vector<double>>&, vector<char>&); // pred, target;
    vector<double> (*_lossFunctionDerivative)(vector<double>&, char);

public:
    Network(int ins, int outs, vector<Layer> &layersList, int batchSize);

    int numOfLayers() const {return _numLayers;}
    int in() const {return _in;}
    int out() const {return _out;}

    double (*lossFunction())(vector<vector<double>>&, vector<char>&){return _lossFunction;}
    vector<double> (*lossDerivate())(vector<double>&, char) {return _lossFunctionDerivative;}
    void setLossFunction(const std::string& function);

    vector<vector<double>> &inputBatch() {return _inputBatch;}
    vector<char> &targets() {return _targets;}
    void setInputs(vector<vector<double>> &input) {_inputBatch = input;}
    void setTargets(vector<char> &target) {_targets = target;}

    Layer layers(int index) {return _layers[index];}

    void feedForward(vector<double> &input, bool train, char target=0);
    void zeroGrad();

    void calculateErrors(vector<double> &input, char target);
    void step();

    void train(vector<vector<double>> &input, vector<char> &target);
    vector<char> predict(vector<vector<double>> &inBatch);
};
#endif //C__MNIST_MYNEURO_H
