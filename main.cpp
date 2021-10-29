/*
 * This is the neural network implementation for MNIST images classification.
 * The code was written from scratch.
 * Author: Vladimir Surtaev
 * Date: 26.10.2021
 */

#include <bits/stdc++.h>

#include "myNeuro.h"

using namespace std;

// Constants:
const string TRAIN_IMPATH = R"(C:\Users\...\C++Mnist\train-images.idx3-ubyte)";
const string TRAIN_LBPATH = R"(C:\Users\...\C++Mnist\train-labels.idx1-ubyte)";
const string TEST_IMPATH = R"(C:\Users\...\C++Mnist\t10k-images.idx3-ubyte)";
const string TEST_LBPATH = R"(C:\Users\...\C++Mnist\t10k-labels.idx1-ubyte)";

const int WIDTH = 28;
const int HEIGHT = 28;
const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int BATCH_SIZE = 64;
const double LEARNING_RATE = 1e-2;
const int EPOCHS = 20;
// End of constant block.


int reverseInt (int i) { // Source: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// This function read mnist files and puts data in structure.
pair<vector<vector<double>>, vector<char>> dataLoader(bool train) {
    pair<vector<vector<double>>, vector<char>> dataset;
    string imagePath;
    string labelPath;
    if (train) {
        dataset.first.resize(TRAIN_SIZE);
        dataset.second.resize(TRAIN_SIZE);
        imagePath = TRAIN_IMPATH;
        labelPath = TRAIN_LBPATH;
    } else {
        dataset.first.resize(TEST_SIZE);
        dataset.second.resize(TEST_SIZE);
        imagePath = TEST_IMPATH;
        labelPath = TEST_LBPATH;
    }

    ifstream file (imagePath, ios::binary); // Source: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    if (file.is_open()) {
        cout << "   File " + imagePath<<" was opened."<<endl;
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        for(auto & j : dataset.first) {
            j.resize(WIDTH*HEIGHT);
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));
                    j[r*n_rows + c] = temp;
                }
            }
        }
    }
    file.close();

    file.open(labelPath, ios::binary); // Source: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    if (file.is_open()) {
        cout << "   File " + labelPath<<" was opened."<<endl;
        int magic_number = 0;
        int number_of_items = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_items, sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);
        for(char & j : dataset.second) {
            unsigned char temp = 0;
            file.read((char *) &temp, sizeof(temp));
            j = (char)temp;
        }
    }

    return dataset;
}

// This function makes dataset from the shuffled data.
vector<pair<vector<vector<double>>, vector<char>>> makeDataset(pair<vector<vector<double>>, vector<char>> & data) {
    vector<pair<vector<vector<double>>, vector<char>>> res(data.first.size() / BATCH_SIZE);

    vector<int> order(data.first.size());
    std::iota(order.begin(), order.end(), 0);

    default_random_engine rng;
    rng = std::default_random_engine{};
    std::shuffle(order.begin(), order.end(), rng);

    int c = 0;
    for(auto & re : res) {
        re.first.resize(BATCH_SIZE);
        re.second.resize(BATCH_SIZE);
        for(int j = 0; j < BATCH_SIZE; ++j) {
            re.first[j] = data.first[order[c]];
            re.second[j] = data.second[order[c]];
            c += 1;
        }
    }
    return res;
}

// This function calculates accuracy for trained model.
double accuracy(Network *model, vector<pair<vector<vector<double>>, vector<char>>> &dataset) {
    int correct = 0;
    int total = 0;
    for(int batch = 0; batch < dataset.size(); ++ batch) {
        vector<char> output(BATCH_SIZE);
        output = model->predict(dataset[batch].first);
        for(int j = 0; j < BATCH_SIZE; j++) {
            if (output[j] == dataset[batch].second[j]) {
                correct += 1;
            }
            total += 1;
        }
    }
    cout << "Correct: " << correct << " Total:" << total <<endl;
    return (double)correct / (double)total;
}


int main() {
    cout << "Program started: \n";
    pair<vector<vector<double>>, vector<char>> traindata = dataLoader(true);
    pair<vector<vector<double>>, vector<char>> testdata = dataLoader(false);

    vector<pair<vector<vector<double>>, vector<char>>> train = makeDataset(traindata);
    vector<pair<vector<vector<double>>, vector<char>>> test = makeDataset(testdata);
    cout << "Train dataset: " << train.size() << " batches." << endl;
    cout << "Test dataset: " << test.size() << " batches." << endl;
    cout << "Batch size is: " << BATCH_SIZE << " images." << endl;

    vector<Layer> layers(3);
    layers[0].setIO(WIDTH*HEIGHT, 256);
    layers[0].setFunct("relu");
    layers[0].setLearningRate(LEARNING_RATE);
    layers[1].setIO(256, 128);
    layers[1].setFunct("relu");
    layers[1].setLearningRate(LEARNING_RATE);
    layers[2].setIO(128, 10);
    layers[2].setFunct("softmax");
    layers[2].setLearningRate(LEARNING_RATE);

    Network myNetwork(WIDTH*HEIGHT, 10, layers, BATCH_SIZE);
    myNetwork.setLossFunction("nllloss");


    vector<double> trainAccuracy(EPOCHS);
    vector<double> testAccuracy(EPOCHS);

    for(int epoch = 0; epoch < EPOCHS; ++epoch) {
        cout << "Current epoch: " << epoch << endl;
        for(int batch = 0; batch < train.size(); ++batch) {
            myNetwork.train(train[batch].first, train[batch].second);
        }
        cout << "   Training finished." << endl;
        trainAccuracy[epoch] = accuracy(&myNetwork, train);
        testAccuracy[epoch] = accuracy(&myNetwork, test);
        cout << "   Train accuracy: " << trainAccuracy[epoch] << endl;
        cout << "   Test accuracy: " << trainAccuracy[epoch] << endl;
        train = makeDataset(traindata);
        test = makeDataset(testdata);
    }


    return 0;
}
