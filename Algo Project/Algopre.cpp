//example of a k-nearest neighbors(KNN) algorithm

//Using knn algorithm the KNN classifier to classify a new instance based on height and weight
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>

using namespace std;

class KNNClassifier {
private:
    int k;
    vector<vector<double>> X_train;
    vector<int> y_train;

public:
    KNNClassifier(int k) : k(k) {}

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        X_train = X;
        y_train = y;
    }

    int predict(const vector<double>& x) const {
        vector<pair<double, int>> distances;

        // Calculate distances to all training examples
        for (size_t i = 0; i < X_train.size(); ++i) {
            double distance = 0.0;
            for (size_t j = 0; j < X_train[i].size(); ++j) {
                distance += pow(X_train[i][j] - x[j], 2);
            }
            distance = sqrt(distance);
            distances.emplace_back(distance, y_train[i]);
        }

        // Sort distances and find the k-nearest neighbors
        sort(distances.begin(), distances.end());
        unordered_map<int, int> class_counts;
        for (int i = 0; i < k; ++i) {
            class_counts[distances[i].second]++;
        }

        // Predict the majority class among k-nearest neighbors
        int prediction = -1, max_count = 0;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                prediction = pair.first;
            }
        }

        return prediction;
    }
};

int main() {
    // Example usage:

    // Training data
    vector<vector<double>> X_train = {{150, 60}, {180, 90}, {160, 70}, {170, 80}};
    vector<int> y_train = {0, 1, 0, 1};  // Class labels (0: Normal BMI, 1: Overweight)

    // User input for test data
   
    vector<double> X_test={157,90};
    
   
    // Create and train the KNN classifier
    KNNClassifier knn(3); // Use 3 nearest neighbors for classification
    knn.fit(X_train, y_train);

    // Make a prediction
    int prediction = knn.predict(X_test);

    // Display the prediction
    cout << "Prediction: ";
    
    if(prediction==0)
    {
        cout<<"Normal BMI";
    }
    else
    {
        cout<<"Overweight";
    }

    return 0;
}
