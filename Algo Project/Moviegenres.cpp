#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>

using namespace std;

class KNNClassifier {
private:
    int k;
    vector<vector<double>> X_train;  // Movie features (e.g., IMDb rating, duration)
    vector<string> y_train;          // Movie genres

public:
    KNNClassifier(int k) : k(k) {}

    void fit(const vector<vector<double>>& X, const vector<string>& y) {
        X_train = X;
        y_train = y;
    }

    string predict(const vector<double>& x) const {
        vector<pair<double, string>> distances;

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
        unordered_map<string, int> class_counts;
        for (int i = 0; i < k; ++i) {
            class_counts[distances[i].second]++;
        }

        // Predict the majority class among k-nearest neighbors
        string prediction = "";
        int max_count = 0;
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
    
    // Training data (features: IMDb rating, duration)
    vector<vector<double>> X_train = {
        {8.5, 120}, {7.2, 90}, {9.0, 150}, {6.5, 110}, {8.0, 130},
        {7.5, 105}, {8.2, 95}, {7.0, 140}, {6.8, 115}, {9.2, 125}
    };

    // Corresponding labels (movie genres)
    vector<string> y_train = {"Action", "Short-Films", "Documentary", "Drama", "Action", "Drama", "Action", "Documentary", "Drama", "Action"};

    // Get the value of k from the user
    int k=3;

    // Create and train the KNN classifier
    KNNClassifier knn(k);
    knn.fit(X_train, y_train);

    // Get test data from the user
    double imdbRating, duration;
    cout << "Enter IMDb rating of the new movie: ";
    cin >> imdbRating;
    cout << "Enter duration of the new movie: ";
    cin >> duration;

    // Make a prediction
    vector<double> X_test = {imdbRating, duration};
    string prediction = knn.predict(X_test);

    // Display the prediction
    cout << "Prediction: " << prediction << endl;

    return 0;
}
