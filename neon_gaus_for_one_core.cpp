#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <arm_neon.h>
#include <pthread.h>
#include <random>

using namespace std;

// метод Гаусса с NEON оптимизацией. Для нескольких ядер
template <typename T>
void gaussianElimination(vector<vector<T>>& A, vector<T>& b) {
    const int n = A.size();
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }

        swap(A[i], A[pivot]);
        swap(b[i], b[pivot]);

        for (int j = i + 1; j < n; ++j) {
            T factor = A[j][i] / A[i][i];

            float32x4_t factorVec = vdupq_n_f32(static_cast<float>(factor));
            for (int k = i; k < n; k += 4) {
                float32x4_t aVec = vld1q_f32(&A[j][k]);
                float32x4_t pivotVec = vld1q_f32(&A[i][k]);
                float32x4_t result = vsubq_f32(aVec, vmulq_f32(factorVec, pivotVec));
                vst1q_f32(&A[j][k], result);
            }
            b[j] -= factor * b[i];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        b[i] /= A[i][i];
        A[i][i] = 1;

        for (int j = 0; j < i; ++j) {
            b[j] -= A[j][i] * b[i];
            A[j][i] = 0;
        }
    }
}

int main() {
    int n, seed;
    cout << "Введите размер системы: ";
    cin >> n;
    cout << "Введите зерно для генерации случайных чисел: ";
    cin >> seed;

    vector<vector<float>> A(n, vector<float>(n));
    vector<float> b(n);

    mt19937 gen(seed);
    uniform_real_distribution<float> dis(1.0, 10.0);

    for (int i = 0; i < n; ++i) {
        A[i].resize(n);
        for (int j = 0; j < n; ++j) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }

    auto start = chrono::high_resolution_clock::now();
    gaussianElimination(A, b);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Time taken: " << duration.count() * 1000 << " milliseconds" << endl;

    return 0;
}
