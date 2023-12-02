#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <tbb/tbb.h>

using namespace std;

// Функция для выполнения метода Гаусса с оптимизацией AVX
template <typename T>
void gaussianElimination(vector<vector<T>>& A, vector<T>& b) {
    const int n = A.size();
    for (int i = 0; i < n; ++i) {
        // Находим опорный элемент
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }

        // Обмениваем строки
        swap(A[i], A[pivot]);
        swap(b[i], b[pivot]);

        // Обнуляем элементы ниже опорного
        tbb::parallel_for(tbb::blocked_range<int>(i + 1, n),
                          [&](const tbb::blocked_range<int>& r) {
                              for (int j = r.begin(); j < r.end(); ++j) {
                                  T factor = A[j][i] / A[i][i];
                                  for (int k = i; k < n; ++k) {
                                      A[j][k] -= factor * A[i][k];
                                  }
                                  b[j] -= factor * b[i];
                              }
                          });
    }

    // Обратная подстановка
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= A[i][i];
        A[i][i] = 1;

        tbb::parallel_for(tbb::blocked_range<int>(0, i),
                          [&](const tbb::blocked_range<int>& r) {
                              for (int j = r.begin(); j < r.end(); ++j) {
                                  b[j] -= A[j][i] * b[i];
                                  A[j][i] = 0;
                              }
                          });
    }
}

int main() {
    // Вводим размер системы и зерно для генерации случайных чисел
    int n, seed;
    cout << "Введите размер системы: ";
    cin >> n;
    cout << "Введите зерно для генерации случайных чисел: ";
    cin >> seed;

    // Генерируем случайную матрицу A и вектор b с использованием указанного зерна
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    mt19937 gen(seed); // Генератор случайных чисел Mersenne Twister
    uniform_real_distribution<double> dis(1.0, 10.0);

    for (int i = 0; i < n; ++i) {
        A[i].resize(n); // Изменяем размер каждой строки
        for (int j = 0; j < n; ++j) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }

    // Замеряем время до выполнения метода Гаусса
    auto start = chrono::high_resolution_clock::now();

    // Решаем систему с использованием метода Гаусса
    gaussianElimination(A, b);

    // Замеряем время после выполнения метода Гаусса
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Затраченное время: " << duration.count() * 1000 << " миллисекунд" << endl;

    return 0;
}
