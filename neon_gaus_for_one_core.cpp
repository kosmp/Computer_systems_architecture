#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <tbb/tbb.h>
#include <arm_neon.h>

using namespace std;

// Функция для выполнения метода Гаусса с оптимизацией NEON
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
        for (int j = i + 1; j < n; ++j) {
            T factor = A[j][i] / A[i][i];

            // Используем векторные инструкции NEON
            for (int k = i; k < n; k += 4) {
                if (k + 4 <= n) {
                    float32x4_t aVec = vld1q_f32(&A[j][k]);
                    float32x4_t pivotVec = vld1q_f32(&A[i][k]);
                    float32x4_t factorVec = vdupq_n_f32(factor);
                    float32x4_t result = vmlsq_f32(aVec, pivotVec, factorVec);
                    vst1q_f32(&A[j][k], result);
                } else {
                    for (int l = k; l < n; ++l) {
                        A[j][l] -= factor * A[i][l];
                    }
                }
            }
            b[j] -= factor * b[i];
        }
    }

    // Обратная подстановка
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= A[i][i];
        A[i][i] = 1;

        for (int j = 0; j < i; ++j) {
            b[j] -= A[j][i] * b[i];
            A[j][i] = 0;
        }
    }
}

// Устанавливает привязку к ядру для текущего потока
void setAffinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

int main() {
    // Вводим размер системы и зерно для генерации случайных чисел
    int n, seed;
    cout << "Введите размер системы: ";
    cin >> n;
    cout << "Введите зерно для генерации случайных чисел: ";
    cin >> seed;

    // Генерируем случайную матрицу A и вектор b с использованием указанного зерна
    vector<vector<float>> A(n, vector<float>(n));
    vector<float> b(n);

    mt19937 gen(seed); // Генератор случайных чисел Mersenne Twister
    uniform_real_distribution<float> dis(1.0, 10.0);

    for (int i = 0; i < n; ++i) {
        A[i].resize(n); // Изменяем размер каждой строки
        for (int j = 0; j < n; ++j) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }

    // Замеряем время до выполнения метода Гаусса
    auto start = chrono::high_resolution_clock::now();

    // Решаем систему с использованием метода Гаусса на одном ядре с привязкой
    setAffinity(0); // Привязываем к первому ядру
    gaussianElimination(A, b);

    // Замеряем время после выполнения метода Гаусса
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Затраченное время: " << duration.count() * 1000 << " миллисекунд" << endl;

    return 0;
}
