#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>
#include <thread>
#include <pthread.h>

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
        for (int j = i + 1; j < n; ++j) {
            T factor = A[j][i] / A[i][i];
            __m256d factorVec = _mm256_set1_pd(factor);
            for (int k = i; k < n; k += 4) {
                if (k + 4 <= n) {
                    __m256d aVec = _mm256_loadu_pd(&A[j][k]);
                    __m256d pivotVec = _mm256_loadu_pd(&A[i][k]);
                    __m256d result = _mm256_sub_pd(aVec, _mm256_mul_pd(factorVec, pivotVec));
                    _mm256_storeu_pd(&A[j][k], result);
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

void setAffinity(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
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

    // Создаем поток и привязываем его к первому ядру
    std::thread gaussThread(gaussianElimination<double>, std::ref(A), std::ref(b));
    setAffinity(gaussThread, 0);

    // Ждем завершения потока
    gaussThread.join();

    // Замеряем время после выполнения метода Гаусса
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Затраченное время: " << duration.count() * 1000 << " миллисекунд" << endl;

    return 0;
}
