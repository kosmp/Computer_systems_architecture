#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <tbb/tbb.h>

// Функция для разделения массива
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Функция быстрой сортировки с использованием TBB
void parallelQuickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);

        // Используем tbb::parallel_invoke для параллельного выполнения обеих частей
        tbb::parallel_invoke(
            [&arr, low, pivotIndex] { parallelQuickSort(arr, low, pivotIndex - 1); },
            [&arr, pivotIndex, high] { parallelQuickSort(arr, pivotIndex + 1, high); }
        );
    }
}

int main() {
    const int dataSize = 10000;
    std::vector<int> data(dataSize);
    
    // Фиксированный seed для воспроизводимости
    std::srand(42);

    // Генерируем одну и ту же последовательность чисел при каждом запуске
    std::generate(data.begin(), data.end(), [] { return std::rand(); });

    auto start = std::chrono::high_resolution_clock::now();

    // Вызываем алгоритм быстрой сортировки на всех ядрах
    parallelQuickSort(data, 0, dataSize - 1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Выводим время выполнения
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}
