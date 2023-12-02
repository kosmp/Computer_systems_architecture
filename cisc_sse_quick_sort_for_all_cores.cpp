#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <tbb/tbb.h>
#include <immintrin.h>

// Функция для разделения массива с использованием SSE
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    // Загружаем pivot в SSE регистр
    __m128i pivot_sse = _mm_set1_epi32(pivot);

    for (int j = low; j < high; j += 4) {
        // Загружаем 4 элемента массива в SSE регистр
        __m128i data_sse = _mm_loadu_si128((__m128i*)&arr[j]);

        // Сравниваем элементы массива с pivot
        __m128i compare_mask = _mm_cmplt_epi32(data_sse, pivot_sse);

        // Копируем compare_mask в массив
        int mask[4];
        _mm_storeu_si128((__m128i*)mask, compare_mask);

        // Обмениваем элементы массива, если они меньше pivot
        for (int k = 0; k < 4 && (j + k) < high; ++k) {
            if (mask[k]) {
                i++;
                std::swap(arr[i], arr[j + k]);
            }
        }
    }

    // Обмениваем pivot с элементом, расположенным после всех меньших элементов
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

