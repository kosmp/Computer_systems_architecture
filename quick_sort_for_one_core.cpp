#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <bitset>

void setAffinity(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

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

// Функция быстрой сортировки
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);

        // Рекурсивные вызовы для левой и правой частей массива
        quickSort(arr, low, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, high);
    }
}

int main() {
    const int dataSize = 100000;
    std::vector<int> data(dataSize);
    
    // Фиксированный seed для воспроизводимости
    std::srand(42);

    // Генерируем одну и ту же последовательность чисел при каждом запуске
    std::generate(data.begin(), data.end(), [] { return std::rand(); });

    auto start = std::chrono::high_resolution_clock::now();

    // Вызываем алгоритм быстрой сортировки на одном ядре
    std::thread sortingThread(quickSort, std::ref(data), 0, dataSize - 1);
    setAffinity(sortingThread, 0);  // Привязываем к первому ядру

    sortingThread.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Выводим время выполнения
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}

