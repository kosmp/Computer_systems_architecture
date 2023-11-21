#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <tbb/tbb.h>
#include <arm_neon.h>

void setAffinity(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// Функция разделения массива на две части с использованием Neon
int partition_neon(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    int32x4_t pivot_v = vdupq_n_s32(pivot);

    for (int j = low; j < high; j += 4) {
        int32x4_t arr_v = vld1q_s32(&arr[j]);

        // Сравнение каждого элемента с pivot
        uint32x4_t cmp_mask = vcleq_s32(arr_v, pivot_v);

        // Маска для выбора элементов, меньших или равных pivot
        int32x4_t mask = vandq_s32(cmp_mask, vdupq_n_s32(1));

        // Индексы элементов, меньших или равных pivot
        int32x4_t indices = vmlaq_n_s32(vdupq_n_s32(i + 1), mask, 1);

        // Обновление индекса i
        i += vgetq_lane_s32(indices, 3);

        // Перестановка элементов массива в соответствии с маской
        int32x4_t rearranged = vbslq_s32(cmp_mask, arr_v, vld1q_s32(&arr[i]));

        // Запись обновленных элементов обратно в массив
        vst1q_s32(&arr[i], rearranged);
    }

    // Перестановка pivot на нужную позицию
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Функция быстрой сортировки с использованием TBB и Neon
void singleCoreQuickSort_neon(int* arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition_neon(arr, low, high);

        singleCoreQuickSort_neon(arr, low, pivotIndex - 1);
        singleCoreQuickSort_neon(arr, pivotIndex + 1, high);
    }
}

int main() {
    const int dataSize = 100000000;
    std::vector<int> data(dataSize);

    // Фиксированный seed для воспроизводимости
    std::srand(42);

    // Генерируем одну и ту же последовательность чисел при каждом запуске
    std::generate(data.begin(), data.end(), [] { return std::rand(); });

    // Создаем массив int для Neon оптимизации
    int* data_neon = data.data();

    auto start = std::chrono::high_resolution_clock::now();

    // Вызываем алгоритм быстрой сортировки на всех ядрах с использованием Neon
    singleCoreQuickSort_neon(data_neon, 0, dataSize - 1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Выводим время выполнения
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}

