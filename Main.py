import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import cumtrapz

class ParticleSizeDistribution:
    def __init__(self, mean_size, num_peaks, variance, num_particles=10000, size_range=(0.1, 100)):
        """
        Инициализация параметров распределения
        
        Parameters:
        mean_size (float or list): Средний размер частиц (или список средних для многомодального распределения)
        num_peaks (int): Количество пиков в распределении
        variance (float or list): Дисперсия распределения (или список дисперсий)
        num_particles (int): Количество частиц для генерации
        size_range (tuple): Диапазон размеров частиц (мин, макс) в микронах
        """
        self.mean_size = mean_size
        self.num_peaks = num_peaks
        self.variance = variance
        self.num_particles = num_particles
        self.size_range = size_range
        self.sizes = None
        self.bins = None
        self.hist = None
        self.pdf = None
        self.cdf = None
        
        # Проверка и нормализация входных параметров
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Проверка и нормализация входных параметров"""
        if isinstance(self.mean_size, (list, tuple, np.ndarray)):
            if len(self.mean_size) != self.num_peaks:
                raise ValueError("Длина списка mean_size должна соответствовать num_peaks")
        else:
            self.mean_size = [self.mean_size] * self.num_peaks
            
        if isinstance(self.variance, (list, tuple, np.ndarray)):
            if len(self.variance) != self.num_peaks:
                raise ValueError("Длина списка variance должна соответствовать num_peaks")
        else:
            self.variance = [self.variance] * self.num_peaks
            
        # Преобразуем в numpy массивы
        self.mean_size = np.array(self.mean_size)
        self.variance = np.array(self.variance)
        
    def generate_distribution(self):
        """Генерация многомодального нормального распределения частиц"""
        # Генерируем частицы для каждого пика
        particles_per_peak = self.num_particles // self.num_peaks
        remaining_particles = self.num_particles % self.num_peaks
        
        all_sizes = []
        
        for i in range(self.num_peaks):
            # Количество частиц для этого пика
            n = particles_per_peak + (1 if i < remaining_particles else 0)
            
            # Генерируем частицы с нормальным распределением
            sizes = np.random.normal(self.mean_size[i], np.sqrt(self.variance[i]), n)
            
            # Обрезаем значения по границам диапазона
            sizes = np.clip(sizes, *self.size_range)
            
            all_sizes.append(sizes)
        
        # Объединяем все частицы
        self.sizes = np.concatenate(all_sizes)
        
        # Вычисляем гистограмму
        self.bins = np.linspace(self.size_range[0], self.size_range[1], 100)
        self.hist, _ = np.histogram(self.sizes, bins=self.bins, density=True)
        
        # Вычисляем PDF (плотность распределения)
        self.pdf = self.hist / np.sum(self.hist)
        
        # Вычисляем CDF (интегральное распределение)
        self.cdf = cumtrapz(self.pdf, self.bins[:-1], initial=0)
        
    def calculate_statistics(self):
        """Вычисление статистических характеристик распределения"""
        stats = {}
        
        # Основные статистики
        stats['mean'] = np.mean(self.sizes)
        stats['median'] = np.median(self.sizes)
        stats['std'] = np.std(self.sizes)
        stats['variance'] = np.var(self.sizes)
        stats['min'] = np.min(self.sizes)
        stats['max'] = np.max(self.sizes)
        
        # Различные средние диаметры
        stats['D10'] = np.mean(self.sizes)  # Средний арифметический
        stats['D20'] = np.sqrt(np.mean(self.sizes**2))  # Средний квадратичный
        stats['D30'] = np.power(np.mean(self.sizes**3), 1/3)  # Средний кубический
        stats['D32'] = np.mean(self.sizes**3) / np.mean(self.sizes**2)  # Sauter mean diameter
        stats['D43'] = np.mean(self.sizes**4) / np.mean(self.sizes**3)  # De Brouckere mean diameter
        stats['D21'] = np.mean(self.sizes**2) / np.mean(self.sizes)
        stats['D31'] = np.mean(self.sizes**3) / np.mean(self.sizes)
        stats['D23'] = np.mean(self.sizes**2) / np.mean(self.sizes**3)
        
        return stats
    
    def plot_distributions(self):
        """Построение графиков распределения"""
        if self.sizes is None:
            self.generate_distribution()
            
        stats = self.calculate_statistics()
        
        # Создаем фигуру с двумя графиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Дифференциальное распределение (PDF)
        ax1.plot(self.bins[:-1], self.pdf, 'b-', linewidth=2)
        ax1.set_title('Дифференциальное распределение (PDF)')
        ax1.set_xlabel('Размер частиц (мкм)')
        ax1.set_ylabel('Плотность вероятности')
        ax1.set_xscale('log')
        ax1.grid(True)
        
        # Интегральное распределение (CDF)
        ax2.plot(self.bins[:-1], self.cdf, 'r-', linewidth=2)
        ax2.set_title('Интегральное распределение (CDF)')
        ax2.set_xlabel('Размер частиц (мкм)')
        ax2.set_xscale('log')
        ax2.set_ylabel('Кумулятивная вероятность')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Выводим статистические данные
        print("\nСтатистические характеристики распределения:")
        print(f"Средний размер (D10): {stats['mean']:.2f} мкм")
        print(f"Медиана: {stats['median']:.2f} мкм")
        print(f"Стандартное отклонение: {stats['std']:.2f} мкм")
        print(f"Минимальный размер: {stats['min']:.2f} мкм")
        print(f"Максимальный размер: {stats['max']:.2f} мкм")
        print("\nСпециальные средние диаметры:")
        print(f"D32 (Sauter mean diameter): {stats['D32']:.2f} мкм")
        print(f"D43 (De Brouckere mean diameter): {stats['D43']:.2f} мкм")
        print(f"D21: {stats['D21']:.2f} мкм")
        print(f"D31: {stats['D31']:.2f} мкм")
        print(f"D23: {stats['D23']:.2f} мкм")

# Пример использования
if __name__ == "__main__":
    # Параметры распределения
    mean_size = 10  # Средний размер (может быть списком для многомодального распределения)
    num_peaks = 2   # Количество пиков
    variance = 10    # Дисперсия (может быть списком)
    
    # Создаем и анализируем распределение
    psd = ParticleSizeDistribution(mean_size=[8, 12], num_peaks=2, variance=[2, 3])
    psd.generate_distribution()
    psd.plot_distributions()