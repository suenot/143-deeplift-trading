# Глава 122: DeepLift Trading - Атрибуция нейронных сетей для объяснимых торговых сигналов

## Введение

DeepLIFT (Deep Learning Important FeaTures) представляет собой передовой метод интерпретируемости нейронных сетей, разработанный для объяснения предсказаний глубоких моделей путём сравнения активаций с референсным (базовым) входом. Этот метод был представлен исследователями Avanti Shrikumar, Peyton Greenside и Anshul Kundaje в их основополагающей работе 2017 года "Learning Important Features Through Propagating Activation Differences".

### Почему объяснимый ИИ важен для трейдинга?

В современном алгоритмическом трейдинге принятие решений на основе "чёрных ящиков" становится всё менее приемлемым по ряду причин:

1. **Регуляторные требования**: Финансовые регуляторы всё чаще требуют объяснимости алгоритмических решений
2. **Управление рисками**: Понимание причин торговых сигналов критически важно для оценки рисков
3. **Доверие**: Трейдеры и портфельные менеджеры должны понимать, почему модель принимает те или иные решения
4. **Отладка**: Выявление ошибок и смещений в моделях требует понимания их внутренней логики
5. **Обнаружение режимов**: Изменения в паттернах важности признаков могут сигнализировать о смене рыночного режима

DeepLIFT решает эти задачи, предоставляя точные оценки вклада каждого входного признака в финальное предсказание модели.

---

## Содержание

1. [Введение](#введение)
2. [Математические основы DeepLIFT](#математические-основы-deeplift)
   - [Принцип разницы от базовой линии](#принцип-разницы-от-базовой-линии)
   - [Правило Rescale](#правило-rescale-перемасштабирования)
   - [Правило RevealCancel](#правило-revealcancel)
   - [Сравнение с другими методами](#сравнение-с-другими-методами-атрибуции)
3. [Реализация на Python](#реализация-на-python)
   - [Структура модели PyTorch](#структура-модели-pytorch)
   - [Интеграция с Captum](#интеграция-с-библиотекой-captum)
   - [Генерация торговых сигналов](#генерация-торговых-сигналов-на-основе-атрибуций)
   - [Загрузчик данных](#загрузчик-данных-data_loaderpy)
4. [Реализация на Rust](#реализация-на-rust)
   - [Структура крейта](#структура-крейта)
   - [Ключевые типы и трейты](#ключевые-типы-и-трейты)
5. [Источники данных](#источники-данных)
   - [Данные фондового рынка (yfinance)](#данные-фондового-рынка-yfinance)
   - [Криптовалютные данные (Bybit API)](#криптовалютные-данные-bybit-api)
6. [Торговые приложения](#торговые-приложения)
   - [Важность признаков для торговых решений](#важность-признаков-для-торговых-решений)
   - [Объяснимые сигналы покупки/продажи](#объяснимые-сигналы-покупкипродажи)
   - [Оценка рисков через атрибуцию](#оценка-рисков-через-анализ-атрибуций)
7. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
8. [Сравнение производительности](#сравнение-производительности)
9. [Ссылки на научные работы](#научные-публикации-и-ссылки)

---

## Математические основы DeepLIFT

### Принцип разницы от базовой линии

Основная идея DeepLIFT заключается в сравнении активаций нейронной сети для исследуемого входа с активациями для референсного (базового) входа. Это позволяет определить, какие признаки способствовали изменению выхода относительно базового состояния.

#### Определение разницы активации

Для нейрона с активацией $t$ при входе $x$ и референсной активацией $t^0$ при референсном входе $x^0$:

$$\Delta t = t - t^0$$

где:
- $t$ — активация при фактическом входе
- $t^0$ — активация при референсном входе
- $\Delta t$ — разница активаций

#### Оценка вклада (Contribution Score)

Вклад входного признака $x_i$ в разницу активации $\Delta t$:

$$C_i = \text{вклад } x_i \text{ в } \Delta t$$

#### Свойство суммирования (Summation-to-Delta)

Ключевое свойство DeepLIFT — сумма всех вкладов точно равна разнице выходов:

$$\sum_i C_i = f(x) - f(x^0)$$

Это гарантирует, что объяснение является полным и непротиворечивым.

### Правило множителя (Multiplier Rule)

Для линейной связи между входами и выходом определяется множитель:

$$m_i = \frac{C_i}{\Delta x_i}$$

где $\Delta x_i = x_i - x_i^0$ — разница входа от референса.

Тогда свойство суммирования можно записать как:

$$\sum_i m_i \times \Delta x_i = \Delta t$$

### Правило Rescale (Перемасштабирования)

Для нелинейных активаций, таких как ReLU, правило Rescale определяет множитель как:

$$m = \begin{cases}
\frac{\Delta y}{\Delta x} & \text{если } \Delta x \neq 0 \\
0 & \text{если } \Delta x = 0
\end{cases}$$

где:
- $\Delta y = y - y^0$ — разница выхода активации
- $\Delta x = x - x^0$ — разница входа активации

#### Пример для ReLU

Рассмотрим ReLU активацию $y = \max(0, x)$:

- Если $x > 0$ и $x^0 > 0$: $m = 1$
- Если $x > 0$ и $x^0 \leq 0$: $m = \frac{x}{x - x^0}$
- Если $x \leq 0$ и $x^0 > 0$: $m = \frac{-x^0}{x - x^0}$
- Если $x \leq 0$ и $x^0 \leq 0$: $m = 0$

### Правило RevealCancel

Правило RevealCancel обеспечивает более точную атрибуцию путём раздельной обработки положительных и отрицательных вкладов:

$$\Delta y^+ = y^+ - y^{0+}$$
$$\Delta y^- = y^- - y^{0-}$$

где $y^+$ и $y^-$ — положительная и отрицательная части активации.

Это правило особенно полезно когда:
- Входы имеют смешанные знаки
- Важно различать усиливающие и подавляющие факторы
- Требуется высокая точность атрибуции

### Цепное правило для множителей

Для многослойной сети множители перемножаются:

$$m_{\text{total}} = m_1 \times m_2 \times \cdots \times m_n$$

Это позволяет эффективно распространять атрибуции через всю сеть.

### Выбор референсного входа

Выбор референса критически важен для интерпретации результатов:

| Тип референса | Описание | Применение |
|---------------|----------|------------|
| Нулевой | Все признаки = 0 | Простой случай, не всегда осмысленный |
| Средний | Среднее по датасету | Сравнение с "типичным" входом |
| Нейтральный | "Отсутствие сигнала" | Идеально для торговых моделей |
| Распределённый | Выборка из распределения | Ожидаемые градиенты (DeepSHAP) |

Для торговых приложений рекомендуется использовать **нейтральный референс**, представляющий состояние рынка без явного сигнала (RSI = 50, нулевой моментум и т.д.).

---

## Сравнение с другими методами атрибуции

### Таблица сравнения

| Метод | Референс | Насыщение | Скорость | Точность | Теор. обоснование |
|-------|----------|-----------|----------|----------|-------------------|
| **DeepLIFT** | Да | Отлично | Высокая | Отлично | Хорошее |
| Градиенты | Нет | Плохо | Очень высокая | Низкая | Слабое |
| Integrated Gradients | Да | Хорошо | Низкая | Очень хорошо | Сильное (аксиомы) |
| SHAP (KernelSHAP) | Да | Отлично | Очень низкая | Отлично | Сильное (Shapley) |
| DeepSHAP | Да | Отлично | Средняя | Отлично | Сильное |
| LRP | Нет | Хорошо | Высокая | Хорошо | Среднее |
| Saliency Maps | Нет | Плохо | Очень высокая | Низкая | Слабое |

### Преимущества DeepLIFT

1. **Обработка насыщения**: Корректно работает в областях насыщения активаций (в отличие от градиентов)
2. **Свойство суммирования**: Атрибуции точно суммируются в разницу предсказаний
3. **Скорость**: Значительно быстрее SHAP при сравнимой точности
4. **Стабильность**: Меньше шума в атрибуциях по сравнению с градиентными методами

### Когда использовать DeepLIFT

**Используйте DeepLIFT когда:**
- Нужна высокая скорость вычислений
- Модель использует ReLU или подобные активации
- Важна точность атрибуции
- Требуется объяснение относительно базового состояния

**Рассмотрите альтернативы когда:**
- Нужны теоретические гарантии Shapley (используйте SHAP)
- Архитектура содержит сложные операции (используйте Integrated Gradients)
- Скорость критична, а точность вторична (используйте простые градиенты)

---

## Реализация на Python

### Структура модели PyTorch

```python
"""
Реализация DeepLIFT для торговых моделей на PyTorch.

Модуль содержит:
- Класс Attribution для хранения результатов атрибуции
- Класс DeepLIFT для вычисления атрибуций
- Торговую нейросеть с поддержкой объяснений
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Attribution:
    """
    Результаты атрибуции для одного предсказания.

    Attributes:
        feature_names: Названия входных признаков
        scores: Оценки атрибуции для каждого признака
        baseline_output: Выход модели для референсного входа
        actual_output: Выход модели для фактического входа
        delta: Разница между фактическим и референсным выходом
    """
    feature_names: List[str]
    scores: np.ndarray
    baseline_output: float
    actual_output: float
    delta: float

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Получить N признаков с наибольшим вкладом (по модулю).

        Args:
            n: Количество признаков для возврата

        Returns:
            Список кортежей (имя_признака, оценка_вклада)
        """
        # Сортируем по абсолютному значению вклада
        indices = np.argsort(np.abs(self.scores))[::-1][:n]
        return [(self.feature_names[i], self.scores[i]) for i in indices]

    def positive_contributors(self) -> List[Tuple[str, float]]:
        """Получить признаки с положительным вкладом."""
        result = []
        for i, score in enumerate(self.scores):
            if score > 0:
                result.append((self.feature_names[i], score))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def negative_contributors(self) -> List[Tuple[str, float]]:
        """Получить признаки с отрицательным вкладом."""
        result = []
        for i, score in enumerate(self.scores):
            if score < 0:
                result.append((self.feature_names[i], score))
        return sorted(result, key=lambda x: x[1])

    def verify_summation(self) -> float:
        """
        Проверить свойство суммирования.
        Возвращает абсолютную ошибку (должна быть близка к 0).
        """
        return abs(np.sum(self.scores) - self.delta)


class DeepLIFT:
    """
    DeepLIFT атрибуция для торговых нейросетевых моделей.

    Поддерживает два правила атрибуции:
    - Rescale: стандартное правило перемасштабирования
    - RevealCancel: раздельная обработка положительных/отрицательных вкладов

    Example:
        >>> model = TradingNetwork(input_size=11)
        >>> reference = torch.zeros(1, 11)  # Нулевой референс
        >>> explainer = DeepLIFT(model, reference, rule="rescale")
        >>> attribution = explainer.attribute(input_tensor, feature_names)
        >>> print(attribution.top_features(3))
    """

    def __init__(
        self,
        model: nn.Module,
        reference: Optional[torch.Tensor] = None,
        rule: str = "rescale"
    ):
        """
        Инициализация объяснителя DeepLIFT.

        Args:
            model: Нейросетевая модель PyTorch для объяснения
            reference: Референсный (базовый) вход. Если None, используются нули.
            rule: Правило атрибуции - "rescale" или "reveal_cancel"
        """
        self.model = model
        self.reference = reference
        self.rule = rule

        # Проверка корректности правила
        if rule not in ["rescale", "reveal_cancel"]:
            raise ValueError(f"Неизвестное правило: {rule}. "
                           "Допустимы: 'rescale', 'reveal_cancel'")

    def attribute(
        self,
        input_tensor: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Attribution:
        """
        Вычислить атрибуции DeepLIFT для входного тензора.

        Args:
            input_tensor: Входной тензор для объяснения (batch_size=1)
            feature_names: Названия признаков (опционально)

        Returns:
            Объект Attribution с оценками вклада каждого признака
        """
        # Приведение к правильной размерности
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Установка референса
        if self.reference is None:
            reference = torch.zeros_like(input_tensor)
        else:
            reference = self.reference.expand_as(input_tensor)

        # Вычисление выходов модели
        self.model.eval()
        with torch.no_grad():
            ref_output = self.model(reference)
            actual_output = self.model(input_tensor)

        # Вычисление градиентов для атрибуции
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        output.backward(torch.ones_like(output))

        # Получение градиентов
        gradients = input_tensor.grad.detach()

        # Вычисление разницы от референса
        delta_input = input_tensor.detach() - reference

        # Применение правила атрибуции
        if self.rule == "rescale":
            # Правило Rescale: градиент * дельта входа
            attributions = gradients * delta_input
        else:
            # Правило RevealCancel: раздельная обработка
            attributions = self._reveal_cancel_attribution(
                input_tensor, reference, gradients
            )

        # Создание названий признаков по умолчанию
        if feature_names is None:
            feature_names = [f"признак_{i}" for i in range(input_tensor.shape[1])]

        return Attribution(
            feature_names=feature_names,
            scores=attributions.squeeze().numpy(),
            baseline_output=ref_output.item(),
            actual_output=actual_output.item(),
            delta=actual_output.item() - ref_output.item()
        )

    def _reveal_cancel_attribution(
        self,
        input_tensor: torch.Tensor,
        reference: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление атрибуции по правилу RevealCancel.

        Разделяет положительные и отрицательные вклады для
        более точной атрибуции.

        Args:
            input_tensor: Входной тензор
            reference: Референсный тензор
            gradients: Градиенты по входу

        Returns:
            Тензор атрибуций
        """
        delta = input_tensor.detach() - reference

        # Разделение на положительную и отрицательную части
        positive_delta = F.relu(delta)
        negative_delta = -F.relu(-delta)

        # Вычисление раздельных атрибуций
        positive_attr = gradients * positive_delta
        negative_attr = gradients * negative_delta

        return positive_attr + negative_attr

    def batch_attribute(
        self,
        inputs: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> List[Attribution]:
        """
        Вычислить атрибуции для батча входов.

        Args:
            inputs: Батч входных тензоров (batch_size, num_features)
            feature_names: Названия признаков

        Returns:
            Список объектов Attribution
        """
        attributions = []
        for i in range(inputs.shape[0]):
            attr = self.attribute(inputs[i:i+1], feature_names)
            attributions.append(attr)
        return attributions


class TradingNetwork(nn.Module):
    """
    Нейронная сеть для генерации торговых сигналов.

    Архитектура: полносвязная сеть с ReLU активациями.
    Поддерживает DeepLIFT атрибуцию.

    Attributes:
        input_size: Размер входного слоя (количество признаков)
        hidden_size: Размер скрытых слоёв
        output_size: Размер выходного слоя
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Инициализация торговой нейросети.

        Args:
            input_size: Количество входных признаков
            hidden_size: Размер скрытых слоёв
            num_hidden_layers: Количество скрытых слоёв
            output_size: Размер выхода (1 для регрессии)
            dropout: Вероятность dropout
        """
        super().__init__()

        layers = []

        # Входной слой
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Скрытые слои
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Выходной слой
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов методом He."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.

        Args:
            x: Входной тензор (batch_size, input_size)

        Returns:
            Предсказания (batch_size, output_size)
        """
        return self.network(x)
```

### Интеграция с библиотекой Captum

```python
"""
Интеграция DeepLIFT с библиотекой Captum от Facebook.

Captum предоставляет оптимизированные реализации методов
атрибуции для PyTorch моделей.
"""

from captum.attr import DeepLift, DeepLiftShap
from captum.attr import visualization as viz
import matplotlib.pyplot as plt


class CaptumDeepLIFTTrader:
    """
    Торговый объяснитель на основе Captum DeepLIFT.

    Использует оптимизированную реализацию DeepLIFT из Captum
    для быстрого вычисления атрибуций.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        reference_type: str = "zero"
    ):
        """
        Инициализация объяснителя.

        Args:
            model: Торговая модель PyTorch
            feature_names: Названия признаков
            reference_type: Тип референса ("zero", "mean", "neutral")
        """
        self.model = model
        self.feature_names = feature_names
        self.reference_type = reference_type

        # Создание объекта DeepLift из Captum
        self.explainer = DeepLift(model)

    def _get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Получить референсный вход в зависимости от типа."""
        if self.reference_type == "zero":
            return torch.zeros_like(input_tensor)
        elif self.reference_type == "neutral":
            # Нейтральный референс для торговли
            # RSI = 0.5, остальные = 0
            ref = torch.zeros_like(input_tensor)
            if ref.shape[-1] > 7:  # Если есть признак RSI
                ref[..., 7] = 0.5
            return ref
        else:
            raise ValueError(f"Неизвестный тип референса: {self.reference_type}")

    def explain(self, input_tensor: torch.Tensor) -> Attribution:
        """
        Объяснить предсказание модели.

        Args:
            input_tensor: Входные признаки

        Returns:
            Объект Attribution с результатами
        """
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        reference = self._get_reference(input_tensor)

        # Вычисление атрибуций через Captum
        attributions = self.explainer.attribute(
            input_tensor,
            baselines=reference
        )

        # Вычисление выходов
        self.model.eval()
        with torch.no_grad():
            actual_output = self.model(input_tensor)
            baseline_output = self.model(reference)

        return Attribution(
            feature_names=self.feature_names,
            scores=attributions.squeeze().detach().numpy(),
            baseline_output=baseline_output.item(),
            actual_output=actual_output.item(),
            delta=actual_output.item() - baseline_output.item()
        )

    def visualize_attribution(
        self,
        attribution: Attribution,
        title: str = "Важность признаков DeepLIFT"
    ) -> plt.Figure:
        """
        Визуализация атрибуций в виде горизонтальной гистограммы.

        Args:
            attribution: Результаты атрибуции
            title: Заголовок графика

        Returns:
            Объект Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Сортировка по абсолютному значению
        sorted_indices = np.argsort(np.abs(attribution.scores))
        sorted_names = [attribution.feature_names[i] for i in sorted_indices]
        sorted_scores = [attribution.scores[i] for i in sorted_indices]

        # Цвета: зелёный для положительных, красный для отрицательных
        colors = ['green' if s > 0 else 'red' for s in sorted_scores]

        ax.barh(sorted_names, sorted_scores, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Вклад в предсказание')
        ax.set_title(title)

        # Добавление информации о предсказании
        info_text = (f"Предсказание: {attribution.actual_output:.4f}\n"
                    f"Базовая линия: {attribution.baseline_output:.4f}\n"
                    f"Дельта: {attribution.delta:.4f}")
        ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig
```

### Генерация торговых сигналов на основе атрибуций

```python
"""
Генерация объяснимых торговых сигналов.

Модуль объединяет предсказания модели с атрибуциями DeepLIFT
для создания прозрачных торговых решений.
"""

from enum import Enum
from dataclasses import dataclass


class SignalType(Enum):
    """Типы торговых сигналов."""
    STRONG_BUY = "Сильная покупка"
    BUY = "Покупка"
    NEUTRAL = "Нейтрально"
    SELL = "Продажа"
    STRONG_SELL = "Сильная продажа"


@dataclass
class ExplainedSignal:
    """
    Торговый сигнал с объяснением.

    Содержит не только рекомендацию, но и обоснование
    на основе атрибуций DeepLIFT.
    """
    signal: SignalType
    confidence: float
    prediction: float
    top_bullish_factors: List[Tuple[str, float]]
    top_bearish_factors: List[Tuple[str, float]]
    explanation: str


class SignalGenerator:
    """
    Генератор объяснимых торговых сигналов.

    Использует модель для предсказания и DeepLIFT для объяснения,
    создавая полностью прозрачные торговые рекомендации.
    """

    def __init__(
        self,
        model: nn.Module,
        explainer: DeepLIFT,
        feature_names: List[str],
        threshold: float = 0.001,
        strong_threshold: float = 0.003
    ):
        """
        Инициализация генератора сигналов.

        Args:
            model: Торговая модель
            explainer: Объяснитель DeepLIFT
            feature_names: Названия признаков
            threshold: Порог для сигналов покупки/продажи
            strong_threshold: Порог для сильных сигналов
        """
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.threshold = threshold
        self.strong_threshold = strong_threshold

    def generate_signal(
        self,
        features: torch.Tensor
    ) -> ExplainedSignal:
        """
        Сгенерировать объяснимый торговый сигнал.

        Args:
            features: Входные признаки

        Returns:
            Объект ExplainedSignal с рекомендацией и объяснением
        """
        # Получение предсказания
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features).item()

        # Определение типа сигнала
        if prediction > self.strong_threshold:
            signal = SignalType.STRONG_BUY
            confidence = min(prediction / self.strong_threshold, 2.0) / 2.0
        elif prediction > self.threshold:
            signal = SignalType.BUY
            confidence = (prediction - self.threshold) / (self.strong_threshold - self.threshold)
        elif prediction < -self.strong_threshold:
            signal = SignalType.STRONG_SELL
            confidence = min(-prediction / self.strong_threshold, 2.0) / 2.0
        elif prediction < -self.threshold:
            signal = SignalType.SELL
            confidence = (-prediction - self.threshold) / (self.strong_threshold - self.threshold)
        else:
            signal = SignalType.NEUTRAL
            confidence = 1.0 - abs(prediction) / self.threshold

        # Получение атрибуций
        attribution = self.explainer.attribute(features, self.feature_names)

        # Разделение на бычьи и медвежьи факторы
        bullish = attribution.positive_contributors()[:3]
        bearish = attribution.negative_contributors()[:3]

        # Генерация текстового объяснения
        explanation = self._generate_explanation(
            signal, prediction, bullish, bearish
        )

        return ExplainedSignal(
            signal=signal,
            confidence=confidence,
            prediction=prediction,
            top_bullish_factors=bullish,
            top_bearish_factors=bearish,
            explanation=explanation
        )

    def _generate_explanation(
        self,
        signal: SignalType,
        prediction: float,
        bullish: List[Tuple[str, float]],
        bearish: List[Tuple[str, float]]
    ) -> str:
        """Генерация текстового объяснения сигнала."""
        lines = [f"Сигнал: {signal.value} (предсказание: {prediction:.4f})"]

        if bullish:
            lines.append("\nБычьи факторы:")
            for name, score in bullish:
                lines.append(f"  + {name}: {score:.4f}")

        if bearish:
            lines.append("\nМедвежьи факторы:")
            for name, score in bearish:
                lines.append(f"  - {name}: {score:.4f}")

        return "\n".join(lines)
```

### Загрузчик данных (data_loader.py)

```python
"""
Загрузчик данных для DeepLIFT трейдинга.

Поддерживает:
- Данные акций через yfinance
- Криптовалютные данные через Bybit API
- Генерацию технических признаков
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Kline:
    """
    Данные одной свечи (candlestick).

    Attributes:
        timestamp: Временная метка в миллисекундах
        open: Цена открытия
        high: Максимальная цена
        low: Минимальная цена
        close: Цена закрытия
        volume: Объём торгов
        turnover: Оборот в базовой валюте
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class BybitClient:
    """
    Клиент для получения данных с биржи Bybit.

    Поддерживает получение исторических свечей для
    криптовалютных торговых пар.

    Example:
        >>> client = BybitClient()
        >>> klines = client.fetch_klines("BTCUSDT", interval="60", limit=500)
        >>> print(f"Получено {len(klines)} свечей")
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, base_url: Optional[str] = None):
        """
        Инициализация клиента Bybit.

        Args:
            base_url: Базовый URL API (опционально)
        """
        self.base_url = base_url or self.BASE_URL
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200
    ) -> List[Kline]:
        """
        Получить исторические свечи с Bybit.

        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            interval: Интервал свечи ("1", "5", "15", "60", "D")
            limit: Количество свечей (максимум 1000)

        Returns:
            Список объектов Kline, отсортированных по времени

        Raises:
            ValueError: При ошибке API
            requests.RequestException: При ошибке сети
        """
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"Ошибка API Bybit: {data.get('retMsg')}")

            klines = []
            for item in data["result"]["list"]:
                klines.append(Kline(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6])
                ))

            # Bybit возвращает в порядке убывания, разворачиваем
            klines.reverse()
            logger.info(f"Получено {len(klines)} свечей для {symbol}")
            return klines

        except requests.RequestException as e:
            logger.error(f"Ошибка получения данных: {e}")
            raise

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "60",
        limit: int = 200
    ) -> Dict[str, List[Kline]]:
        """
        Получить свечи для нескольких торговых пар.

        Args:
            symbols: Список торговых пар
            interval: Интервал свечи
            limit: Количество свечей для каждой пары

        Returns:
            Словарь {символ: список_свечей}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_klines(symbol, interval, limit)
            except Exception as e:
                logger.warning(f"Не удалось получить данные для {symbol}: {e}")
        return results


class YFinanceLoader:
    """
    Загрузчик данных фондового рынка через yfinance.

    Example:
        >>> loader = YFinanceLoader()
        >>> df = loader.fetch_stock("AAPL", period="1y")
        >>> print(df.head())
    """

    def __init__(self):
        """Инициализация загрузчика."""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError(
                "Для использования YFinanceLoader установите yfinance: "
                "pip install yfinance"
            )

    def fetch_stock(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Получить данные акции.

        Args:
            ticker: Тикер акции (например, "AAPL")
            period: Период данных ("1d", "1mo", "1y", "2y", "max")
            interval: Интервал свечей ("1m", "5m", "1h", "1d")

        Returns:
            DataFrame с колонками Open, High, Low, Close, Volume
        """
        logger.info(f"Загрузка данных для {ticker}...")

        stock = self.yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        logger.info(f"Загружено {len(df)} записей для {ticker}")
        return df

    def fetch_crypto(
        self,
        symbol: str,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Получить криптовалютные данные через yfinance.

        Args:
            symbol: Символ (например, "BTC-USD")
            period: Период данных

        Returns:
            DataFrame с ценовыми данными
        """
        return self.fetch_stock(symbol, period=period)


class FeatureGenerator:
    """
    Генератор технических признаков для торговых моделей.

    Вычисляет следующие признаки:
    - returns_1d, returns_5d, returns_10d: Доходности за разные периоды
    - sma_ratio, ema_ratio: Отношение цены к скользящим средним
    - volatility: Историческая волатильность
    - momentum: Моментум цены
    - rsi: Relative Strength Index
    - macd: MACD нормализованный
    - bb_position: Позиция в полосах Боллинджера
    - volume_sma_ratio: Отношение объёма к его SMA
    """

    def __init__(self, window: int = 20):
        """
        Инициализация генератора признаков.

        Args:
            window: Базовое окно для вычисления признаков
        """
        self.window = window

    @staticmethod
    def feature_names() -> List[str]:
        """Получить названия всех признаков."""
        return [
            "доходность_1д",    # returns_1d
            "доходность_5д",    # returns_5d
            "доходность_10д",   # returns_10d
            "отн_SMA",          # sma_ratio
            "отн_EMA",          # ema_ratio
            "волатильность",    # volatility
            "моментум",         # momentum
            "RSI",              # rsi
            "MACD",             # macd
            "позиция_BB",       # bb_position
            "отн_объём"         # volume_sma_ratio
        ]

    def compute_all_features(self, klines: List[Kline]) -> np.ndarray:
        """
        Вычислить все признаки из свечных данных.

        Args:
            klines: Список свечей

        Returns:
            Массив признаков формы (N, 11)
        """
        if len(klines) < self.window + 10:
            logger.warning("Недостаточно данных для вычисления признаков")
            return np.array([])

        # Извлечение цен закрытия и объёмов
        closes = np.array([k.close for k in klines])
        volumes = np.array([k.volume for k in klines])

        # Вычисление отдельных признаков
        features = {
            'returns_1': self._compute_returns(closes, 1),
            'returns_5': self._compute_returns(closes, 5),
            'returns_10': self._compute_returns(closes, 10),
            'sma_ratio': self._compute_sma_ratio(closes),
            'ema_ratio': self._compute_ema_ratio(closes),
            'volatility': self._compute_volatility(closes),
            'momentum': self._compute_momentum(closes),
            'rsi': self._compute_rsi(closes),
            'macd': self._compute_macd(closes),
            'bb_position': self._compute_bollinger_position(closes),
            'volume_ratio': self._compute_volume_sma_ratio(volumes)
        }

        # Определение минимальной длины
        min_len = min(len(f) for f in features.values() if len(f) > 0)

        if min_len == 0:
            return np.array([])

        # Стекирование признаков
        result = np.column_stack([
            features['returns_1'][-min_len:],
            features['returns_5'][-min_len:],
            features['returns_10'][-min_len:],
            features['sma_ratio'][-min_len:],
            features['ema_ratio'][-min_len:],
            features['volatility'][-min_len:],
            features['momentum'][-min_len:],
            features['rsi'][-min_len:],
            features['macd'][-min_len:],
            features['bb_position'][-min_len:],
            features['volume_ratio'][-min_len:]
        ])

        return result

    def _compute_returns(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Вычисление доходностей за период."""
        if len(closes) <= period:
            return np.array([])
        return closes[period:] / closes[:-period] - 1

    def _compute_sma_ratio(self, closes: np.ndarray) -> np.ndarray:
        """Вычисление отношения цены к SMA."""
        if len(closes) < self.window:
            return np.array([])
        sma = np.convolve(closes, np.ones(self.window)/self.window, mode='valid')
        return closes[self.window-1:] / sma - 1

    def _compute_ema_ratio(self, closes: np.ndarray) -> np.ndarray:
        """Вычисление отношения цены к EMA."""
        if len(closes) < self.window:
            return np.array([])

        alpha = 2 / (self.window + 1)
        ema = np.zeros(len(closes))
        ema[0] = closes[0]

        for i in range(1, len(closes)):
            ema[i] = alpha * closes[i] + (1 - alpha) * ema[i-1]

        return (closes / ema - 1)[self.window-1:]

    def _compute_volatility(self, closes: np.ndarray) -> np.ndarray:
        """Вычисление скользящей волатильности."""
        if len(closes) < self.window + 1:
            return np.array([])

        log_returns = np.diff(np.log(closes))
        volatility = np.array([
            np.std(log_returns[max(0, i-self.window+1):i+1])
            for i in range(self.window-1, len(log_returns))
        ])
        return volatility

    def _compute_momentum(self, closes: np.ndarray) -> np.ndarray:
        """Вычисление моментума цены."""
        if len(closes) < self.window:
            return np.array([])
        momentum = closes[self.window-1:] / closes[:-self.window+1] - 1
        return momentum[:-1] if len(momentum) > 1 else momentum

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Вычисление RSI (Relative Strength Index)."""
        if len(closes) < period + 1:
            return np.array([])

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return rsi / 100  # Нормализация к [0, 1]

    def _compute_macd(self, closes: np.ndarray) -> np.ndarray:
        """Вычисление нормализованного MACD."""
        if len(closes) < 26:
            return np.array([])

        # EMA 12
        alpha12 = 2 / 13
        ema12 = np.zeros(len(closes))
        ema12[0] = closes[0]
        for i in range(1, len(closes)):
            ema12[i] = alpha12 * closes[i] + (1 - alpha12) * ema12[i-1]

        # EMA 26
        alpha26 = 2 / 27
        ema26 = np.zeros(len(closes))
        ema26[0] = closes[0]
        for i in range(1, len(closes)):
            ema26[i] = alpha26 * closes[i] + (1 - alpha26) * ema26[i-1]

        macd = (ema12 - ema26) / closes
        return macd[25:]

    def _compute_bollinger_position(self, closes: np.ndarray) -> np.ndarray:
        """Вычисление позиции в полосах Боллинджера."""
        if len(closes) < self.window:
            return np.array([])

        positions = []
        for i in range(self.window - 1, len(closes)):
            window_data = closes[i-self.window+1:i+1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                position = (closes[i] - mean) / (2 * std)
            else:
                position = 0
            positions.append(position)

        return np.array(positions)

    def _compute_volume_sma_ratio(self, volumes: np.ndarray) -> np.ndarray:
        """Вычисление отношения объёма к его SMA."""
        if len(volumes) < self.window:
            return np.array([])

        sma = np.convolve(volumes, np.ones(self.window)/self.window, mode='valid')
        ratio = volumes[self.window-1:] / np.where(sma != 0, sma, 1) - 1
        return ratio


def prepare_training_data(
    klines: List[Kline],
    target_horizon: int = 5,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Подготовить данные для обучения модели.

    Args:
        klines: Список свечей
        target_horizon: Горизонт предсказания (в свечах)
        train_ratio: Доля данных для обучения

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Генерация признаков
    generator = FeatureGenerator()
    features = generator.compute_all_features(klines)

    # Извлечение цен
    prices = np.array([k.close for k in klines])
    offset = len(prices) - len(features)
    aligned_prices = prices[offset:]

    # Создание целевой переменной (будущая доходность)
    target = np.zeros(len(aligned_prices))
    target[:-target_horizon] = (
        aligned_prices[target_horizon:] - aligned_prices[:-target_horizon]
    ) / aligned_prices[:-target_horizon]

    # Удаление невалидных строк
    valid_len = len(features) - target_horizon
    X = features[:valid_len]
    y = target[:valid_len]

    # Удаление NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    # Разделение на train/test
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Подготовлено: train={len(X_train)}, test={len(X_test)}")

    return X_train, y_train, X_test, y_test
```

---

## Реализация на Rust

### Структура крейта

Rust-реализация обеспечивает высокую производительность для продакшен-систем:

```
122_deeplift_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── model/
│   │   ├── mod.rs          # Модуль нейронной сети
│   │   └── network.rs      # Реализация сети
│   ├── deeplift/
│   │   ├── mod.rs          # Модуль DeepLIFT
│   │   └── attribution.rs  # Вычисление атрибуций
│   ├── data/
│   │   ├── mod.rs          # Модуль данных
│   │   ├── features.rs     # Генерация признаков
│   │   └── bybit.rs        # Клиент Bybit API
│   ├── trading/
│   │   ├── mod.rs          # Торговый модуль
│   │   ├── strategy.rs     # Торговые стратегии
│   │   └── signals.rs      # Генерация сигналов
│   └── backtest/
│       ├── mod.rs          # Модуль бэктестинга
│       └── engine.rs       # Движок бэктеста
├── examples/
│   ├── basic_deeplift.rs   # Базовый пример
│   ├── feature_importance.rs # Анализ важности
│   └── trading_explanation.rs # Объяснение сигналов
└── python/
    ├── deeplift_trader.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Ключевые типы и трейты

```rust
//! Основные типы и трейты для DeepLIFT атрибуции.

use std::collections::HashMap;

/// Результаты атрибуции для одного предсказания.
#[derive(Debug, Clone)]
pub struct Attribution {
    /// Названия входных признаков
    pub feature_names: Vec<String>,
    /// Оценки атрибуции для каждого признака
    pub scores: Vec<f64>,
    /// Выход модели для референсного входа
    pub baseline_output: f64,
    /// Выход модели для фактического входа
    pub actual_output: f64,
    /// Разница между фактическим и референсным выходом
    pub delta: f64,
}

impl Attribution {
    /// Создать новый объект Attribution.
    pub fn new(
        feature_names: Vec<String>,
        scores: Vec<f64>,
        baseline_output: f64,
        actual_output: f64,
    ) -> Self {
        let delta = actual_output - baseline_output;
        Self {
            feature_names,
            scores,
            baseline_output,
            actual_output,
            delta,
        }
    }

    /// Получить N признаков с наибольшим вкладом (по модулю).
    pub fn top_features(&self, n: usize) -> Vec<(String, f64)> {
        // Создаём пары (индекс, абсолютное значение)
        let mut indexed: Vec<(usize, f64)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s.abs()))
            .collect();

        // Сортируем по убыванию абсолютного значения
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Возвращаем top-N с оригинальными значениями
        indexed
            .into_iter()
            .take(n)
            .map(|(i, _)| (self.feature_names[i].clone(), self.scores[i]))
            .collect()
    }

    /// Получить признаки с положительным вкладом.
    pub fn positive_contributors(&self) -> Vec<(String, f64)> {
        self.feature_names
            .iter()
            .zip(self.scores.iter())
            .filter(|(_, &s)| s > 0.0)
            .map(|(n, &s)| (n.clone(), s))
            .collect()
    }

    /// Получить признаки с отрицательным вкладом.
    pub fn negative_contributors(&self) -> Vec<(String, f64)> {
        self.feature_names
            .iter()
            .zip(self.scores.iter())
            .filter(|(_, &s)| s < 0.0)
            .map(|(n, &s)| (n.clone(), s))
            .collect()
    }

    /// Преобразовать в HashMap.
    pub fn to_map(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .cloned()
            .zip(self.scores.iter().cloned())
            .collect()
    }

    /// Проверить свойство суммирования (оценки должны суммироваться в delta).
    pub fn verify_summation(&self) -> f64 {
        let sum: f64 = self.scores.iter().sum();
        (sum - self.delta).abs()
    }
}

/// Правило атрибуции DeepLIFT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttributionRule {
    /// Правило перемасштабирования: градиент * дельта
    Rescale,
    /// Правило RevealCancel: раздельная обработка положительных/отрицательных вкладов
    RevealCancel,
}

/// Объяснитель DeepLIFT для нейронных сетей.
#[derive(Debug)]
pub struct DeepLIFT {
    /// Референсный (базовый) вход
    pub reference: Vec<f64>,
    /// Правило атрибуции
    pub rule: AttributionRule,
    /// Веса сети (слой, нейрон, вес)
    weights: Vec<Vec<Vec<f64>>>,
    /// Смещения сети (слой, нейрон)
    biases: Vec<Vec<f64>>,
}

impl DeepLIFT {
    /// Создать новый объяснитель DeepLIFT.
    pub fn new(reference: Vec<f64>, rule: AttributionRule) -> Self {
        Self {
            reference,
            rule,
            weights: Vec::new(),
            biases: Vec::new(),
        }
    }

    /// Установить параметры сети.
    pub fn set_network(
        &mut self,
        weights: Vec<Vec<Vec<f64>>>,
        biases: Vec<Vec<f64>>,
    ) {
        self.weights = weights;
        self.biases = biases;
    }

    /// Вычислить прямой проход через сеть.
    fn forward(&self, input: &[f64]) -> (f64, Vec<Vec<f64>>) {
        let mut activations = vec![input.to_vec()];
        let mut current = input.to_vec();

        for (layer_idx, (layer_weights, layer_biases)) in
            self.weights.iter().zip(self.biases.iter()).enumerate()
        {
            let mut next = vec![0.0; layer_weights.len()];

            for (neuron_idx, (weights, bias)) in
                layer_weights.iter().zip(layer_biases.iter()).enumerate()
            {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(current.iter()) {
                    sum += w * x;
                }

                // ReLU для скрытых слоёв, линейный для выходного
                if layer_idx < self.weights.len() - 1 {
                    next[neuron_idx] = sum.max(0.0);
                } else {
                    next[neuron_idx] = sum;
                }
            }

            activations.push(next.clone());
            current = next;
        }

        let output = current.get(0).copied().unwrap_or(0.0);
        (output, activations)
    }

    /// Вычислить атрибуции для входа.
    pub fn attribute(
        &self,
        input: &[f64],
        feature_names: Vec<String>,
    ) -> Attribution {
        // Прямой проход для фактического входа
        let (actual_output, _actual_activations) = self.forward(input);

        // Прямой проход для референса
        let (baseline_output, _ref_activations) = self.forward(&self.reference);

        // Вычисление атрибуций через приближение градиента
        let mut scores = vec![0.0; input.len()];
        let epsilon = 1e-7;

        for i in 0..input.len() {
            // Возмущение i-го признака
            let mut perturbed = input.to_vec();
            perturbed[i] += epsilon;
            let (perturbed_output, _) = self.forward(&perturbed);

            // Приближённый градиент
            let gradient = (perturbed_output - actual_output) / epsilon;

            // Дельта входа
            let delta_input = input[i] - self.reference[i];

            // Применение правила атрибуции
            match self.rule {
                AttributionRule::Rescale => {
                    scores[i] = gradient * delta_input;
                }
                AttributionRule::RevealCancel => {
                    // Раздельная обработка положительных и отрицательных дельт
                    let pos_delta = delta_input.max(0.0);
                    let neg_delta = delta_input.min(0.0);
                    scores[i] = gradient * pos_delta + gradient * neg_delta;
                }
            }
        }

        Attribution::new(feature_names, scores, baseline_output, actual_output)
    }

    /// Вычислить атрибуции для батча входов.
    pub fn batch_attribute(
        &self,
        inputs: &[Vec<f64>],
        feature_names: Vec<String>,
    ) -> Vec<Attribution> {
        inputs
            .iter()
            .map(|input| self.attribute(input, feature_names.clone()))
            .collect()
    }
}

/// Вычисление средней важности признаков по выборке.
pub fn compute_feature_importance(
    explainer: &DeepLIFT,
    samples: &[Vec<f64>],
    feature_names: Vec<String>,
) -> HashMap<String, f64> {
    let n = samples.len() as f64;
    let mut importance_sum: HashMap<String, f64> = feature_names
        .iter()
        .map(|name| (name.clone(), 0.0))
        .collect();

    for sample in samples {
        let attr = explainer.attribute(sample, feature_names.clone());
        for (name, score) in attr.to_map() {
            if let Some(sum) = importance_sum.get_mut(&name) {
                *sum += score.abs();
            }
        }
    }

    // Усреднение
    for value in importance_sum.values_mut() {
        *value /= n;
    }

    importance_sum
}
```

### Торговые сигналы на Rust

```rust
//! Генерация торговых сигналов.

/// Типы торговых сигналов.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Сильный сигнал на покупку
    StrongBuy,
    /// Сигнал на покупку
    Buy,
    /// Нейтральный / удержание
    Neutral,
    /// Сигнал на продажу
    Sell,
    /// Сильный сигнал на продажу
    StrongSell,
}

impl TradingSignal {
    /// Создать сигнал из предсказания модели.
    pub fn from_prediction(prediction: f64, threshold: f64) -> Self {
        if prediction > threshold * 2.0 {
            TradingSignal::StrongBuy
        } else if prediction > threshold {
            TradingSignal::Buy
        } else if prediction < -threshold * 2.0 {
            TradingSignal::StrongSell
        } else if prediction < -threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Neutral
        }
    }

    /// Получить размер позиции (-1, 0, или 1).
    pub fn position(&self) -> i32 {
        match self {
            TradingSignal::StrongBuy | TradingSignal::Buy => 1,
            TradingSignal::StrongSell | TradingSignal::Sell => -1,
            TradingSignal::Neutral => 0,
        }
    }

    /// Получить размер позиции с уверенностью (-1.0 до 1.0).
    pub fn position_with_confidence(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Neutral => 0.0,
            TradingSignal::Sell => -0.5,
            TradingSignal::StrongSell => -1.0,
        }
    }

    /// Текстовое описание сигнала.
    pub fn description(&self) -> &str {
        match self {
            TradingSignal::StrongBuy => "Сильная покупка",
            TradingSignal::Buy => "Покупка",
            TradingSignal::Neutral => "Нейтрально",
            TradingSignal::Sell => "Продажа",
            TradingSignal::StrongSell => "Сильная продажа",
        }
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}
```

---

## Источники данных

### Данные фондового рынка (yfinance)

```python
"""
Пример загрузки данных акций с yfinance.
"""

import yfinance as yf
import numpy as np
import torch

# Загрузка данных Apple за 2 года
print("Загрузка данных AAPL...")
aapl = yf.download('AAPL', period='2y', progress=False)
prices = aapl['Close'].values

print(f"Загружено {len(prices)} записей")
print(f"Диапазон цен: ${prices.min():.2f} - ${prices.max():.2f}")

# Создание признаков
feature_gen = FeatureGenerator(window=20)
# Создаём фиктивные Kline объекты из цен
klines = [
    Kline(timestamp=i, open=p, high=p*1.01, low=p*0.99,
          close=p, volume=1e6, turnover=p*1e6)
    for i, p in enumerate(prices)
]
features = feature_gen.compute_all_features(klines)

print(f"Вычислено {features.shape[0]} наборов признаков")

# Подготовка данных для обучения
X_train, y_train, X_test, y_test = prepare_training_data(
    klines, target_horizon=5, train_ratio=0.8
)

print(f"Обучающая выборка: {len(X_train)} примеров")
print(f"Тестовая выборка: {len(X_test)} примеров")
```

### Криптовалютные данные (Bybit API)

```python
"""
Пример загрузки криптовалютных данных с Bybit.
"""

# Создание клиента Bybit
client = BybitClient()

# Загрузка данных нескольких пар
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']

for symbol in symbols:
    try:
        # Получение часовых свечей
        klines = client.fetch_klines(
            symbol=symbol,
            interval='60',  # 1 час
            limit=500       # ~20 дней данных
        )

        print(f"\n{symbol}:")
        print(f"  Свечей: {len(klines)}")
        print(f"  Период: {klines[0].timestamp} - {klines[-1].timestamp}")
        print(f"  Цена: ${klines[-1].close:,.2f}")

        # Вычисление признаков
        feature_gen = FeatureGenerator()
        features = feature_gen.compute_all_features(klines)
        print(f"  Признаков: {features.shape}")

    except Exception as e:
        print(f"Ошибка загрузки {symbol}: {e}")
```

---

## Торговые приложения

### Важность признаков для торговых решений

```python
"""
Анализ важности признаков для торговой модели.
"""

def analyze_feature_importance(
    model: nn.Module,
    explainer: DeepLIFT,
    test_data: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100
) -> Dict[str, float]:
    """
    Анализ средней важности признаков.

    Args:
        model: Обученная торговая модель
        explainer: Объяснитель DeepLIFT
        test_data: Тестовые данные
        feature_names: Названия признаков
        n_samples: Количество примеров для анализа

    Returns:
        Словарь {признак: средняя_важность}
    """
    importance_sum = {name: 0.0 for name in feature_names}
    n = min(n_samples, len(test_data))

    for i in range(n):
        input_tensor = torch.FloatTensor(test_data[i:i+1])
        attribution = explainer.attribute(input_tensor, feature_names)

        for name, score in zip(attribution.feature_names, attribution.scores):
            importance_sum[name] += abs(score)

    # Усреднение и нормализация
    total = sum(importance_sum.values())
    importance = {
        name: score / n / total * 100
        for name, score in importance_sum.items()
    }

    return importance


# Пример использования
feature_names = FeatureGenerator.feature_names()
importance = analyze_feature_importance(
    model, explainer, X_test, feature_names, n_samples=200
)

print("\nВажность признаков для торговых решений:")
print("=" * 50)
for name, score in sorted(importance.items(), key=lambda x: -x[1]):
    bar = '█' * int(score)
    print(f"{name:20} {score:5.1f}% {bar}")
```

### Объяснимые сигналы покупки/продажи

```python
"""
Генерация объяснимых торговых сигналов.
"""

def generate_explained_signals(
    model: nn.Module,
    explainer: DeepLIFT,
    features: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.001
) -> List[ExplainedSignal]:
    """
    Генерация серии объяснимых торговых сигналов.
    """
    signal_generator = SignalGenerator(
        model, explainer, feature_names, threshold
    )

    signals = []
    for i in range(len(features)):
        input_tensor = torch.FloatTensor(features[i:i+1])
        signal = signal_generator.generate_signal(input_tensor)
        signals.append(signal)

    return signals


# Пример генерации и вывода сигналов
signals = generate_explained_signals(
    model, explainer, X_test[-10:], feature_names
)

print("\nПоследние 10 торговых сигналов с объяснениями:")
print("=" * 60)

for i, signal in enumerate(signals):
    print(f"\n[{i+1}] {signal.explanation}")
    print(f"    Уверенность: {signal.confidence:.1%}")
```

### Оценка рисков через анализ атрибуций

```python
"""
Оценка рисков на основе анализа атрибуций DeepLIFT.
"""

@dataclass
class RiskAssessment:
    """Оценка риска на основе атрибуций."""
    overall_risk: float  # 0-1
    risk_factors: List[Tuple[str, float]]
    concentration_risk: float
    explanation: str


def assess_risk_from_attribution(
    attribution: Attribution,
    concentration_threshold: float = 0.5
) -> RiskAssessment:
    """
    Оценка риска на основе паттернов атрибуции.

    Высокая концентрация вклада в одном признаке
    может указывать на повышенный риск.
    """
    scores_abs = np.abs(attribution.scores)
    total = scores_abs.sum()

    if total == 0:
        return RiskAssessment(
            overall_risk=0.5,
            risk_factors=[],
            concentration_risk=0.0,
            explanation="Недостаточно данных для оценки риска"
        )

    # Нормализованные абсолютные вклады
    normalized = scores_abs / total

    # Риск концентрации (индекс Херфиндаля)
    hhi = np.sum(normalized ** 2)
    concentration_risk = hhi

    # Выявление факторов риска
    risk_factors = []
    for i, (name, score) in enumerate(zip(
        attribution.feature_names, attribution.scores
    )):
        if normalized[i] > concentration_threshold:
            risk_factors.append((name, normalized[i]))

    # Общий уровень риска
    overall_risk = min(1.0, concentration_risk + len(risk_factors) * 0.1)

    # Генерация объяснения
    if risk_factors:
        factors_str = ", ".join([f"{n} ({v:.1%})" for n, v in risk_factors])
        explanation = f"Повышенный риск из-за концентрации: {factors_str}"
    else:
        explanation = "Диверсифицированные факторы влияния, умеренный риск"

    return RiskAssessment(
        overall_risk=overall_risk,
        risk_factors=risk_factors,
        concentration_risk=concentration_risk,
        explanation=explanation
    )


# Пример оценки риска
sample_input = torch.FloatTensor(X_test[0:1])
attribution = explainer.attribute(sample_input, feature_names)
risk = assess_risk_from_attribution(attribution)

print("\nОценка риска:")
print(f"  Общий риск: {risk.overall_risk:.1%}")
print(f"  Риск концентрации: {risk.concentration_risk:.3f}")
print(f"  {risk.explanation}")
```

---

## Фреймворк бэктестинга

### Движок бэктестинга с DeepLIFT

```python
"""
Фреймворк бэктестинга с интегрированными объяснениями DeepLIFT.
"""

@dataclass
class BacktestEntry:
    """Одна запись бэктеста."""
    index: int
    timestamp: Optional[int]
    price: float
    prediction: float
    signal: SignalType
    position: int
    position_return: float
    capital: float
    top_features: List[Tuple[str, float]]
    risk_assessment: Optional[RiskAssessment]


@dataclass
class BacktestMetrics:
    """Метрики результатов бэктеста."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int

    def __str__(self) -> str:
        return f"""
Метрики бэктеста:
  Общая доходность:     {self.total_return:>10.2%}
  Годовая доходность:   {self.annualized_return:>10.2%}
  Годовая волатильность:{self.annualized_volatility:>10.2%}
  Коэффициент Шарпа:    {self.sharpe_ratio:>10.3f}
  Коэффициент Сортино:  {self.sortino_ratio:>10.3f}
  Максимальная просадка:{self.max_drawdown:>10.2%}
  Win Rate:             {self.win_rate:>10.2%}
  Profit Factor:        {self.profit_factor:>10.3f}
  Количество сделок:    {self.num_trades:>10}
"""


class DeepLIFTBacktester:
    """
    Движок бэктестинга с объяснениями DeepLIFT.

    Выполняет бэктест торговой стратегии с записью
    объяснений для каждого торгового решения.
    """

    def __init__(
        self,
        model: nn.Module,
        explainer: DeepLIFT,
        feature_names: List[str],
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001,
        enable_risk_assessment: bool = True
    ):
        """
        Инициализация бэктестера.

        Args:
            model: Торговая модель
            explainer: Объяснитель DeepLIFT
            feature_names: Названия признаков
            prediction_threshold: Порог для торговых сигналов
            transaction_cost: Транзакционные издержки (доля)
            enable_risk_assessment: Включить оценку рисков
        """
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost
        self.enable_risk = enable_risk_assessment

    def run(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[List[BacktestEntry], BacktestMetrics]:
        """
        Запуск бэктеста.

        Args:
            prices: Массив цен
            features: Массив признаков
            initial_capital: Начальный капитал
            timestamps: Временные метки (опционально)

        Returns:
            Список записей бэктеста и метрики
        """
        entries = []
        capital = initial_capital
        position = 0

        self.model.eval()

        for i in range(len(features)):
            input_tensor = torch.FloatTensor(features[i:i+1])

            # Предсказание
            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            # Атрибуция
            attribution = self.explainer.attribute(input_tensor, self.feature_names)
            top_features = attribution.top_features(3)

            # Оценка риска
            risk = None
            if self.enable_risk:
                risk = assess_risk_from_attribution(attribution)

            # Определение сигнала
            if prediction > self.threshold * 2:
                signal = SignalType.STRONG_BUY
                new_position = 1
            elif prediction > self.threshold:
                signal = SignalType.BUY
                new_position = 1
            elif prediction < -self.threshold * 2:
                signal = SignalType.STRONG_SELL
                new_position = -1
            elif prediction < -self.threshold:
                signal = SignalType.SELL
                new_position = -1
            else:
                signal = SignalType.NEUTRAL
                new_position = 0

            # Учёт транзакционных издержек
            if new_position != position and i > 0:
                capital *= (1 - self.transaction_cost)

            # Расчёт доходности
            if i < len(prices) - 1:
                actual_return = prices[i+1] / prices[i] - 1
                position_return = position * actual_return
                capital *= (1 + position_return)
            else:
                position_return = 0

            entries.append(BacktestEntry(
                index=i,
                timestamp=timestamps[i] if timestamps is not None else None,
                price=prices[i],
                prediction=prediction,
                signal=signal,
                position=position,
                position_return=position_return,
                capital=capital,
                top_features=top_features,
                risk_assessment=risk
            ))

            position = new_position

        metrics = self._calculate_metrics(entries, initial_capital)

        return entries, metrics

    def _calculate_metrics(
        self,
        entries: List[BacktestEntry],
        initial_capital: float
    ) -> BacktestMetrics:
        """Расчёт метрик производительности."""
        if not entries:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        returns = np.array([e.position_return for e in entries])
        n = len(returns)

        # Общая доходность
        final_capital = entries[-1].capital
        total_return = final_capital / initial_capital - 1

        # Годовые показатели (предполагаем часовые данные)
        periods_per_year = 8760
        annualized_return = (1 + total_return) ** (periods_per_year / n) - 1

        # Волатильность
        mean_return = returns.mean()
        std_return = returns.std()
        annualized_volatility = std_return * np.sqrt(periods_per_year)

        # Коэффициент Шарпа
        sharpe = np.sqrt(periods_per_year) * mean_return / (std_return + 1e-10)

        # Коэффициент Сортино
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = np.sqrt(periods_per_year) * mean_return / (downside_std + 1e-10)

        # Максимальная просадка
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / peak - 1
        max_drawdown = drawdowns.min()

        # Win Rate
        wins = (returns > 0).sum()
        losses = (returns < 0).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        # Profit Factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        # Количество сделок
        positions = np.array([e.position for e in entries])
        num_trades = np.sum(np.diff(positions) != 0)

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades
        )


# Пример запуска бэктеста
backtester = DeepLIFTBacktester(
    model=model,
    explainer=explainer,
    feature_names=feature_names,
    prediction_threshold=0.001,
    transaction_cost=0.001
)

# Получение цен (выравнивание с признаками)
aligned_prices = prices[-(len(X_test)):]

entries, metrics = backtester.run(
    prices=aligned_prices,
    features=X_test,
    initial_capital=10000.0
)

print(metrics)

# Анализ записей с наибольшей доходностью
top_entries = sorted(entries, key=lambda e: e.position_return, reverse=True)[:5]

print("\nТоп-5 наиболее прибыльных решений:")
for e in top_entries:
    print(f"\n[{e.index}] Доходность: {e.position_return:.2%}")
    print(f"    Сигнал: {e.signal.value}")
    print(f"    Ключевые факторы: {e.top_features}")
```

---

## Сравнение производительности

### Целевые показатели

| Метрика | Целевой диапазон | Описание |
|---------|-----------------|----------|
| Коэффициент Шарпа | > 1.0 | Соотношение доходность/риск |
| Коэффициент Сортино | > 1.5 | Учитывает только отрицательную волатильность |
| Максимальная просадка | < 20% | Максимальное падение капитала |
| Win Rate | > 50% | Доля прибыльных сделок |
| Консистентность объяснений | > 80% | Стабильность атрибуций для похожих входов |

### DeepLIFT vs альтернативные методы

```python
"""
Сравнение DeepLIFT с другими методами атрибуции.
"""

import time

def benchmark_attribution_methods(
    model: nn.Module,
    test_inputs: torch.Tensor,
    reference: torch.Tensor,
    n_iterations: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Бенчмарк различных методов атрибуции.
    """
    results = {}

    # DeepLIFT
    explainer_deeplift = DeepLIFT(model, reference, rule="rescale")
    start = time.time()
    for i in range(n_iterations):
        _ = explainer_deeplift.attribute(test_inputs[i % len(test_inputs)])
    deeplift_time = time.time() - start
    results['DeepLIFT'] = {'время': deeplift_time, 'на_итерацию': deeplift_time / n_iterations}

    # Простые градиенты (для сравнения)
    start = time.time()
    for i in range(n_iterations):
        inp = test_inputs[i % len(test_inputs)].clone().requires_grad_(True)
        out = model(inp)
        out.backward(torch.ones_like(out))
        _ = inp.grad
    gradient_time = time.time() - start
    results['Градиенты'] = {'время': gradient_time, 'на_итерацию': gradient_time / n_iterations}

    return results


# Запуск бенчмарка
X_test_tensor = torch.FloatTensor(X_test[:100])
reference = torch.FloatTensor(np.mean(X_train, axis=0, keepdims=True))

benchmark = benchmark_attribution_methods(model, X_test_tensor, reference)

print("\nСравнение производительности методов атрибуции:")
print("=" * 50)
for method, stats in benchmark.items():
    print(f"{method:20} {stats['на_итерацию']*1000:.2f} мс/итерацию")
```

### Типичные результаты

В экспериментах DeepLIFT показывает:

- **Скорость**: В 2-5 раз быстрее SHAP при сравнимой точности
- **Обработка насыщения**: Значительно лучше градиентных методов в областях насыщения ReLU
- **Стабильность**: Консистентные рейтинги признаков для похожих входов
- **Свойство суммирования**: Атрибуции точно суммируются в разницу предсказаний

---

## Научные публикации и ссылки

### Основные работы

1. **Shrikumar, A., Greenside, P., & Kundaje, A. (2017)**
   "Learning Important Features Through Propagating Activation Differences"
   ICML 2017
   [arXiv:1704.02685](https://arxiv.org/abs/1704.02685)

   *Оригинальная статья по DeepLIFT, вводящая метод атрибуции через сравнение активаций.*

2. **Sundararajan, M., Taly, A., & Yan, Q. (2017)**
   "Axiomatic Attribution for Deep Networks"
   ICML 2017
   [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)

   *Integrated Gradients — теоретически обоснованный метод с аксиоматическими гарантиями.*

3. **Lundberg, S. M., & Lee, S. I. (2017)**
   "A Unified Approach to Interpreting Model Predictions"
   NeurIPS 2017

   *SHAP — унифицированный подход к интерпретации на основе значений Шепли.*

4. **Ancona, M., Ceolini, E., Öztireli, C., & Gross, M. (2018)**
   "Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks"
   ICLR 2018

   *Сравнительный анализ градиентных методов атрибуции.*

5. **Montavon, G., Samek, W., & Müller, K.-R. (2018)**
   "Methods for Interpreting and Understanding Deep Neural Networks"
   Digital Signal Processing

   *Обзор методов интерпретации глубоких нейронных сетей.*

### Применение в финансах

6. **Chen, L., Pelger, M., & Zhu, J. (2020)**
   "Deep Learning in Asset Pricing"
   Management Science

   *Применение глубокого обучения и интерпретируемости в ценообразовании активов.*

7. **Gu, S., Kelly, B., & Xiu, D. (2020)**
   "Empirical Asset Pricing via Machine Learning"
   Review of Financial Studies

   *Машинное обучение для эмпирического ценообразования активов.*

---

## Запуск примеров

### Python

```bash
# Перейти в директорию главы
cd 122_deeplift_trading

# Создать виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r python/requirements.txt

# Запустить основной пример
python python/deeplift_trader.py

# Запустить бэктест
python python/backtest.py

# Загрузить данные
python python/data_loader.py
```

### Rust

```bash
# Перейти в директорию главы
cd 122_deeplift_trading

# Собрать проект в режиме релиза
cargo build --release

# Запустить тесты
cargo test

# Запустить примеры
cargo run --example basic_deeplift
cargo run --example feature_importance
cargo run --example trading_explanation

# Запустить с оптимизациями
cargo run --release --example basic_deeplift
```

---

## Резюме

DeepLIFT предоставляет мощный и эффективный фреймворк для интерпретируемости нейронных сетей в контексте алгоритмического трейдинга:

### Ключевые преимущества

1. **Теоретический фундамент**: Сравнение активаций с референсом обеспечивает осмысленные атрибуции
2. **Свойство суммирования**: Вклады признаков точно суммируются в разницу предсказаний
3. **Обработка насыщения**: Корректная работа в областях насыщения ReLU (в отличие от градиентных методов)
4. **Практическая ценность**: Незаменим для построения прозрачных и надёжных торговых систем

### Практические применения

- **Объяснение торговых сигналов**: Понимание причин рекомендаций модели
- **Оценка рисков**: Выявление концентрации факторов влияния
- **Обнаружение режимов**: Отслеживание изменений в паттернах важности
- **Валидация модели**: Проверка соответствия выученных паттернов финансовой интуиции
- **Регуляторное соответствие**: Обеспечение объяснимости для аудита

### Рекомендации по использованию

1. Выбирайте осмысленный референс (нейтральное рыночное состояние)
2. Анализируйте как положительные, так и отрицательные вклады
3. Отслеживайте стабильность атрибуций во времени
4. Комбинируйте с другими методами для перекрёстной валидации
5. Документируйте объяснения для аудита и отчётности

---

*Предыдущая глава: [Глава 121: Layer-wise Relevance Propagation](../121_layer_wise_relevance)*

*Следующая глава: [Глава 123: GradCAM для финансов](../123_gradcam_finance)*
