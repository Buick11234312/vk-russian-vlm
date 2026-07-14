# Финальный отчёт: адаптация VLM под русскоязычный Visual Question Answering

## 1. Цель проекта

Цель проекта — адаптировать компактную vision-language модель под русскоязычный visual question answering и проверить, насколько LoRA-дообучение на открытых русскоязычных VQA-данных улучшает качество модели.

В качестве базовой модели была выбрана `Qwen/Qwen2.5-VL-3B-Instruct`. В качестве основного датасета использовался `deepvk/GQA-ru`.

Формулировка задачи:

> По изображению и вопросу на русском языке модель должна дать короткий текстовый ответ, обычно одним словом или короткой фразой.

Пример:

```text
<image>
Кто в рубашке?
Ответь одним словом.
```

Ожидаемый ответ:

```text
парень
```

## 2. Данные

В проекте использовался датасет `deepvk/GQA-ru`.

Основные конфигурации датасета:

- `train_balanced_images`
- `train_balanced_instructions`
- `testdev_balanced_images`
- `testdev_balanced_instructions`

Инструкции соединяются с изображениями по ключу:

```text
instruction.imageId == image.id
```

### 2.1. Аудит датасета

В ходе аудита было проверено:

- количество train и testdev примеров;
- наличие изображений для вопросов;
- пересечение изображений между train и testdev;
- структура полей в instruction split;
- структура `semantic` и `types`, используемых для анализа ошибок.

Результаты аудита сохранены в:

```text
results/data_audit.json
docs/experiment_protocol.md
```

Ключевые результаты аудита:

| Split | Questions |
|---|---:|
| GQA-ru train | 40 000 |
| GQA-ru testdev | 12 216 |

Пересечение изображений между train и testdev отсутствует.

### 2.2. Image-level validation split

Для model selection был создан отдельный validation split внутри train части.

Использовался именно image-level split, а не question-level split. Это важно, потому что в GQA несколько вопросов могут относиться к одному изображению. Если случайно разделить вопросы, но не изображения, модель может видеть одно и то же изображение и в train, и в validation, что приводит к утечке данных.

Итоговый split:

| Split | Questions |
|---|---:|
| Train | 38 018 |
| Validation | 1 982 |

Image overlap между train и validation равен 0.

Файл split:

```text
data/splits/gqa_image_split.json
```

## 3. Модель

Базовая модель:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Модель является компактной instruction-tuned vision-language моделью. В проекте она используется как base VLM, поверх которой обучается LoRA-адаптер.

## 4. Метод обучения

Использовалась LoRA-адаптация только языковой части модели. Vision encoder был заморожен.

### 4.1. LoRA target modules

LoRA применялась к attention projection слоям language model:

```text
.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$
```

Vision encoder не обучался.

### 4.2. Основные гиперпараметры

| Параметр | Значение |
|---|---:|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Precision | bf16 |
| Effective batch size | 8 |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Vision encoder | frozen |

### 4.3. Completion-only loss

Первый pilot experiment с full-sequence loss показал ухудшение exact match качества. Поэтому в финальных экспериментах использовался completion-only objective: loss считался только по ответу ассистента, а не по всему диалоговому шаблону.

Такой формат лучше соответствует целевой задаче: модель должна не реконструировать весь prompt, а выдавать короткий ответ на вопрос.

## 5. Evaluation protocol

Для оценки качества использовался exact match после простой нормализации:

- приведение к нижнему регистру;
- удаление пунктуации;
- сравнение нормализованного предсказания с нормализованным правильным ответом.

Prompt для evaluation:

```text
<image>
{question}
Ответь одним словом.
```

Генерация deterministic:

- `do_sample=False`
- короткий `max_new_tokens`

Метрика:

```text
accuracy = number_of_exact_matches / number_of_examples
```

Дополнительно использовался paired comparison между предсказаниями моделей:

- both correct;
- base only;
- new only;
- both wrong;
- McNemar exact p-value.

## 6. Эксперименты

### 6.1. E0: baseline

Baseline — исходная модель `Qwen/Qwen2.5-VL-3B-Instruct` без дообучения.

### 6.2. E1: pilot LoRA на 512 примерах

Первый успешный LoRA pilot был обучен на 512 train examples с completion-only loss.

Результат на fixed GQA testdev 2000:

| Model | Train examples | Accuracy |
|---|---:|---:|
| Base | 0 | 0.4020 |
| E1 LoRA | 512 | 0.4455 |

Прирост:

```text
+0.0435
```

### 6.3. E2: LoRA на 2000 примерах

E2 был обучен на 2000 train examples и оценён на held-out validation split.

Training result:

| Metric | Value |
|---|---:|
| Global step | 250 |
| Train loss | 0.559945 |
| Pre-train validation loss | 1.195966 |
| Post-train validation loss | 0.417449 |
| Peak CUDA memory | 10.87 GB |

Validation accuracy:

| Model | Validation examples | Accuracy |
|---|---:|---:|
| Base | 1982 | 0.4369 |
| E2 LoRA | 1982 | 0.5721 |

Прирост:

```text
+0.1352
```

Paired comparison Base vs E2:

| Metric | Value |
|---|---:|
| Both correct | 783 |
| Base only | 83 |
| E2 only | 351 |
| Both wrong | 765 |
| McNemar exact p | rounded to 0.000000 |

E2 заметно улучшил baseline и был использован как промежуточная сильная модель.

### 6.4. E3: LoRA на 8000 примерах

E3 был обучен на 8000 train examples с теми же основными настройками.

Training result:

| Metric | Value |
|---|---:|
| Global step | 1000 |
| Train loss | 0.458945 |
| Pre-train validation loss | 1.195966 |
| Post-train validation loss | 0.363149 |
| Peak CUDA memory | 11.28 GB |

Validation accuracy:

| Model | Validation examples | Accuracy |
|---|---:|---:|
| Base | 1982 | 0.4369 |
| E2 LoRA | 1982 | 0.5721 |
| E3 LoRA | 1982 | 0.5984 |

E3 улучшил E2:

| Metric | Value |
|---|---:|
| E2 correct | 1134 / 1982 |
| E3 correct | 1186 / 1982 |
| Delta | +0.0262 |
| E2 only | 86 |
| E3 only | 138 |
| McNemar exact p | 0.000623 |

Так как E3 статистически значимо улучшил E2, он был выбран как финальная модель.

## 7. Финальная оценка на testdev

Финальная оценка проводилась на фиксированной выборке из 2000 примеров GQA testdev.

| Model | Train examples | Testdev examples | Accuracy |
|---|---:|---:|---:|
| Base Qwen2.5-VL-3B-Instruct | 0 | 2000 | 0.4020 |
| E1 LoRA | 512 | 2000 | 0.4455 |
| E3 LoRA | 8000 | 2000 | 0.5265 |

Финальный прирост:

```text
Base → E3: 0.4020 → 0.5265
Delta:     +0.1245
```

Paired comparison Base vs E3:

| Metric | Value |
|---|---:|
| Base correct | 804 / 2000 |
| E3 correct | 1053 / 2000 |
| Delta | +0.1245 |

Таким образом, финальная модель улучшила baseline на 12.45 процентных пункта на testdev subset.

## 8. Анализ по semantic types

На validation split E3 улучшил baseline по всем semantic types.

| Type | Base | E3 | Delta |
|---|---:|---:|---:|
| rel | 0.344 | 0.471 | +0.127 |
| attr | 0.502 | 0.669 | +0.168 |
| obj | 0.624 | 0.878 | +0.255 |
| cat | 0.461 | 0.594 | +0.133 |
| global | 0.305 | 0.576 | +0.271 |

Наибольший прирост относительно baseline:

- `global`: +0.271
- `obj`: +0.255
- `attr`: +0.168

Relation questions остаются сложным классом, но и там есть заметный прирост:

```text
rel: 0.344 → 0.471
```

## 9. Примеры улучшений

Примеры, где E3 исправляет baseline:

| Question | Gold | Base | E3 |
|---|---|---|---|
| Что держит женщина? | мяч | Футбольный мяч | мяч |
| С какой стороны свеча? | слева | на стене | слева |
| Кто присматривает за ребенком? | женщина | Мать | женщина |
| Есть ли мужчины справа от девушки в поезде? | нет | Да | нет |
| Кто на скейтборде? | мальчик | Девушка | мальчик |

Модель после адаптации чаще отвечает в нужном формате и лучше подстраивается под русскоязычные короткие ответы.

## 10. Ошибки и ограничения

### 10.1. Exact match слишком строгий

Exact match может считать ошибками семантически близкие ответы:

```text
кот vs кошка
машина vs автомобиль
из металла vs металл
```

Поэтому реальное качество может быть несколько выше, чем exact match accuracy.

### 10.2. Ошибки остаются в пространственных вопросах

Некоторые ошибки связаны с left/right reasoning и relation questions:

```text
слева vs справа
перед vs за
над vs под
```

Это ожидаемо: такие вопросы требуют более точного grounding изображения.

### 10.3. Vision encoder был frozen

Так как vision encoder не обучался, адаптация в основном улучшает language head, формат ответа и alignment с русскими VQA-ответами. Это делает обучение дешёвым, но ограничивает потенциал улучшения визуального понимания.

### 10.4. Обучение не проводилось на полном train split

Финальная модель E3 была обучена на 8000 train examples, а не на всех 38 018 training questions. Полное обучение могло бы дать дополнительный прирост, но потребовало бы существенно больше GPU времени.

### 10.5. Не выполнена полноценная cross-benchmark evaluation

В проекте был рассмотрен MMBench-ru при аудите, но финальная оценка сфокусирована на GQA-ru. Для более сильного вывода о generalization стоило бы дополнительно оценить модель на MMBench-ru или другом независимом русскоязычном VLM benchmark.

## 11. Воспроизводимость

Ключевые скрипты:

```text
src/data/audit_and_split.py
src/training/train_2000_completion_only.py
src/training/train_8000_completion_only.py
src/evaluation/evaluate_validation_1982.py
src/evaluation/evaluate_validation_1982_e3.py
src/evaluation/evaluate_gqa_2000.py
src/evaluation/evaluate_gqa_2000_e3.py
src/evaluation/compare_validation_base_vs_e3.py
src/evaluation/compare_validation_e2_vs_e3.py
src/evaluation/compare_gqa_2000_base_vs_e3.py
```

Финальный адаптер:

```text
artifacts/train_8000_completion_only/
```

Финальные результаты:

```text
results/validation_1982_e3/
results/adapter_8000_testdev_2000/
```

## 12. Вывод

В проекте была проведена LoRA-адаптация `Qwen/Qwen2.5-VL-3B-Instruct` под русскоязычный VQA на датасете `deepvk/GQA-ru`.

Основной результат:

```text
Base testdev accuracy: 0.4020
E3 testdev accuracy:   0.5265
Improvement:           +0.1245
```

Также на held-out validation split качество выросло:

```text
0.4369 → 0.5984
```

Это показывает, что даже относительно небольшой объём русскоязычных VQA-данных и параметрически эффективная LoRA-адаптация могут заметно улучшить качество compact VLM на русском visual question answering.
