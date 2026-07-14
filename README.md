# VK Russian Vision-Language Modeling

Проект по адаптации компактной vision-language модели под русскоязычный visual question answering на открытых данных VK / DeepVK.

Основная идея: взять готовую мультимодальную модель `Qwen/Qwen2.5-VL-3B-Instruct`, дообучить её через LoRA на русской версии GQA и сравнить качество до и после адаптации.

## Кратко

- **Базовая модель:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Основной датасет:** `deepvk/GQA-ru`
- **Метод обучения:** LoRA only over language model attention projections
- **Vision encoder:** frozen
- **Objective:** completion-only loss
- **Финальная модель:** E3, LoRA на 8000 train examples
- **Финальный результат на GQA testdev 2000:** `40.20% → 52.65%`, прирост `+12.45 п.п.`

## Задача

Цель проекта — проверить, можно ли недорогой LoRA-адаптацией улучшить качество компактной VLM на русскоязычном VQA.

Модель получает изображение и вопрос на русском языке, после чего должна дать короткий ответ, обычно одним словом или короткой фразой.

Пример промпта:

```text
<image>
Кто в рубашке?
Ответь одним словом.
```

## Данные

Используется датасет `deepvk/GQA-ru`.

Основные части:

- `train_balanced_images`
- `train_balanced_instructions`
- `testdev_balanced_images`
- `testdev_balanced_instructions`

Инструкции соединяются с изображениями по ключу:

```text
instruction.imageId == image.id
```

Для model selection был сделан отдельный image-level split внутри train:

- train: 38 018 questions
- validation: 1 982 questions
- image overlap между train и validation: 0

Это важно, потому что обычный question-level split мог бы дать leakage через одинаковые изображения.

Аудит данных и split лежат в:

```text
results/data_audit.json
data/splits/gqa_image_split.json
docs/experiment_protocol.md
```

## Метод

Используется LoRA-адаптация только языковой части модели.

LoRA target modules:

```text
.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$
```

Ключевые настройки:

| Параметр | Значение |
|---|---:|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Precision | bf16 |
| Vision encoder | frozen |
| Train objective | completion-only loss |
| Effective batch size | 8 |

Почему completion-only loss: при full-sequence loss модель училась предсказывать не только ответ, но и служебную часть диалога, что ухудшало downstream exact match. Completion-only training оказался заметно стабильнее.

## Эксперименты

### Validation split

| Эксперимент | Train examples | Validation examples | Accuracy |
|---|---:|---:|---:|
| Base Qwen2.5-VL-3B-Instruct | 0 | 1982 | 0.4369 |
| E2 LoRA | 2000 | 1982 | 0.5721 |
| E3 LoRA | 8000 | 1982 | 0.5984 |

Paired comparison E2 vs E3:

| Метрика | Значение |
|---|---:|
| E2 correct | 1134 / 1982 |
| E3 correct | 1186 / 1982 |
| Delta | +0.0262 |
| E2 only | 86 |
| E3 only | 138 |
| McNemar exact p | 0.000623 |

E3 улучшает E2 статистически значимо, поэтому E3 выбран как финальная модель.

### Testdev 2000

Финальная оценка проводилась на фиксированной выборке из 2000 примеров GQA testdev.

| Эксперимент | Train examples | Testdev examples | Accuracy |
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

```text
Base correct: 804 / 2000
E3 correct:   1053 / 2000
Delta:        +0.1245
```

## Результаты по semantic types

На validation E3 улучшает baseline по всем semantic types:

| Type | Base | E3 | Delta |
|---|---:|---:|---:|
| rel | 0.344 | 0.471 | +0.127 |
| attr | 0.502 | 0.669 | +0.168 |
| obj | 0.624 | 0.878 | +0.255 |
| cat | 0.461 | 0.594 | +0.133 |
| global | 0.305 | 0.576 | +0.271 |

Наиболее сильный прирост виден в `obj` и `global`, но улучшение есть и для relation questions, которые остаются более сложным типом вопросов.

## Структура репозитория

```text
configs/                         experiment configs
src/data/                        dataset audit and split scripts
src/training/                    LoRA training scripts
src/evaluation/                  evaluation and paired comparison scripts
src/inference/                   smoke inference
data/splits/                     fixed split and sample ids
results/                         metrics, predictions, comparisons
artifacts/                       saved LoRA adapters
docs/                            protocol and final report
examples/                        example inputs/outputs
```

Ключевые файлы:

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

## Установка

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Для GPU-обучения использовались CUDA, bf16 и PyTorch с поддержкой CUDA.

## Воспроизведение

### 1. Аудит данных и split

```bash
python -m src.data.audit_and_split
```

### 2. Baseline evaluation

```bash
python -m src.evaluation.evaluate_gqa_2000 --mode base
```

### 3. E3 training

```bash
python -m src.training.train_8000_completion_only
```

Результат сохраняется в:

```text
outputs/train_8000_completion_only
```

Финальный сохранённый адаптер лежит в:

```text
artifacts/train_8000_completion_only
```

### 4. E3 validation evaluation

```bash
python -m src.evaluation.evaluate_validation_1982_e3 --mode adapter
```

### 5. E3 testdev evaluation

```bash
python -m src.evaluation.evaluate_gqa_2000_e3 --mode adapter
```

### 6. Paired comparisons

```bash
python -m src.evaluation.compare_validation_base_vs_e3
python -m src.evaluation.compare_validation_e2_vs_e3
python -m src.evaluation.compare_gqa_2000_base_vs_e3
```

## Артефакты

Финальный LoRA-адаптер:

```text
artifacts/train_8000_completion_only/
  adapter_config.json
  adapter_model.safetensors
  train_metrics.json
```

Основные результаты:

```text
results/validation_1982/
results/validation_1982_e3/
results/base_2000/
results/adapter_8000_testdev_2000/
```

## Ограничения

- Используется exact match, поэтому некоторые семантически близкие ответы считаются ошибками, например `кот` vs `кошка` или `машина` vs `автомобиль`.
- Обучение проводилось на подвыборках GQA-ru, а не на полном train split.
- Vision encoder был заморожен, поэтому адаптация в основном улучшает языковую часть ответа и alignment с русским форматом.
- Финальная модель проверялась на GQA-ru, но не была полноценно протестирована на независимых русскоязычных VLM-бенчмарках вроде MMBench-ru.

## Вывод

LoRA-адаптация `Qwen2.5-VL-3B-Instruct` на `GQA-ru` существенно улучшает качество русскоязычного VQA.

Финальная модель E3, обученная на 8000 примерах, улучшает качество на фиксированной testdev-выборке:

```text
40.20% → 52.65%
```

Это показывает, что даже сравнительно небольшой объём русскоязычной VQA-разметки может заметно улучшить поведение compact VLM на русском языке без полного fine-tuning модели.
